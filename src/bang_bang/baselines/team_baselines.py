"""Team-level game-outcome baselines.

Each baseline is a walk-forward predictor: given everything before game g,
it produces P(home win) for game g.

Baselines:
  1. home_floor: fixed P(home)=0.58 (the AGENTS.md floor)
  2. elo_538: ELO with MoV-scaled K, +100 home, +rest bonus
  3. netrtg_pyth: season-to-date ORtg/DRtg, regressed to league avg, Pythagorean
"""

from __future__ import annotations

import math

import polars as pl


# ---------------------------------------------------------------- ELO
class EloTracker:
    """538-style NBA ELO.

    K = 20 * (1 + (MoV - 10)/20), clamped to [10, 30]
    home advantage: +100 elo
    rest bonus: +30 per day beyond 3 days off, capped +150
    """

    def __init__(self, k_base: float = 20.0, home: float = 100.0):
        self.k_base = k_base
        self.home = home
        self.ratings: dict[int, float] = {}
        self.last_played: dict[int, pl.Date | None] = {}

    def _rating(self, team_id: int) -> float:
        return self.ratings.setdefault(team_id, 1500.0)

    @staticmethod
    def _rest_bonus(team_id: int, date, last: dict) -> float:
        prev = last.get(team_id)
        if prev is None:
            return 0.0
        days_off = (date - prev).days
        if days_off <= 3:
            return 0.0
        return min(30.0 * (days_off - 3), 150.0)

    def predict(self, home_id: int, away_id: int, date) -> float:
        exp_home = self._rating(home_id) + self.home + self._rest_bonus(home_id, date, self.last_played)
        exp_away = self._rating(away_id) + self._rest_bonus(away_id, date, self.last_played)
        return 1.0 / (1.0 + 10 ** (-(exp_home - exp_away) / 400.0))

    def update(self, home_id: int, away_id: int, home_pts: float, away_pts: float, date) -> None:
        pre_home = self._rating(home_id) + self.home + self._rest_bonus(home_id, date, self.last_played)
        pre_away = self._rating(away_id) + self._rest_bonus(away_id, date, self.last_played)
        expected = 1.0 / (1.0 + 10 ** (-(pre_home - pre_away) / 400.0))
        actual = 1.0 if home_pts > away_pts else (0.5 if home_pts == away_pts else 0.0)
        mov = abs(home_pts - away_pts)
        k = min(max(self.k_base * (1 + (mov - 10) / 20.0), 10.0), 30.0)
        self.ratings[home_id] = self._rating(home_id) + k * (actual - expected)
        self.ratings[away_id] = self._rating(away_id) + k * ((1 - actual) - (1 - expected))
        self.last_played[home_id] = date
        self.last_played[away_id] = date


def elo_predictions(games: pl.DataFrame) -> pl.DataFrame:
    """games: build.build_games output sorted by game_date.
    Returns games + elo_pred_home_win."""
    tr = EloTracker()
    preds = []
    for row in games.iter_rows(named=True):
        p = tr.predict(row["home_team_id"], row["away_team_id"], row["game_date"])
        preds.append(p)
        tr.update(row["home_team_id"], row["away_team_id"],
                  row["home_pts"], row["away_pts"], row["game_date"])
    return games.with_columns(pl.Series("elo_pred", preds))


# ------------------------------------------------- net rating pythagorean
def netrtg_predictions(games: pl.DataFrame) -> pl.DataFrame:
    """Season-to-date ORtg/DRtg, regressed to league average, Pythagorean win%.

    regression weight: 1 / (1 + 0.1 * (gp - 10)), floor 0.35 (like 538's SRS-ish shrink)
    Pythagorean exponent: 13.3 (empirical NBA fit)
    """
    rows = []
    s2d: dict[int, dict[str, float]] = {}  # team -> {poss, pts, opp_pts}

    for row in games.iter_rows(named=True):
        gid = row["game_id"]
        # predict first, then update
        preds = []
        for tid, drtg_src in ((row["home_team_id"], "away"), (row["away_team_id"], "home")):
            st = s2d.get(tid)
            if st is None or st["poss"] < 4 * 10:  # need ~10 games worth
                preds.append(0.5)
                continue
            ortg = st["pts"] / st["poss"] * 100
            drtg = st["opp_pts"] / st["poss"] * 100
            gp = st["poss"] / 100  # possessions / ~100 per game
            league_ortg, league_drtg = st["league_ortg"], st["league_drtg"]
            reg = 1.0 / (1.0 + 0.1 * max(gp - 10, 0))
            reg = max(reg, 0.35)
            adj_ortg = league_ortg + (ortg - league_ortg) * reg
            adj_drtg = league_drtg + (drtg - league_drtg) * reg
            expn = 13.3
            win_pct = adj_ortg ** expn / (adj_ortg ** expn + adj_drtg ** expn)
            # home team gets the advantage baked in via drtg difference;
            # add a small fixed home bump (~0.03) since drtg already captures venue? No —
            # drtg is venue-neutral here, so add it explicitly.
            if tid == row["home_team_id"]:
                win_pct = min(win_pct + 0.03, 0.99)
            preds.append(win_pct)
        home_pred = preds[0]
        rows.append((gid, home_pred))

        # update s2d
        for tid, pts, opp_pts, poss in (
            (row["home_team_id"], row["home_pts"], row["away_pts"], row["poss"]),
            (row["away_team_id"], row["away_pts"], row["home_pts"], row["poss_away"]),
        ):
            st = s2d.setdefault(tid, {"poss": 0.0, "pts": 0.0, "opp_pts": 0.0,
                                      "league_ortg": 116.0, "league_drtg": 116.0})
            st["poss"] += poss
            st["pts"] += pts
            st["opp_pts"] += opp_pts
        # league averages from all s2d data so far
        if s2d:
            tot_pts = sum(v["pts"] for v in s2d.values())
            tot_opp = sum(v["opp_pts"] for v in s2d.values())
            tot_poss = sum(v["poss"] for v in s2d.values())
            if tot_poss > 0:
                avg_ortg = tot_pts / tot_poss * 100
                avg_drtg = tot_opp / tot_poss * 100
                for v in s2d.values():
                    v["league_ortg"] = avg_ortg
                    v["league_drtg"] = avg_drtg

    pred_map = {gid: p for gid, p in rows}
    return games.with_columns(
        pl.Series("netrtg_pred", [pred_map[g] for g in games["game_id"].to_list()])
    )


def home_floor_predictions(games: pl.DataFrame) -> pl.DataFrame:
    return games.with_columns(pl.lit(0.58).alias("home_floor_pred"))
