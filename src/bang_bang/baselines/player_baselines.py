"""Player component baselines.

Each baseline predicts a player's totals over the NEXT ~30 DAYS (PTS, REB,
AST, STL, BLK) given their history up to the prediction date.

Baselines:
  1. per36_flat : last game's per-36 rates x projected minutes
  2. per36_decay: exponentially decayed per-36 rates (half-life 28 days)
                  x projected minutes (50/50 blend of the implicit
                  decay-weighted projection and an explicit walk-forward
                  regression of next-30d minutes on (m8, m30, gp30))
  3. marcel     : prior-season totals, regressed to league avg, age-adjusted,
                  scaled 181 -> 30 days
"""

from __future__ import annotations

import datetime

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

SEASON_DAYS = 181
MONTH_DAYS = 30.0
SEASON_GAMES = 82
GAMES_PER_MONTH = SEASON_GAMES / SEASON_DAYS * MONTH_DAYS  # ~13.6
AVG_MIN_PER_GAME = 28.0

LEAGUE_AVG_PER36 = {
    "pts": 10.8, "reb": 4.6, "ast": 2.7, "stl": 0.9, "blk": 0.5,
}

COMPONENTS = ["pts", "reb", "ast", "stl", "blk"]


def _league_baseline_30d() -> dict[str, float]:
    """League-average per-36 output over a 30-day window (assumes ~28 min/game)."""
    base_min = AVG_MIN_PER_GAME * GAMES_PER_MONTH
    return {c: LEAGUE_AVG_PER36[c] / 36.0 * base_min for c in COMPONENTS}


def _proj_minutes(pg: pl.DataFrame, player_id: int, asof, min_games: int) -> float:
    h = pg.filter((pl.col("player_id") == player_id) & (pl.col("game_date") < asof))
    if len(h) < min_games:
        return 0.0
    return h.tail(8)["min"].mean() or 0.0


def predict_per36_flat(pg: pl.DataFrame, asof) -> pl.DataFrame:
    """Last game's per-36 rates x last-8-game average minutes x games/month."""
    players = (pg.filter(pl.col("game_date") < asof)
               .group_by("player_id").agg(pl.len().alias("gp"))
               .filter(pl.col("gp") >= 5)["player_id"].to_list())
    rows = []
    for pid in players:
        h = pg.filter((pl.col("player_id") == pid) & (pl.col("game_date") < asof))
        last = h.tail(1).row(0, named=True)
        mins = _proj_minutes(pg, pid, asof, 3)
        name = last["player_name"]
        row = {"player_id": pid, "player_name": name}
        for c in COMPONENTS:
            per36 = (last[c] / last["min"] * 36.0) if last["min"] > 1 else 0.0
            row[f"pred_{c}"] = per36 / 36.0 * mins * GAMES_PER_MONTH
        rows.append(row)
    return pl.DataFrame(rows)


def fit_minutes_model(history: pl.DataFrame) -> LinearRegression:
    """Walk-forward linear regression: next-30d total minutes ~ (m8, m30, gp30).

    m8 = avg minutes over the player's last 8 games; m30 = total minutes in
    the trailing 30 days; gp30 = games in that window. Target = total minutes
    in the NEXT 30 days. `history` must contain only games STRICTLY BEFORE the
    prediction asof (fit on all available seasons otherwise).

    2026-10-26 A/B (exp_minutes_model.py): blending this explicit projection
    50/50 with the implicit decay-weighted minutes beats the implicit-only
    default on pts MAE in all 3 windows, incl. both 2025-26 holdout windows
    (45.1->37.7, 43.4->40.1) with Spearman wins/ties. See NOTES.md.
    """
    X, y = [], []
    for pid in history["player_id"].unique().to_list():
        rows = (history.filter(pl.col("player_id") == pid)
                .sort("game_date").rows(named=True))
        for i, r in enumerate(rows):
            prior = rows[:i]
            if len(prior) < 8:
                continue
            d = r["game_date"]
            m8 = sum(x["min"] for x in prior[-8:]) / 8.0
            w30 = [x for x in prior if (d - x["game_date"]).days <= 30]
            fut = [x for x in rows[i + 1:] if (x["game_date"] - d).days < 30]
            if not fut:
                continue
            X.append((m8, sum(x["min"] for x in w30), len(w30)))
            y.append(sum(x["min"] for x in fut))
    if len(X) < 50:
        raise ValueError(f"fit_minutes_model: only {len(X)} samples")
    return LinearRegression().fit(np.array(X, dtype=float), np.array(y, dtype=float))


def predict_per36_decay(pg: pl.DataFrame, asof, half_life: float = 28.0,
                        minutes_model: LinearRegression | None = None) -> pl.DataFrame:
    """Exponential-decay weighted per-36 rates x projected 30-day minutes.

    half_life default 28d: 2026-10-26 sweep (7/10/14/21/28) — 28 beats 14 on
    pts MAE in all 3 eval windows (45.1/43.4/39.5 vs 45.7/44.6/41.1) with
    Spearman within noise (see NOTES.md).

    minutes_model: fit via fit_minutes_model(history before asof). When given,
    projected minutes = 0.5 * implicit (decay-weighted x 13.6 games) +
    0.5 * explicit regression — beats implicit-only in all 3 A/B windows
    (2026-10-26, see NOTES.md). When None, minutes stay implicit.
    """
    players = (pg.filter(pl.col("game_date") < asof)
               .group_by("player_id").agg(pl.len().alias("gp"))
               .filter(pl.col("gp") >= 8)["player_id"].to_list())
    rows = []
    for pid in players:
        h = (pg.filter((pl.col("player_id") == pid) & (pl.col("game_date") < asof))
             .sort("game_date"))
        num = {c: 0.0 for c in COMPONENTS}
        wmin = 0.0
        wtot = 0.0
        name = None
        for r in h.rows(named=True):
            age_days = (asof - r["game_date"]).days
            w = 0.5 ** (age_days / half_life)
            m = r["min"]
            name = r["player_name"]
            if m < 1:
                continue
            for c in COMPONENTS:
                num[c] += w * (r[c] / m * 36.0)
            wmin += w * m
            wtot += w
        if wtot == 0:
            continue
        implicit_min30 = (wmin / wtot) * GAMES_PER_MONTH
        if minutes_model is not None:
            prior8 = h.rows(named=True)[-8:]
            m8 = sum(x["min"] for x in prior8) / 8.0
            w30 = [x for x in h.rows(named=True)
                   if (asof - x["game_date"]).days <= 30]
            min30_exp = float(minutes_model.predict(
                np.array([[m8, sum(x["min"] for x in w30), len(w30)]]))[0])
            min30 = 0.5 * max(min30_exp, 0.0) + 0.5 * implicit_min30
        else:
            min30 = implicit_min30
        row = {"player_id": pid, "player_name": name}
        for c in COMPONENTS:
            row[f"pred_{c}"] = (num[c] / wtot) / 36.0 * min30
        rows.append(row)
    return pl.DataFrame(rows)


def predict_marcel(pg: pl.DataFrame, season_stats_prev: pl.DataFrame,
                   asof) -> pl.DataFrame:
    """Marcel: regressed prior-season totals scaled to a 30-day window.

    reg = 0.5 + 0.5 * min(gp_prev, 70)/70
    pred_30d = reg * (prev_total * 30/181) + (1-reg) * league_baseline_30d
    age: >=32 -> x0.95, <=23 -> x1.05
    """
    prev = season_stats_prev.filter(pl.col("gp") >= 15)
    base = _league_baseline_30d()
    rows = []
    for r in prev.rows(named=True):
        pid = r["player_id"]
        a = r.get("age")
        age_mult = 1.0
        if a is not None and not (a != a):  # not NaN
            if a >= 32:
                age_mult = 0.95
            elif a <= 23:
                age_mult = 1.05
        reg = 0.5 + 0.5 * min(r["gp"], 70.0) / 70.0
        row = {"player_id": pid, "player_name": r["player_name"], "reg": reg}
        for c in COMPONENTS:
            prev_30 = r[c] * (MONTH_DAYS / SEASON_DAYS)
            row[f"pred_{c}"] = (reg * prev_30 + (1 - reg) * base[c]) * age_mult
        rows.append(row)
    return pl.DataFrame(rows)
