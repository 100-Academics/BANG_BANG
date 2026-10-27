"""Live prediction entry point for BANG_BANG.

Usage: uv run python scripts/predict.py [season] [asof]
Defaults: season=2026-27, asof=today.

Data hygiene (found 2026-10-26, see NOTES.md):
  - ScheduleLeagueV2 includes non-regular games: preseason/global games and
    placeholder "TBD" rows. For future games `status` carries the tip-off time.
  - games.parquet (built from team logs) includes regular season + playoffs +
    a handful of junk exhibition games (2025-26: 12 vs Team Melo/Stripes/World/
    Loong-Lions/Jerusalem B.C.; 2024-25: 2).
  - "NBA teams" for a season = teams with >=15 scheduled games that season.

Teams (elo_538 default baseline):
  1. A/B on the prior season (holdout), 4 arms:
       fresh/carry x unfiltered/filtered, scored walk-forward (10-game warmup)
       exactly like scripts/eval_team.py. 'fresh+unfiltered' should reproduce
       the established number -> pipeline validation.
  2. Winning config walks forward to asof; predicts P(home win) for every
     remaining scheduled NBA game. Also per-team projected wins/final record.
  3. Est. point spread: linear calibration (through origin) of actual MoV on
     home-minus-away ELO diff (no home/rest bonus) over the walk.

Players:
  - marcel (prior-season stats, gp>=15, regressed, age-adjusted) — season-start
  - per36_decay (hl=14d, needs >=8 games) takes over once eligible
  - active filter: players with >=1 game in the current season

Outputs (gitignored):
  data/predictions/{season}_team_preds.parquet        (projected standings)
  data/predictions/{season}_team_preds_games.parquet  (per remaining game)
  data/predictions/{season}_player_preds.parquet
"""
from __future__ import annotations

import copy
import datetime
import sys

import polars as pl

from bang_bang.data.build import build_games, build_player_games
from bang_bang.data.fetch import RAW_DIR, fetch_schedule
from bang_bang.baselines.team_baselines import EloTracker
from bang_bang.baselines.player_baselines import predict_marcel, predict_per36_decay
from bang_bang.eval.metrics import team_outcome_metrics

PRED_DIR = RAW_DIR.parent / "predictions"
WARMUP = 10  # games per team before scoring (same as eval_team.py)
MIN_GAMES_FOR_NBA = 15  # schedule-based filter for real NBA teams


def _prev_season(season: str) -> str:
    y = season.split("-")[0]
    return f"{int(y) - 1}-{y[-2:]}"


def _nba_team_ids(season: str) -> set[int]:
    """Teams with >= MIN_GAMES_FOR_NBA scheduled games that season."""
    sched = fetch_schedule(season)
    cnt = (sched.select(pl.col("home_team_id").alias("t")).vstack(
           sched.select(pl.col("away_team_id").alias("t")))
           .group_by("t").agg(pl.len().alias("n")))
    return set(cnt.filter(pl.col("n") >= MIN_GAMES_FOR_NBA)["t"].to_list())


def _filter_nba(games: pl.DataFrame, nba_ids: set[int]) -> pl.DataFrame:
    return games.filter(
        pl.col("home_team_id").is_in(list(nba_ids))
        & pl.col("away_team_id").is_in(list(nba_ids))
    )


def _walk(tr: EloTracker, games: pl.DataFrame,
          collect: list | None = None) -> None:
    """Update tracker over completed games (sorted by date). Optionally collect
    (elo_diff_no_home, signed mov) pairs for spread calibration."""
    for row in games.iter_rows(named=True):
        d = (tr.ratings.get(row["home_team_id"], 1500.0)
             - tr.ratings.get(row["away_team_id"], 1500.0))
        tr.update(row["home_team_id"], row["away_team_id"],
                  row["home_pts"], row["away_pts"], row["game_date"])
        if collect is not None:
            collect.append((d, row["home_pts"] - row["away_pts"]))


def _warmup_mask(games: pl.DataFrame) -> set[str]:
    played: dict[int, int] = {}
    keep: set[str] = set()
    for row in games.iter_rows(named=True):
        for tid in (row["home_team_id"], row["away_team_id"]):
            played[tid] = played.get(tid, 0) + 1
        if played[row["home_team_id"]] >= WARMUP + 1 and played[row["away_team_id"]] >= WARMUP + 1:
            keep.add(row["game_id"])
    return keep


def _score(tr_factory, games: pl.DataFrame) -> dict:
    """Walk-forward score of a tracker factory on `games` (like eval_team.py)."""
    tr = tr_factory()
    warm = _warmup_mask(games)
    preds, gids = [], []
    for row in games.iter_rows(named=True):
        if row["game_id"] not in warm:
            continue
        p = tr.predict(row["home_team_id"], row["away_team_id"], row["game_date"])
        tr.update(row["home_team_id"], row["away_team_id"],
                  row["home_pts"], row["away_pts"], row["game_date"])
        preds.append(p)
        gids.append(row["game_id"])
    g = games.filter(pl.col("game_id").is_in(gids)).with_columns(pl.Series("p", preds))
    return team_outcome_metrics(g, "p")


def _available_seasons() -> list[str]:
    """Seasons with a built games.parquet, oldest first."""
    return sorted(p.name.split("_")[0] for p in RAW_DIR.glob("*_games.parquet"))


def _deep_carry(target_season: str, use_filter: bool) -> EloTracker:
    """Walk ELO through every available season before target_season, oldest first.

    2026-10-26: 3-season carry beats 2-season carry on the 2025-26 holdout
    (+0.356 vs +0.344 betEV) — see NOTES.md. So the live walk starts from the
    earliest available season, not just the pre-prior."""
    tr = EloTracker()
    for s in _available_seasons():
        if s >= target_season:
            break
        games = build_games(s)
        if use_filter:
            sched_p = RAW_DIR / f"{s}_schedule.parquet"
            if sched_p.exists():
                games = _filter_nba(games, _nba_team_ids(s))
        _walk(tr, games)
    return tr


def main(season: str, asof: datetime.date | None = None):
    asof = asof or datetime.date.today()
    prior = _prev_season(season)
    pre_prior = _prev_season(prior)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    have_pre = (RAW_DIR / f"{pre_prior}_schedule.parquet").exists()
    nba_cur = _nba_team_ids(season)
    nba_prior = _nba_team_ids(prior)
    nba_pre = _nba_team_ids(pre_prior) if have_pre else None
    print(f"asof={asof}  NBA teams: {season}={len(nba_cur)} {prior}={len(nba_prior)} "
          f"{pre_prior}={'n/a' if nba_pre is None else len(nba_pre)}")

    # ---------------- current season: played (clean, from team logs) + future
    cur = build_games(season)
    played = cur.filter(pl.col("status").str.starts_with("Final")).sort("game_date")
    sched = fetch_schedule(season)
    future = (sched.filter(pl.col("game_date") >= asof)
              .filter(~pl.col("status").str.starts_with("Final"))
              .filter(pl.col("home_team_id").is_in(list(nba_cur))
                      & pl.col("away_team_id").is_in(list(nba_cur)))
              .sort("game_date"))
    print(f"{season}: {len(played)} games played, {len(future)} NBA games remaining")

    # ---------------- A/B on the prior season (holdout)
    prior_games = build_games(prior)
    arms: dict[str, tuple] = {
        "fresh+unfiltered": (lambda: EloTracker(), prior_games),
        "fresh+filtered": (lambda: EloTracker(), _filter_nba(prior_games, nba_prior)),
    }
    if len(_available_seasons()) > 1:
        snap_u = _deep_carry(prior, use_filter=False)
        snap_f = _deep_carry(prior, use_filter=True)
        arms["carry+unfiltered"] = (lambda s=snap_u: copy.deepcopy(s), prior_games)
        arms["carry+filtered"] = (lambda s=snap_f: copy.deepcopy(s),
                                  _filter_nba(prior_games, nba_prior))

    results = {}
    print(f"\nA/B on {prior} (holdout, warmup={WARMUP}g):")
    for name, (factory, games) in arms.items():
        m = _score(factory, games)
        results[name] = m
        print(f"  {name:19s} acc={m['acc']:.3f} brier={m['brier']:.4f} "
              f"betEV={m['bet_ev_per_unit']:+.3f}/u (n={m['n']})")
    winner = max(results, key=lambda k: results[k]["bet_ev_per_unit"])
    use_carry = winner.startswith("carry")
    use_filter = "filtered" in winner
    print(f"  -> shipping: {winner}")

    # ---------------- live team walk to asof
    if use_carry:
        tr = _carry_snapshot(pre_prior, nba_pre if use_filter else None)
    else:
        tr = EloTracker()
    src = prior_games if not use_filter else _filter_nba(prior_games, nba_prior)
    calib: list[tuple[float, float]] = []
    _walk(tr, src, collect=calib)
    _walk(tr, played, collect=calib)  # current season is clean (team logs only)

    num = sum(d * m for d, m in calib)
    den = sum(d * d for d, _ in calib)
    pts_per_elo = num / den if den else 0.0
    print(f"spread calibration: {pts_per_elo:.4f} pts/ELO over {len(calib)} games "
          f"(~{1 / pts_per_elo:.0f} ELO per point)")

    # ---------------- per remaining game
    rows = []
    for row in future.iter_rows(named=True):
        p = tr.predict(row["home_team_id"], row["away_team_id"], row["game_date"])
        d = (tr.ratings.get(row["home_team_id"], 1500.0)
             - tr.ratings.get(row["away_team_id"], 1500.0))
        rows.append({
            "game_id": row["game_id"],
            "game_date": row["game_date"],
            "home_team": row["home_team_name"],
            "away_team": row["away_team_name"],
            "p_home": round(p, 4),
            "p_away": round(1 - p, 4),
            "elo_home": round(tr.ratings.get(row["home_team_id"], 1500.0), 1),
            "elo_away": round(tr.ratings.get(row["away_team_id"], 1500.0), 1),
            "est_spread_pts": round(pts_per_elo * d, 1),  # + = home favored
        })
    team_preds = pl.DataFrame(rows)

    # team id -> full name (games.parquet uses abbreviations; schedule has names)
    id2name_df = (sched.select(["home_team_id", "home_team_name"]).unique()
                  .vstack(sched.select(["away_team_id", "away_team_name"]).rename(
                      {"away_team_id": "home_team_id", "away_team_name": "home_team_name"}))
                  .unique())
    id2name = {r["home_team_id"]: r["home_team_name"]
               for r in id2name_df.iter_rows(named=True)}

    tw: dict[str, float] = {}
    tl: dict[str, float] = {}
    for row in team_preds.iter_rows(named=True):
        tw[row["home_team"]] = tw.get(row["home_team"], 0.0) + row["p_home"]
        tw[row["away_team"]] = tw.get(row["away_team"], 0.0) + row["p_away"]
        tl[row["away_team"]] = tl.get(row["away_team"], 0.0) + row["p_home"]
        tl[row["home_team"]] = tl.get(row["home_team"], 0.0) + row["p_away"]

    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    for r in played.select(["home_team_id", "away_team_id", "home_pts", "away_pts"]) \
            .iter_rows(named=True):
        h = id2name.get(r["home_team_id"], str(r["home_team_id"]))
        a = id2name.get(r["away_team_id"], str(r["away_team_id"]))
        if r["home_pts"] > r["away_pts"]:
            wins[h] = wins.get(h, 0) + 1
            losses[a] = losses.get(a, 0) + 1
        else:
            wins[a] = wins.get(a, 0) + 1
            losses[h] = losses.get(h, 0) + 1

    proj_rows = []
    for name in sorted(set(list(tw) + list(wins))):
        pw, pl_ = tw.get(name, 0.0), tl.get(name, 0.0)
        proj_rows.append({
            "team": name,
            "record": f"{wins.get(name, 0)}-{losses.get(name, 0)}",
            "proj_remaining_wins": round(pw, 1),
            "proj_remaining_losses": round(pl_, 1),
            "proj_final_wins": round(wins.get(name, 0) + pw, 1),
            "proj_final_losses": round(losses.get(name, 0) + pl_, 1),
        })
    team_wins = pl.DataFrame(proj_rows).sort("proj_final_wins", "team", descending=[True, False])
    team_wins.write_parquet(PRED_DIR / f"{season}_team_preds.parquet")
    team_preds.write_parquet(PRED_DIR / f"{season}_team_preds_games.parquet")

    # ---------------- player projections
    pg = build_player_games(season)
    prev_stats = pl.read_parquet(RAW_DIR / f"{prior}_season_stats.parquet")
    active = set(pg["player_id"].to_list())

    marcel = predict_marcel(pg, prev_stats, asof).filter(pl.col("player_id").is_in(active))
    decay = predict_per36_decay(pg, asof, half_life=28.0)  # needs >=8 games; hl from 2026-10-26 sweep

    if len(decay):
        base = marcel.join(
            decay.select([c for c in decay.columns if c != "player_name"]),
            on="player_id", how="left", suffix="_decay")
    else:
        base = marcel
        for c in ["pred_pts_decay", "pred_reb_decay", "pred_ast_decay",
                  "pred_stl_decay", "pred_blk_decay"]:
            base = base.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
    base = base.with_columns(
        pl.when(pl.col("pred_pts_decay").is_not_null())
        .then(pl.lit("per36_decay")).otherwise(pl.lit("marcel")).alias("source"),
        pl.col("pred_pts_decay").fill_null(pl.col("pred_pts")).alias("pred_pts"),
        pl.col("pred_reb_decay").fill_null(pl.col("pred_reb")).alias("pred_reb"),
        pl.col("pred_ast_decay").fill_null(pl.col("pred_ast")).alias("pred_ast"),
        pl.col("pred_stl_decay").fill_null(pl.col("pred_stl")).alias("pred_stl"),
        pl.col("pred_blk_decay").fill_null(pl.col("pred_blk")).alias("pred_blk"),
    )
    gp = (pg.group_by("player_id").agg(
        pl.len().alias("gp_2627"),
        pl.col("team_abbrev").last().alias("team_abbrev")))
    player_preds = (base.join(gp, on="player_id", how="left")
                    .drop(["pred_pts_decay", "pred_reb_decay", "pred_ast_decay",
                           "pred_stl_decay", "pred_blk_decay", "reg"])
                    .sort("pred_pts", descending=True))
    player_preds.write_parquet(PRED_DIR / f"{season}_player_preds.parquet")

    # ---------------- boards
    print(f"\n== {season} projected final standings (top 15) ==")
    print(team_wins.head(15))
    print(f"\n== next 12 games ==")
    print(team_preds.head(12).select(
        ["game_date", "away_team", "home_team", "p_home", "est_spread_pts"]))
    print(f"\n== top 25 players by projected 30-day points "
          f"(sources: {player_preds['source'].unique().sort().to_list()}) ==")
    print(player_preds.head(25).select(
        ["player_name", "team_abbrev", "gp_2627", "source",
         "pred_pts", "pred_reb", "pred_ast"]).with_columns(
        [pl.col(c).round(1) for c in ["pred_pts", "pred_reb", "pred_ast"]]))
    print(f"\nwrote {PRED_DIR / (season + '_team_preds.parquet')}")
    print(f"wrote {PRED_DIR / (season + '_team_preds_games.parquet')}")
    print(f"wrote {PRED_DIR / (season + '_player_preds.parquet')}")


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "2026-27"
    a = datetime.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else None
    main(s, a)
