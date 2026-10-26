"""Live prediction entry point for BANG_BANG.

Usage: uv run python scripts/predict.py [season] [asof]
Defaults: season=2026-27, asof=today.

Teams:
  1. A/B validates ELO carry-over on the prior season (holdout):
     'fresh' (start at 1500) vs 'carry' (ratings carried from the pre-prior season).
     10-game warmup per team, walk-forward, scored on acc/brier/betEV.
  2. The winning config walks forward to asof, then predicts P(home win) for
     every remaining game of the season. Also emits per-team projected wins.
  3. Estimated point spread: linear calibration (through origin) of actual MoV
     on home-minus-away ELO diff (no home/rest bonus) over the walk.

Players:
  - marcel (prior-season stats, gp>=15, regressed, age-adjusted) — season-start default
  - per36_decay (hl=14d, needs >=8 games) takes over once eligible; used when eligible
  - active filter: players with >=1 game in the current season

Outputs (gitignored):
  data/predictions/{season}_team_preds.parquet
  data/predictions/{season}_player_preds.parquet
"""
from __future__ import annotations

import datetime
import sys

import polars as pl

from bang_bang.data.build import build_games, build_player_games
from bang_bang.data.fetch import RAW_DIR
from bang_bang.baselines.team_baselines import EloTracker
from bang_bang.baselines.player_baselines import predict_marcel, predict_per36_decay
from bang_bang.eval.metrics import team_outcome_metrics

PRED_DIR = RAW_DIR.parent / "predictions"
WARMUP = 10  # games per team before scoring


def _prev_season(season: str) -> str:
    y = season.split("-")[0]
    return f"{int(y) - 1}-{y[-2:]}"


def _walk(tr: EloTracker, games: pl.DataFrame,
          collect: list | None = None) -> None:
    """Update tracker over completed games (sorted by date). Optionally collect
    (elo_diff_no_home, signed mov) pairs for spread calibration."""
    for row in games.iter_rows(named=True):
        elo_h = tr.ratings.get(row["home_team_id"], 1500.0)
        elo_a = tr.ratings.get(row["away_team_id"], 1500.0)
        d = elo_h - elo_a
        tr.update(row["home_team_id"], row["away_team_id"],
                  row["home_pts"], row["away_pts"], row["game_date"])
        if collect is not None:
            collect.append((d, row["home_pts"] - row["away_pts"]))


def _warmup_mask(games: pl.DataFrame) -> list[str]:
    played: dict[int, int] = {}
    keep = []
    for row in games.iter_rows(named=True):
        for tid in (row["home_team_id"], row["away_team_id"]):
            played[tid] = played.get(tid, 0) + 1
        if played[row["home_team_id"]] >= WARMUP + 1 and played[row["away_team_id"]] >= WARMUP + 1:
            keep.append(row["game_id"])
    return keep


def _score(tr_factory, games: pl.DataFrame) -> dict:
    """Run a walk-forward prediction on `games` with a fresh tracker, score it."""
    tr = tr_factory()
    preds, gids = [], []
    warm = set(_warmup_mask(games))
    for row in games.iter_rows(named=True):
        if row["game_id"] not in warm:
            continue
        p = tr.predict(row["home_team_id"], row["away_team_id"], row["game_date"])
        tr.update(row["home_team_id"], row["away_team_id"],
                  row["home_pts"], row["away_pts"], row["game_date"])
        preds.append(p)
        gids.append(row["game_id"])
    g = games.with_columns(pl.Series("p", preds)).filter(pl.col("game_id").is_in(gids))
    return team_outcome_metrics(g, "p")


def main(season: str, asof: datetime.date | None = None):
    asof = asof or datetime.date.today()
    prior = _prev_season(season)
    pre_prior = _prev_season(prior)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    cur = build_games(season)
    played = cur.filter(pl.col("status").str.starts_with("Final"))
    future = cur.filter(~pl.col("status").str.starts_with("Final"))
    future = future.filter(pl.col("game_date") >= asof)
    print(f"asof={asof}  {season}: {len(played)} games played, "
          f"{len(future)} remaining on schedule")

    # ---------------- A/B: carry-over vs fresh, on the prior season (holdout)
    prior_games = build_games(prior)
    fresh_m = _score(lambda: EloTracker(), prior_games)

    carry_tr = EloTracker()
    if (RAW_DIR / f"{pre_prior}_games.parquet").exists():
        _walk(carry_tr, build_games(pre_prior))
    carry_factory = None
    import copy
    carry_snapshot = copy.deepcopy(carry_tr)

    def _carry_factory():
        return copy.deepcopy(carry_snapshot)

    carry_m = _score(_carry_factory, prior_games)

    print(f"\nA/B on {prior} (holdout, warmup={WARMUP}g):")
    print(f"  fresh: acc={fresh_m['acc']:.3f} brier={fresh_m['brier']:.4f} "
          f"betEV={fresh_m['bet_ev_per_unit']:+.3f}/u (n={fresh_m['n']})")
    print(f"  carry: acc={carry_m['acc']:.3f} brier={carry_m['brier']:.4f} "
          f"betEV={carry_m['bet_ev_per_unit']:+.3f}/u (n={carry_m['n']})")
    use_carry = carry_m["bet_ev_per_unit"] > fresh_m["bet_ev_per_unit"]
    print(f"  -> shipping: {'carry' if use_carry else 'fresh'}")

    # ---------------- live team walk to asof
    tr = copy.deepcopy(carry_snapshot) if use_carry else EloTracker()
    calib: list[tuple[float, float]] = []
    _walk(tr, prior_games, collect=calib)
    _walk(tr, played, collect=calib)

    # spread calibration: points per ELO (through origin)
    num = sum(d * m for d, m in calib)
    den = sum(d * d for d, _ in calib)
    pts_per_elo = num / den if den else 0.0
    print(f"spread calibration: {pts_per_elo:.4f} pts/ELO over {len(calib)} games "
          f"(~{100 / pts_per_elo:.0f} ELO per point)")

    # ---------------- team predictions for remaining games
    rows = []
    for row in future.sort("game_date").iter_rows(named=True):
        p = tr.predict(row["home_team_id"], row["away_team_id"], row["game_date"])
        d = tr.ratings.get(row["home_team_id"], 1500.0) - tr.ratings.get(row["away_team_id"], 1500.0)
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

    # per-team projected remaining wins + final record
    tw: dict[str, float] = {}
    tl: dict[str, float] = {}
    for row in team_preds.iter_rows(named=True):
        tw[row["home_team"]] = tw.get(row["home_team"], 0.0) + row["p_home"]
        tw[row["away_team"]] = tw.get(row["away_team"], 0.0) + row["p_away"]
        tl[row["away_team"]] = tl.get(row["away_team"], 0.0) + row["p_home"]
        tl[row["home_team"]] = tl.get(row["home_team"], 0.0) + row["p_away"]
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    res = (played.select(["home_team_name", "away_team_name", "home_pts", "away_pts"]))
    for r in res.iter_rows(named=True):
        if r["home_pts"] > r["away_pts"]:
            wins[r["home_team_name"]] = wins.get(r["home_team_name"], 0) + 1
            losses[r["away_team_name"]] = losses.get(r["away_team_name"], 0) + 1
        else:
            wins[r["away_team_name"]] = wins.get(r["away_team_name"], 0) + 1
            losses[r["home_team_name"]] = losses.get(r["home_team_name"], 0) + 1
    proj_rows = []
    for name in sorted(set(list(tw) + list(wins))):
        pw = round(tw.get(name, 0.0), 1)
        pl_ = round(tl.get(name, 0.0), 1)
        proj_rows.append({
            "team": name,
            "record": f"{wins.get(name, 0)}-{losses.get(name, 0)}",
            "proj_remaining_wins": pw,
            "proj_remaining_losses": pl_,
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
    decay = predict_per36_decay(pg, asof, half_life=14.0)  # needs >=8 games

    base = marcel
    if len(decay):
        m2 = decay.select([c for c in decay.columns if c != "player_name"])
        base = base.join(m2, on="player_id", how="left", suffix="_decay")
    else:
        for c in ["pred_pts_decay", "pred_reb_decay", "pred_ast_decay", "pred_stl_decay", "pred_blk_decay"]:
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
    gp = (pg.group_by("player_id").agg(pl.len().alias("gp_2627"),
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
          f"(source: {player_preds['source'].unique().to_list()}) ==")
    print(player_preds.head(25).select(
        ["player_name", "team_abbrev", "gp_2627", "source",
         "pred_pts", "pred_reb", "pred_ast"]).round(1))
    print(f"\nwrote {PRED_DIR / (season + '_team_preds.parquet')}")
    print(f"wrote {PRED_DIR / (season + '_player_preds.parquet')}")


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "2026-27"
    a = datetime.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else None
    main(s, a)
