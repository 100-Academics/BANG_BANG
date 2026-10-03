"""Team baseline backtest.

Usage: uv run python scripts/eval_team.py [season]   (default: 2025-26)

Walks the season forward, scoring each baseline on games after a warmup
(10 games per team, i.e. the first ~3 weeks are training).
"""

import sys

import polars as pl

from bang_bang.data.build import build_games
from bang_bang.baselines.team_baselines import (
    elo_predictions, netrtg_predictions, home_floor_predictions,
)
from bang_bang.eval.metrics import team_outcome_metrics


def main(season: str):
    games = build_games(season)
    # warmup: drop games before each team has played 10
    played: dict[int, int] = {}
    keep = []
    for row in games.iter_rows(named=True):
        for tid in (row["home_team_id"], row["away_team_id"]):
            played[tid] = played.get(tid, 0) + 1
        if played[row["home_team_id"]] >= 11 and played[row["away_team_id"]] >= 11:
            keep.append(row["game_id"])
    games = games.filter(pl.col("game_id").is_in(keep))
    print(f"{season}: {len(games)} eval games")

    results = {}
    for name, fn in [
        ("home_floor", lambda g: home_floor_predictions(g).select(["game_id", "home_floor_pred"])),
        ("elo_538", elo_predictions),
        ("netrtg_pyth", netrtg_predictions),
    ]:
        preds = fn(games)
        col = preds.columns[-1]
        merged = games.join(preds, on="game_id")
        m = team_outcome_metrics(merged, col)
        results[name] = m
        print(f"{name:14s} acc={m['acc']:.3f} brier={m['brier']:.4f} "
              f"logloss={m['logloss']:.4f} bet_acc={m['bet_acc']:.3f} "
              f"betEV={m['bet_ev_per_unit']:+.3f}/u (n={m['n']})")

    # reference: home win rate in sample
    hp, ap = games["home_pts"].to_list(), games["away_pts"].to_list()
    home_wr = sum(1 for h, a in zip(hp, ap) if h > a) / len(hp)
    print(f"reference: home win rate in eval set = {home_wr:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2025-26")
