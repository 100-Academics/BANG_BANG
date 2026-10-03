"""Parameterized player baseline eval: season, prior season, asof date.

Usage: uv run python scripts/eval_player_window.py <season> <prior_season> <asof>
"""
import datetime
import sys

import polars as pl

from bang_bang.data.build import build_player_games
from bang_bang.data.fetch import RAW_DIR
from bang_bang.baselines.player_baselines import (
    predict_per36_flat, predict_per36_decay, predict_marcel, COMPONENTS,
)
from bang_bang.eval.metrics import player_total_metrics


def main(season: str, prior_season: str, asof: str):
    asof_d = datetime.date.fromisoformat(asof)
    end_d = asof_d + datetime.timedelta(days=30)

    pg_all = build_player_games(season)
    pg = pg_all.filter(pl.col("game_date") < asof_d)
    actuals = (
        pg_all.filter((pl.col("game_date") >= asof_d) & (pl.col("game_date") < end_d))
        .group_by("player_id", "player_name")
        .agg([(pl.col(c).sum()).alias(f"actual_{c}") for c in COMPONENTS])
        .filter(pl.col("actual_pts") >= 30)
    )
    prev = pl.read_parquet(RAW_DIR / f"{prior_season}_season_stats.parquet")

    preds = {
        "per36_flat": predict_per36_flat(pg, asof_d),
        "per36_decay": predict_per36_decay(pg, asof_d),
        "marcel": predict_marcel(pg, prev, asof_d),
    }
    print(f"== {season} asof={asof} (n={len(actuals)}) ==")
    for name, p in preds.items():
        line = []
        for c in ["pts", "reb", "ast"]:
            m = player_total_metrics(actuals, p, f"pred_{c}", c)
            line.append(f"{c} MAE={m['mae']:.1f} rho={m['spearman']:.3f}(n={m['n']})")
        print(f"  {name:12s} " + "  ".join(line))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
