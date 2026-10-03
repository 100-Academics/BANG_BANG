"""Player baseline backtest on a 30-day window.

Usage: uv run python scripts/eval_player.py [asof_date]
  default asof: 2025-12-01 (mid-season in the 2025-26 holdout)

Predicts 30-day totals from history before asof, scores vs actual totals
for [asof, asof+30d).

Marcel uses 2024-25 season_stats as the prior year.
"""

import datetime

import polars as pl

from bang_bang.data.build import build_player_games
from bang_bang.data.fetch import RAW_DIR
from bang_bang.baselines.player_baselines import (
    predict_per36_flat, predict_per36_decay, predict_marcel, COMPONENTS,
)
from bang_bang.eval.metrics import player_total_metrics


def main(asof: str):
    asof_d = datetime.date.fromisoformat(asof)
    end_d = asof_d + datetime.timedelta(days=30)

    # history: full 2025-26 season table, filter < asof
    pg_all = build_player_games("2025-26")
    pg = pg_all.filter(pl.col("game_date") < asof_d)

    # actuals: 30-day window
    actual_rows = (
        pg_all.filter((pl.col("game_date") >= asof_d) & (pl.col("game_date") < end_d))
        .group_by("player_id", "player_name", "team_abbrev")
        .agg([
            (pl.col(c).sum()).alias(f"actual_{c}") for c in COMPONENTS
        ])
    )
    actuals = actual_rows.filter(pl.col("actual_pts") >= 30)  # ~1+ game of production

    print(f"asof={asof} window=[{asof_d}..{end_d}) players with actuals: {len(actuals)}")

    results = {}
    preds_flat = predict_per36_flat(pg, asof_d)
    preds_decay = predict_per36_decay(pg, asof_d)
    prev_stats = pl.read_parquet(RAW_DIR / "2024-25_season_stats.parquet")
    preds_marcel = predict_marcel(pg, prev_stats, asof_d)

    for name, preds in [
        ("per36_flat", preds_flat),
        ("per36_decay", preds_decay),
        ("marcel", preds_marcel),
    ]:
        line = [name]
        for c in ["pts", "reb", "ast"]:
            m = player_total_metrics(actuals, preds, f"pred_{c}", c)
            line.append(f"{c}: MAE={m['mae']:.2f} rho={m['spearman']:.3f}")
            results.setdefault(c, {})[name] = m
        print("  ".join(line))

    print(f"matched n per baseline (pts): "
          f"{ {c: {k: v['n'] for k, v in d.items()} for c, d in results.items()} }")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "2025-12-01")
