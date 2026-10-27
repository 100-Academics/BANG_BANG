"""Experiment: per36_decay half-life sweep (NOTES item 2).

Current default: hl=14, min games 8. Sweeps hl in {7, 10, 14, 21, 28} on the
established eval windows. 14 must appear in the table as the baseline.

Decision rule: a new hl ships only if it beats hl=14 on the HOLDOUT windows
(2025-26 x2) on pts MAE and pts Spearman; 2024-25 is diagnostic only.
"Basically the same" -> keep 14.

Usage: uv run python scripts/exp_decay_sweep.py
"""
import datetime

import polars as pl

from bang_bang.data.build import build_player_games
from bang_bang.baselines.player_baselines import predict_per36_decay, COMPONENTS
from bang_bang.eval.metrics import player_total_metrics

HALF_LIVES = [7.0, 10.0, 14.0, 21.0, 28.0]
WINDOWS = [
    # (season, asof, holdout?)
    ("2025-26", "2025-12-01", True),
    ("2025-26", "2026-01-15", True),
    ("2024-25", "2024-12-01", False),
]
CACHE: dict[str, pl.DataFrame] = {}


def _pg(season: str) -> pl.DataFrame:
    if season not in CACHE:
        CACHE[season] = build_player_games(season)
    return CACHE[season]


def main():
    for season, asof_s, holdout in WINDOWS:
        asof = datetime.date.fromisoformat(asof_s)
        end = asof + datetime.timedelta(days=30)
        pg_all = _pg(season)
        actuals = (
            pg_all.filter((pl.col("game_date") >= asof) & (pl.col("game_date") < end))
            .group_by("player_id", "player_name")
            .agg([(pl.col(c).sum()).alias(f"actual_{c}") for c in COMPONENTS])
            .filter(pl.col("actual_pts") >= 30)
        )
        print(f"\n== {season} asof={asof_s} {'(HOLDOUT)' if holdout else '(in-sample)'} "
              f"(n={len(actuals)}) ==")
        print(f"{'hl':>5s} | " + " | ".join(
            f"{c} MAE / rho(pts)" for c in ["pts", "reb", "ast"]))
        for hl in HALF_LIVES:
            preds = predict_per36_decay(
                pg_all.filter(pl.col("game_date") < asof), asof, half_life=hl)
            cells = []
            for c in ["pts", "reb", "ast"]:
                m = player_total_metrics(actuals, preds, f"pred_{c}", c)
                cells.append(f"{m['mae']:.1f} / {m['spearman']:.3f}")
            tag = "  <- default" if hl == 14.0 else ""
            print(f"{hl:5.0f} | " + " | ".join(cells) + tag)


if __name__ == "__main__":
    main()
