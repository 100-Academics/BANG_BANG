"""Experiment: minutes as an explicit target (NOTES item 3).

per36_decay projects minutes implicitly: decay-weighted trailing minutes
x 13.6 games/30d. This replaces ONLY the minutes component with an explicit
walk-forward linear regression of next-30-day total minutes on
(m8, m30, gp30) — fit strictly on data before asof (injury/rest aware).
Per-36 rates are identical to per36_decay (hl=28); only minutes change.

Eligibility = per36_decay eligibility (>=8 games before asof). Eval windows
and actuals filter identical to exp_decay_sweep.
Decision: explicit minutes ships only if it beats per36_decay (implicit) on
the holdout windows on pts MAE AND pts Spearman.

Usage: uv run python scripts/exp_minutes_model.py
"""
import datetime

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from bang_bang.data.build import build_player_games
from bang_bang.baselines.player_baselines import (
    COMPONENTS, GAMES_PER_MONTH, predict_per36_decay)
from bang_bang.eval.metrics import player_total_metrics

HALF_LIFE = 28.0
WINDOWS = [
    # (season, asof, train seasons, holdout?)
    ("2025-26", "2025-12-01", ["2023-24", "2024-25"], True),
    ("2025-26", "2026-01-15", ["2023-24", "2024-25"], True),
    ("2024-25", "2024-12-01", ["2023-24"], False),
]
PG_CACHE: dict[str, pl.DataFrame] = {}


def _pg(season: str) -> pl.DataFrame:
    if season not in PG_CACHE:
        PG_CACHE[season] = build_player_games(season)
    return PG_CACHE[season]


def _per36_rates(h: list[dict], asof: datetime.date) -> tuple[dict, float] | None:
    """Same decay per-36 rates as predict_per36_decay (hl=28). Returns (rates, decay_mins)."""
    num = {c: 0.0 for c in COMPONENTS}
    wmin, wtot = 0.0, 0.0
    for r in h:
        age = (asof - r["game_date"]).days
        w = 0.5 ** (age / HALF_LIFE)
        m = r["min"]
        if m < 1:
            continue
        for c in COMPONENTS:
            num[c] += w * (r[c] / m * 36.0)
        wmin += w * m
        wtot += w
    if wtot == 0:
        return None
    return ({c: num[c] / wtot for c in COMPONENTS}, wmin / wtot)


def _build_samples(pg: pl.DataFrame) -> list[tuple]:
    """Per-player walk-forward samples: (m8, m30, gp30, next-30d minutes)."""
    samples = []
    for pid in pg["player_id"].unique().to_list():
        h = pg.filter(pl.col("player_id") == pid).sort("game_date")
        rows = h.rows(named=True)
        for i, r in enumerate(rows):
            prior = rows[:i]
            if len(prior) < 8:
                continue
            d = r["game_date"]
            m8 = sum(x["min"] for x in prior[-8:]) / 8.0
            w30 = [x for x in prior if (d - x["game_date"]).days <= 30]
            m30 = sum(x["min"] for x in w30)
            gp30 = len(w30)
            fut = [x for x in rows[i + 1:] if (x["game_date"] - d).days < 30]
            if not fut:
                continue
            samples.append((m8, m30, gp30, sum(x["min"] for x in fut)))
    return samples


def main():
    for season, asof_s, prior_season, holdout in WINDOWS:
        asof = datetime.date.fromisoformat(asof_s)
        end = asof + datetime.timedelta(days=30)
        pg_all = _pg(season)
        actuals = (
            pg_all.filter((pl.col("game_date") >= asof) & (pl.col("game_date") < end))
            .group_by("player_id", "player_name")
            .agg([(pl.col(c).sum()).alias(f"actual_{c}") for c in COMPONENTS])
            .filter(pl.col("actual_pts") >= 30)
        )
        # training samples: current season before asof (+ prior season if available)
        samples = _build_samples(pg_all.filter(pl.col("game_date") < asof))
        if prior_season:
            samples += _build_samples(_pg(prior_season))
        X = np.array([(a, b, c) for a, b, c, _ in samples], dtype=float)
        y = np.array([d for *_, d in samples], dtype=float)
        model = LinearRegression().fit(X, y)

        # predictions: implicit (default) vs explicit minutes
        implicit = predict_per36_decay(pg_all.filter(pl.col("game_date") < asof),
                                       asof, half_life=HALF_LIFE)
        rows_exp = []
        hist = pg_all.filter(pl.col("game_date") < asof)
        eligible = (hist.group_by("player_id").agg(pl.len().alias("gp"))
                    .filter(pl.col("gp") >= 8)["player_id"].to_list())
        for pid in eligible:
            h = hist.filter(pl.col("player_id") == pid).rows(named=True)
            rates = _per36_rates(h, asof)
            if rates is None:
                continue
            rmap, _ = rates
            prior8 = [x for x in h][-8:]
            m8 = sum(x["min"] for x in prior8) / 8.0
            w30 = [x for x in h if (asof - x["game_date"]).days <= 30]
            Xp = np.array([[m8, sum(x["min"] for x in w30), len(w30)]])
            min30_exp = max(float(model.predict(Xp)[0]), 0.0)
            name = h[-1]["player_name"]
            row = {"player_id": pid, "player_name": name}
            for c in COMPONENTS:
                row[f"pred_{c}"] = rmap[c] / 36.0 * min30_exp
            rows_exp.append(row)
        explicit = pl.DataFrame(rows_exp)

        tag = "(HOLDOUT)" if holdout else "(in-sample)"
        print(f"\n== {season} asof={asof_s} {tag} (n={len(actuals)}, "
              f"train samples={len(samples)}) ==")
        for label, preds in (("implicit (default)", implicit), ("explicit minutes", explicit)):
            cells = []
            for c in ["pts", "reb", "ast"]:
                m = player_total_metrics(actuals, preds, f"pred_{c}", c)
                cells.append(f"{c} MAE={m['mae']:.1f} rho(pts)={m['spearman']:.3f}")
            print(f"  {label:18s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
