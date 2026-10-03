"""Shared eval metrics for team + player baselines."""

from __future__ import annotations

import polars as pl
from sklearn.metrics import brier_score_loss, log_loss


def team_outcome_metrics(games: pl.DataFrame, pred_col: str) -> dict:
    """games must have: home_pts, away_pts, and pred_col (P(home win)).

    Returns accuracy, brier, logloss, and the 'bet' metrics:
      bet_acc: accuracy when predicting the side with p > 0.5 (ties -> home)
      breakeven at -110 is 52.38%; target 54%+
    """
    preds = games[pred_col].to_list()
    hp, ap = games["home_pts"].to_list(), games["away_pts"].to_list()
    actual = [h > a for h, a in zip(hp, ap)]
    n = len(preds)
    acc = sum(1 for p, a in zip(preds, actual) if (p > 0.5) == a) / n
    actual_i = [1 if a else 0 for a in actual]
    brier = brier_score_loss(actual_i, preds)
    ll = log_loss(actual_i, preds, labels=[0, 1])
    # bet: pick side with higher prob; tie -> home (p>0.5)
    bet_correct = 0
    bets = 0
    for p, a in zip(preds, actual):
        if p == 0.5:
            continue
        bets += 1
        if (p > 0.5) == a:
            bet_correct += 1
    bet_acc = bet_correct / bets if bets else float("nan")
    # expected profit per unit at -110
    ev_per_unit = (2.0 * bet_correct - bets) / bets if bets else float("nan")
    return {"n": n, "acc": acc, "brier": brier, "logloss": ll,
            "bet_acc": bet_acc, "bet_ev_per_unit": ev_per_unit}


def player_total_metrics(actuals: pl.DataFrame, preds: pl.DataFrame,
                         pred_col: str, component: str = "pts") -> dict:
    """actuals: player_id + actual_{component} (30-day sum).
    preds: player_id + pred_col (30-day sum).
    MAE over matched players + Spearman rank corr over matched players."""
    m = actuals.select(["player_id", pl.col(f"actual_{component}")]) \
        .join(preds.select(["player_id", pl.col(pred_col)]), on="player_id")
    m = m.filter(pl.col("actual_pts") > 0) if component == "pts" else m
    if len(m) < 10:
        return {"n": len(m), "mae": float("nan")}
    a = m[f"actual_{component}"].to_list()
    p = m[pred_col].to_list()
    mae = sum(abs(x - y) for x, y in zip(a, p)) / len(a)
    # spearman via rank
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks
    ra, rp = rank(a), rank(p)
    ma, mp = sum(ra) / len(ra), sum(rp) / len(rp)
    cov = sum((x - ma) * (y - mp) for x, y in zip(ra, rp))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vp = sum((y - mp) ** 2 for y in rp) ** 0.5
    spearman = cov / (va * vp) if va * vp else float("nan")
    return {"n": len(m), "mae": mae, "spearman": spearman}
