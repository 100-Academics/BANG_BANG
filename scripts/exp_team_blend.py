"""Experiment: team ELO + net-rating Pythagorean blend + tuned home offset
(NOTES item 5).

Protocol:
  1. Tune (w_elo, home_offset) on 2024-25 (in-sample), fresh walk, filtered.
  2. Evaluate candidates on 2025-26 (HOLDOUT):
       a) best fresh blend (tuned)
       b) best carry blend (tuned config, ratings carried 2023-24 -> 2024-25 -> 2025-26)
       c) carry+filtered elo_538  <- current live default (bar: 0.672/+0.344)
       d) fresh+filtered elo_538  <- established in-season baseline
  Ship the blend only if it beats (c) on the holdout.

Usage: uv run python scripts/exp_team_blend.py
"""
import importlib.util
import itertools
from pathlib import Path

import polars as pl

from bang_bang.data.build import build_games
from bang_bang.baselines.team_baselines import EloTracker, netrtg_predictions
from bang_bang.eval.metrics import team_outcome_metrics

_spec = importlib.util.spec_from_file_location(
    "predict_mod", str(Path(__file__).resolve().parent / "predict.py"))
predict_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predict_mod)

W_ELO = [1.0, 0.7, 0.5, 0.3]
HOME_OFFSETS = [80.0, 100.0, 120.0]


def _season_games(season: str) -> pl.DataFrame:
    nba = predict_mod._nba_team_ids(season)
    return predict_mod._filter_nba(build_games(season), nba).sort("game_date")


def _elo_walk(games, home_offset, ratings0=None, last0=None):
    """Walk ELO over games. Returns (pred map, final ratings, final last_played)."""
    tr = EloTracker(home=home_offset)
    if ratings0 is not None:
        tr.ratings = dict(ratings0)
    if last0 is not None:
        tr.last_played = dict(last0)
    preds = {}
    for row in games.iter_rows(named=True):
        preds[row["game_id"]] = tr.predict(
            row["home_team_id"], row["away_team_id"], row["game_date"])
        tr.update(row["home_team_id"], row["away_team_id"],
                  row["home_pts"], row["away_pts"], row["game_date"])
    return preds, tr.ratings, tr.last_played


def _score(games, warm, pmap) -> dict:
    rows = [(row["game_id"], pmap[row["game_id"]], row["home_pts"], row["away_pts"])
            for row in games.iter_rows(named=True) if row["game_id"] in warm]
    return team_outcome_metrics(
        pl.DataFrame(rows, schema=["game_id", "p", "home_pts", "away_pts"]), "p")


def _blend_score(games, warm, elo_map, nr_map, w) -> dict:
    rows = [(row["game_id"],
             w * elo_map[row["game_id"]] + (1 - w) * nr_map[row["game_id"]],
             row["home_pts"], row["away_pts"])
            for row in games.iter_rows(named=True) if row["game_id"] in warm]
    return team_outcome_metrics(
        pl.DataFrame(rows, schema=["game_id", "p", "home_pts", "away_pts"]), "p")


def main():
    g245, g2526, g2324 = (_season_games(s) for s in ("2024-25", "2025-26", "2023-24"))
    for s, g in (("2024-25", g245), ("2025-26", g2526), ("2023-24", g2324)):
        print(f"{s}: {len(g)} filtered games")

    print("computing netrtg predictions...")
    nr245 = {r["game_id"]: r["netrtg_pred"]
             for r in netrtg_predictions(g245).iter_rows(named=True)}
    nr2526 = {r["game_id"]: r["netrtg_pred"]
              for r in netrtg_predictions(g2526).iter_rows(named=True)}
    warm245, warm2526 = predict_mod._warmup_mask(g245), predict_mod._warmup_mask(g2526)

    # ---------- step 1: tune (w, home) on 2024-25 (in-sample)
    print("\n== tuning on 2024-25 (in-sample, fresh walk) ==")
    best = None
    for w, home in itertools.product(W_ELO, HOME_OFFSETS):
        elo, _, _ = _elo_walk(g245, home)
        m = _blend_score(g245, warm245, elo, nr245, w)
        if best is None or m["bet_ev_per_unit"] > best[0]["bet_ev_per_unit"]:
            best = (m, w, home)
        print(f"  w={w:.1f} home={home:3.0f}: acc={m['acc']:.3f} "
              f"brier={m['brier']:.4f} betEV={m['bet_ev_per_unit']:+.3f}/u (n={m['n']})")
    m_best, w_best, home_best = best
    print(f"  -> tuned: w={w_best}, home={home_best:.0f} "
          f"(acc={m_best['acc']:.3f}, betEV={m_best['bet_ev_per_unit']:+.3f}/u)")

    # ---------- step 2: evaluate on 2025-26 (holdout)
    print("\n== 2025-26 (HOLDOUT) ==")
    elo_fresh, _, _ = _elo_walk(g2526, home_best)
    _, t23, l23 = _elo_walk(g2324, home_best)
    _, t24, l24 = _elo_walk(g245, home_best, ratings0=t23, last0=l23)
    elo_carry, _, _ = _elo_walk(g2526, home_best, ratings0=t24, last0=l24)
    _, td23, ld23 = _elo_walk(g2324, 100.0)
    _, td24, ld24 = _elo_walk(g245, 100.0, ratings0=td23, last0=ld23)
    elo_default, _, _ = _elo_walk(g2526, 100.0, ratings0=td24, last0=ld24)
    elo_est, _, _ = _elo_walk(g2526, 100.0)

    arms = [
        (f"a) blend fresh  w={w_best} home={home_best:.0f}",
         lambda: _blend_score(g2526, warm2526, elo_fresh, nr2526, w_best)),
        (f"b) blend carry  w={w_best} home={home_best:.0f}",
         lambda: _blend_score(g2526, warm2526, elo_carry, nr2526, w_best)),
        ("c) elo_538 carry (LIVE DEFAULT)",
         lambda: _score(g2526, warm2526, elo_default)),
        ("d) elo_538 fresh (established)",
         lambda: _score(g2526, warm2526, elo_est)),
    ]
    results = {}
    for name, fn in arms:
        results[name] = fn()
        m = results[name]
        print(f"  {name:34s} acc={m['acc']:.3f} brier={m['brier']:.4f} "
              f"betEV={m['bet_ev_per_unit']:+.3f}/u (n={m['n']})")

    bar = results["c) elo_538 carry (LIVE DEFAULT)"]["bet_ev_per_unit"]
    winners = [k for k in results if k.startswith(("a)", "b)"))
               and results[k]["bet_ev_per_unit"] > bar]
    print(f"\n  bar (live default) betEV={bar:+.3f}/u -> "
          f"{'BLEND SHIPS: ' + ', '.join(winners) if winners else 'no blend beats it; default stands'}")


if __name__ == "__main__":
    main()
