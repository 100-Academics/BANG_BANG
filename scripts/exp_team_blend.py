"""Experiment: team ELO + net-rating Pythagorean blend + tuned home offset
(NOTES item 5).

Protocol:
  1. Tune (w_elo, home_offset) on 2024-25 (in-sample), fresh walk, filtered.
  2. Evaluate candidates on 2025-26 (HOLDOUT):
       a) best fresh blend (tuned)
       b) best carry blend (tuned config, ratings carried from 2023-24)
       c) carry+filtered elo_538  <- current live default (bar to beat: 0.672/+0.344)
       d) fresh+filtered elo_538  <- established in-season baseline
  Ship the blend only if it beats (c) on the holdout.

Usage: uv run python scripts/exp_team_blend.py
"""
import copy
import itertools

import polars as pl

from bang_bang.data.build import build_games
from bang_bang.data.fetch import RAW_DIR, fetch_schedule
from bang_bang.baselines.team_baselines import EloTracker, netrtg_predictions
from bang_bang.eval.metrics import team_outcome_metrics
from scripts.exp_decay_sweep import _nba_team_ids, _filter_nba  # noqa: F401  (helpers live in predict)

# import helpers from predict.py (same package dir)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "predict_mod", str(RAW_DIR.parent.parent / "scripts" / "predict.py"))
predict_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predict_mod)

W_ELO = [1.0, 0.7, 0.5, 0.3]
HOME_OFFSETS = [80.0, 100.0, 120.0]
SEASONS = {
    "2024-25": (build_games("2024-25"), _nba_ids("2024-25")),
}


def _nba_ids(season: str) -> set[int]:
    return predict_mod._nba_team_ids(season)


def _elo_pred(games, home_offset, ratings0=None):
    """Walk elo over games, return pred map (game_id -> p_home) and final tracker."""
    tr = EloTracker(home=home_offset)
    if ratings0 is not None:
        tr.ratings = dict(ratings0)
        tr.last_played = dict(ratings0_last if (ratings0_last := None) else {})
    preds = {}
    for row in games.iter_rows(named=True):
        p = tr.predict(row["home_team_id"], row["away_team_id"], row["game_date"])
        preds[row["game_id"]] = p
        tr.update(row["home_team_id"], row["away_team_id"],
                  row["home_pts"], row["away_pts"], row["game_date"])
    return preds, tr


def main():
    for season in ["2024-25", "2025-26", "2023-24"]:
        if (RAW_DIR / f"{season}_games.parquet").exists():
            g = build_games(season)
            nba = predict_mod._nba_team_ids(season)
            games = predict_mod._filter_nba(g, nba).sort("game_date")
            print(f"{season}: {len(games)} filtered games")
    # keep only what we need in memory
    g245 = predict_mod._filter_nba(build_games("2024-25"), _nba_ids("2024-25")).sort("game_date")
    g2526 = predict_mod._filter_nba(build_games("2025-26"), _nba_ids("2025-26")).sort("game_date")
    g2324 = predict_mod._filter_nba(build_games("2023-24"), _nba_ids("2023-24")).sort("game_date")

    # netrtg preds (fixed model, computed once per season)
    print("computing netrtg predictions...")
    nr245 = {r["game_id"]: r["netrtg_pred"] for r in
             netrtg_predictions(g245).iter_rows(named=True)}
    nr2526 = {r["game_id"]: r["netrtg_pred"] for r in
              netrtg_predictions(g2526).iter_rows(named=True)}

    warm245 = predict_mod._warmup_mask(g245)
    warm2526 = predict_mod._warmup_mask(g2526)

    # ---------- step 1: tune (w, home) on 2024-25
    print("\n== tuning on 2024-25 (in-sample, fresh walk) ==")
    best = None
    for w, home in itertools.product(W_ELO, HOME_OFFSETS):
        elo, _ = _elo_pred(g245, home)
        merged = g245.filter(pl.col("game_id").is_in(warm245))
        rows = []
        for row in merged.iter_rows(named=True):
            p = w * elo[row["game_id"]] + (1 - w) * nr245[row["game_id"]]
            rows.append((row["game_id"], p, row["home_pts"], row["away_pts"]))
        m = team_outcome_metrics(
            pl.DataFrame(rows, schema=["game_id", "p", "home_pts", "away_pts"]), "p")
        tag = ""
        if best is None or m["bet_ev_per_unit"] > best[0]["bet_ev_per_unit"]:
            best = (m, w, home)
        print(f"  w={w:.1f} home={home:.0f}: acc={m['acc']:.3f} "
              f"brier={m['brier']:.4f} betEV={m['bet_ev_per_unit']:+.3f}/u (n={m['n']})")
    m_best, w_best, home_best = best
    print(f"  -> best on 2024-25: w={w_best}, home={home_best} "
          f"(acc={m_best['acc']:.3f}, betEV={m_best['bet_ev_per_unit']:+.3f})")

    # ---------- step 2: evaluate on 2025-26 holdout
    print(f"\n== 2025-26 (HOLDOUT) ==")
    # a) best fresh blend
    elo2526, _ = _elo_pred(g2526, home_best)
    # b) best carry blend: walk 2023-24 + 2024-25 with tuned home, then 2025-26
    _, tr2324 = _elo_pred(g2324, home_best)
    _, tr245c = _elo_pred(g245, home_best, ratings0=tr2324.ratings)
    elo2526_carry, _ = _elo_pred(g2526, home_best, ratings0=tr245c.ratings)
    # c) current live default: carry+filtered elo_538 (home=100)
    _, tr2324d = _elo_pred(g2324, 100.0)
    _, tr245d = _elo_pred(g245, 100.0, ratings0=tr2324d.ratings)
    elo2526_def, _ = _elo_pred(g2526, 100.0, ratings0=tr245d.ratings)
    # d) fresh+filtered elo_538
    elo2526_fresh, _ = _elo_pred(g2526, 100.0)

    merged = g2526.filter(pl.col("game_id").is_in(warm2526))
    results = {}
    for name, elo_map in [
        (f"a) blend fresh w={w_best} home={home_best:.0f}", elo2526),
        (f"b) blend carry w={w_best} home={home_best:.0f}", elo2526_carry),
        ("c) elo_538 carry (LIVE DEFAULT)", elo2526_def),
        ("d) elo_538 fresh (established)", elo2526_fresh),
    ]:
        rows = []
        for row in merged.iter_rows(named=True):
            p = (w_best * elo_map[row["game_id"]]
                 + (1 - w_best) * nr2526[row["game_id"]]) if "blend" in name \
                else elo_map[row["game_id"]]
            rows.append((row["game_id"], p, row["home_pts"], row["away_pts"]))
        m = team_outcome_metrics(
            pl.DataFrame(rows, schema=["game_id", "p", "home_pts", "away_pts"]), "p")
        results[name] = m
        print(f"  {name:38s} acc={m['acc']:.3f} brier={m['brier']:.4f} "
              f"betEV={m['bet_ev_per_unit']:+.3f}/u (n={m['n']})")

    default_ev = results["c) elo_538 carry (LIVE DEFAULT)"]["bet_ev_per_unit"]
    winners = [k for k in ("a) blend fresh w=1.0 home=100.0", "b) blend carry w=1.0 home=100.0")
               if k in results and results[k]["bet_ev_per_unit"] > default_ev]
    print(f"\n  bar: live default betEV={default_ev:+.3f} -> "
          f"{'BLEND SHIPS: ' + str(winners) if winners else 'default stands (no blend beats it)'}")


if __name__ == "__main__":
    main()
