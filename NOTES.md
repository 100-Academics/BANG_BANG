# BANG_BANG — NBA Projection Model

Durable record of decisions, what was tried, and what was NOT tried.
Keep this current — it exists so no experiment gets re-run.

## Scope (from AGENTS.md)
- Player component projections (PTS/REB/AST/STL/BLK) over **1 month → end of season**
- Team win projections over the same time spans
- Postseason series winners: **separate technique** (not started)
- Data sources: nba_api (primary, keyless), balldontlie (needs free API key in `.env`),
  basketball-reference/statmuse/cleaningtheglass/dunksandthrees/bball-index as references

## Environment
- uv project, Python 3.11, polars + sklearn + nba_api. Always `uv run`.
- Data cached to `data/raw/*.parquet` (gitignored). Fetch: `uv run python scripts/fetch_seasons.py`
  (2023-24, 2024-25, 2025-26; ~5 min, 91 API calls/season batch)
- **Holdout discipline: 2025-26 is the clean holdout season. 2024-25 in-sample for diagnosis.**
  2026-27 (live) fetched via `scripts/fetch_live.py`.
- Today = 2026-10-26. 2026-27 in progress (104 games played; 30 teams, 82-86 games each).

## Layout
- `src/bang_bang/data/fetch.py` — nba_api fetchers (schedule, team logs, per-team player logs, season stats w/ AGE)
- `src/bang_bang/data/build.py` — builds `games.parquet` (per-game, home/away box + poss/ortg/drtg) and `player_games.parquet`
- `src/bang_bang/baselines/team_baselines.py` — home_floor, elo_538, netrtg_pyth
- `src/bang_bang/baselines/player_baselines.py` — per36_flat, per36_decay, marcel
- `src/bang_bang/eval/metrics.py` — acc/brier/logloss/bet-EV (teams); MAE/Spearman (players)
- `scripts/eval_team.py [season]`, `scripts/eval_player_window.py <season> <prior> <asof>`
- **Live entry points (added 2026-10-26):**
  - `scripts/fetch_live.py [season] [refresh]` — fetch an in-progress season (default 2026-27)
  - `scripts/predict.py [season] [asof]` — A/B-validates team config on the prior season,
    walks ELO to asof, predicts all remaining games + per-team proj. wins; player 30-day
    projections (marcel, per36_decay once >=8 games)
  - `scripts/board.py [season]` — renders `data/predictions/board_{season}.html` widget
  - Outputs in `data/predictions/` (gitignored): team_preds, team_preds_games, player_preds

## Baselines — settled results

### Teams (game outcome, P(home win), walk-forward, 10-game warmup)
| baseline | 2025-26 acc | 2025-26 betEV/u | 2024-25 acc | 2024-25 betEV/u |
|---|---|---|---|---|
| home_floor (58%) | 0.554 | +0.109 | 0.550 | +0.100 |
| **elo_538** | **0.648** | **+0.295** | **0.636** | **+0.272** |
| netrtg_pyth | 0.616 | +0.238 | 0.607 | +0.213 |

- ELO config: K=20*(1+(MoV-10)/20) clamped [10,30], +100 home, rest bonus 30/day past 3 days off capped 150.
- Both seasons clear the AGENTS.md 60% bar and are consistent.

### Teams — live model A/B (2026-10-26, on 2025-26 holdout, warmup=10g)
| arm | acc | brier | betEV/u |
|---|---|---|---|
| fresh+unfiltered | 0.648 | 0.2156 | +0.295 |
| fresh+filtered | 0.649 | 0.2154 | +0.298 |
| carry+unfiltered | 0.671 | 0.2168 | +0.343 |
| **carry+filtered (ships)** | **0.672** | 0.2169 | **+0.344** |

- fresh+unfiltered exactly reproduces the established 0.648/+0.295 → pipeline validated.
- **Carrying ELO ratings across seasons improves the holdout** (+2.4pp acc, +0.049/u) → live
  predictions start from end-of-prior-season ratings (walked from pre-prior).
- "filtered" = drop non-NBA junk games from the walk (see data hygiene below); neutral-to-slight.
- Est. spread on the board = MoV linear calibration on ELO diff (no home/rest bonus),
  ~29 ELO per point over the 2-season walk. Not a market line.
- **Vegas line baseline NOT yet implemented** — ESPN public API does NOT expose historical
  spreads/totals (scoreboard `odds` empty, boxscore 404s for past games). Options: balldontlie
  (needs key; has odds endpoints), or scraping. **This is the #1 missing baseline.**

### Players (30-day component totals; window eval, MAE + Spearman ranking)
2025-26, asof 2025-12-01 (n≈330 matched):
| baseline | pts MAE | pts ρ | reb MAE | ast MAE |
|---|---|---|---|---|
| per36_flat | 70.0 | 0.630 | 29.2 | 19.7 |
| **per36_decay** (hl=14d) | **45.7** | **0.766** | **17.9** | **12.0** |
| marcel (2024-25 prior) | 48.1 | 0.637 | 16.6 | 12.4 |

- **per36_decay is the default baseline** (best on MAE and by far best on ranking quality).
- Marcel wins reb MAE slightly in Dec-25; per36_decay wins ast/pts MAE and all Spearman.

Multi-window stability (confirms the default):
| window | per36_flat pts | per36_decay pts | marcel pts |
|---|---|---|---|
| 2025-26 @ 2025-12-01 | 70.0 / ρ0.630 | **45.7 / ρ0.766** | 48.1 / ρ0.637 |
| 2025-26 @ 2026-01-15 | 67.9 / ρ0.602 | **44.6 / ρ0.723** | 57.4 / ρ0.526 |
| 2024-25 @ 2024-12-01 | 61.9 / ρ0.695 | 41.1 / ρ0.839 | **39.9** / ρ0.764 |

per36_decay best-or-tie on MAE and best Spearman in all 3 windows. Marcel competitive on
2024-25 only (MAE 39.9 vs 41.1 but ρ 0.764 vs 0.839). Default stays per36_decay (hl=14d).

## Data hygiene (found 2026-10-26, all handled in fetch.py/build.py/predict.py)
- **nba_api returns exact-duplicate rows for in-progress seasons** (2026-27: every team-log
  and player-log row doubled). Deduped: fetch.py `.unique()` on logs; build.py
  `.unique(subset=[game_id,team_id])` / `[game_id,player_id]`. Old seasons were clean.
- **ScheduleLeagueV2 includes non-NBA games**: 2025-26 had 12 (Oct preseason vs
  Loong-Lions/Jerusalem B.C./United + Feb "Team Melo/Stripes/Stars/World" 3x3-style),
  2026-27 has 1 (Trail Blazers vs "Lions"). games.parquet also includes playoffs + these
  junk games. predict.py filters walks to NBA teams = teams with >=15 scheduled games that season.
- Schedule has **TBD placeholder rows** (team ids 0, null names, Dec dates) and future-game
  `status` carries the tip-off time ("7:00 pm ET"), not "Scheduled". Use
  `status.str.starts_with("Final")` — "Final/OT" exists (5 games in 2026-27).
- 2026-27: 30 teams, 82-86 games each (1157 remaining + 104 played asof 10/26).

## Bugs found & fixed (do not re-investigate)
- nba_api `gameDate` is a **string** `MM/DD/YYYY HH:MM:SS` (schedule) / ISO (logs) — cast via strptime.
- `AVAILABLE_FLAG` arrives as **string** — cast before comparing to 1.
- `pl.concat(how="diagonal")` chokes on empty team frames (string dtypes) → use `diagonal_relaxed`.
- **GAMES_PER_MONTH**: teams play ~82/181 days → ~13.6 games per 30 days, NOT 4.3.
  The 4.3 error (treating it like weekly) made per-36 baselines predict 1/3 of reality.
- Polars `&` binds tighter than `==` in filters: always parenthesize `(col==x) & (col<y)`.
- polars DataFrames have no `.to_string()` — print the df directly.

## NOT yet tried (candidate next experiments)
1. **Vegas line baseline** (balldontlie w/ key) — required before claiming "beats Vegas".
2. Decay half-life sweep (7/10/14/21/28 days) on the held-out window.
3. Minutes model as an explicit target (injury/rest-based) instead of trailing minutes.
4. O/U win-rate metric on player totals (needs totals lines → same balldontlie dependency).
5. Team: blending ELO + net-rating Pythagorean; home advantage tuned from data.
6. Postseason series model (separate technique per AGENTS.md).
7. DunksAndThrees EPM / Cleaning The Glass per-36 as *features* for a regression model.

## Conventions
- A baseline ships only if it beats the current default on held-out data; "basically the same" → keep the simpler one.
- Record every experiment here (result + script + date).
- No `.env` key committed; balldontlie key goes in `.env` as `BALLDONTLIE_API_KEY=...`.
