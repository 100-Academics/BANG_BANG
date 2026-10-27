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
    deep-carries ELO from the earliest available season, walks to asof, predicts all
    remaining games + per-team proj. wins; player 30-day projections (marcel, per36_decay
    once >=8 games, with 50/50 implicit+explicit minutes blend)
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
Scoring convention here: warmup games are SKIPPED (no ELO updates) — matches
eval_team.py. (exp_team_blend.py instead updates on all games and only filters
scoring — that convention scores ~+1.8pp acc higher; never mix the two.)
| arm | acc | brier | betEV/u |
|---|---|---|---|
| fresh+unfiltered | 0.648 | 0.2156 | +0.295 |
| fresh+filtered | 0.649 | 0.2154 | +0.298 |
| deep-carry+unfiltered | 0.670 | 0.2191 | +0.339 |
| **deep-carry+filtered (ships)** | **0.674** | 0.2192 | **+0.348** |

- fresh+unfiltered exactly reproduces the established 0.648/+0.295 → pipeline validated.
- **Carrying ELO ratings across seasons improves the holdout** (+2.6pp acc, +0.053/u).
- "filtered" = drop non-NBA junk games from the walk (see data hygiene below); neutral-to-slight.
- Est. spread on the board = MoV linear calibration on ELO diff (no home/rest bonus),
  ~29 ELO per point over the 2-season walk. Not a market line.
- **Vegas line baseline NOT yet implemented** — ESPN public API does NOT expose historical
  spreads/totals (scoreboard `odds` empty, boxscore 404s for past games). Options: balldontlie
  (needs key; has odds endpoints), or scraping. **This is the #1 missing baseline.**

### Teams — carry depth + netrtg blend (2026-10-26, scripts/exp_team_blend.py)
NOTE: this harness uses the UPDATE-ALL convention (warmup games update ELO,
only scoring is warmup-filtered) — ~+1.8pp acc higher than the skip-warmup
A/B table above. Compare arms only WITHIN a harness.
Tuned w_elo × home_offset on 2024-25 in-sample (fresh walk); holdout = 2025-26:
| arm (2025-26 holdout) | acc | brier | betEV/u |
|---|---|---|---|
| ELO fresh, home=80 | 0.671 | 0.2086 | +0.343 |
| ELO carry 3-season, home=80 | 0.679 | 0.2111 | +0.357 |
| **ELO carry 3-season, home=100** | **0.678** | 0.2132 | **+0.356** |
| ELO fresh, home=100 (established) | 0.666 | 0.2116 | +0.333 |
- **netrtg_pyth blend: NO added value** — at the winning home offset (80),
  blending monotonically hurts in-sample (w=1.0 +0.315/u > w=0.7 +0.306 > w=0.5
  +0.301 > w=0.3 +0.280); tuning picked w=1.0 (no blend), so no w<1 config was
  ever competitive enough to test on the holdout. Do not re-try a linear
  ELO+netrtg blend.
- **home=80: NOT shipped** — strong in-sample (home=80 +0.315 vs home=100 +0.280 on 2024-25)
  but a tie on holdout (+0.357 vs +0.356, ~1 game) → "basically the same" → keep +100.
- **Deep carry: SHIPPED** — in the same-convention A/B (skip-warmup, above):
  2-season carry 0.672/+0.344 → deep (3-season) carry 0.674/+0.348. Modest,
  within betEV noise, but a no-parameter use-all-history policy and both
  conventions agree on the ordering (fresh < 2-season < 3-season). predict.py
  now deep-carries for every target (generalizes to future years).

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
2024-25 only (MAE 39.9 vs 41.1 but ρ 0.764 vs 0.839). Default stays per36_decay.

### Players — half-life sweep + minutes model (2026-10-26)
`scripts/exp_decay_sweep.py` — pts MAE improves monotonically with half-life in all 3
windows: hl=14 45.7/44.6/41.1 → hl=28 45.1/43.4/39.5; Spearman within noise.
**SHIPPED hl=28** (predict_per36_decay default).

`scripts/exp_minutes_model.py` — explicit walk-forward regression of next-30d total
minutes on (m8, m30, gp30), fit strictly on pre-asof history (all available prior
seasons + current season before asof; ~54-60k samples), vs implicit minutes
(decay-weighted trailing × 13.6 games/30d). Per-36 rates identical (hl=28):
| window | implicit pts MAE | explicit | 50/50 blend |
|---|---|---|---|
| 2025-26 @ 2025-12-01 (HOLDOUT) | 45.1 | 36.6 | **37.7** |
| 2025-26 @ 2026-01-15 (HOLDOUT) | 43.4 | 43.9 | **40.1** |
| 2024-25 @ 2024-12-01 (in-sample) | 39.5 | 30.2 | **30.7** |
- **50/50 minutes blend: SHIPPED** — beats implicit on pts MAE in ALL 3 windows
  (incl. both holdout), Spearman wins/ties. Explicit alone wins 2/3 (loses 1.2% in the
  mid-Jan holdout window) → the blend is the robust choice. Wired into
  predict_per36_decay(minutes_model=fit_minutes_model(...)) in predict.py.
- Explicit minutes with <30k training samples (e.g. 2024-25 window trained on one prior
  season only) is unreliable (MAE 70.3) — always fit on >=2 full seasons.

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
- polars DataFrames have no `.round()` in this version — `with_columns([pl.col(c).round(1) ...])`.
- Background terminal sessions here may lack `~/.local/bin` on PATH — invoke uv by full path
  `/c/Users/steve/.local/bin/uv`.

## NOT yet tried (candidate next experiments)
1. **Vegas line baseline** — **parked 2026-10-26**: balldontlie key is in `.env`
   (BALLDONTLIE_API_KEY, free tier verified working) but odds/props endpoints are
   GOAT-tier only (401). ESPN scoreboard carries NO odds (even live). User chose to skip
   for now; requires either GOAT upgrade or a scraper before claiming "beats Vegas".
2. O/U win-rate metric on player totals (needs totals lines → same balldontlie dependency).
3. Postseason series model (separate technique per AGENTS.md).
4. DunksAndThrees EPM / Cleaning The Glass per-36 as *features* for a regression model.
(Tried & settled 2026-10-26: decay half-life sweep → hl=28; explicit minutes model →
50/50 blend shipped; ELO+netrtg blend → no added value; home-offset tuning → kept +100.)

## Conventions
- A baseline ships only if it beats the current default on held-out data; "basically the same" → keep the simpler one.
- Record every experiment here (result + script + date).
- No `.env` key committed; balldontlie key goes in `.env` as `BALLDONTLIE_API_KEY=...`.
