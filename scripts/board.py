"""Render the live prediction board as a self-contained themed HTML widget.

Usage: uv run python scripts/board.py [season]
Reads data/predictions/{season}_*.parquet (from scripts/predict.py) and writes
data/predictions/board_{season}.html.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PRED_DIR = RAW_DIR.parent / "predictions"

CSS = """
body { color: var(--foreground); font: 14px/1.5 system-ui, -apple-system, 'Segoe UI', sans-serif; }
h1 { font-size: 18px; margin: 0 0 2px; font-weight: 700; }
.sub { color: var(--muted-foreground); font-size: 12px; margin-bottom: 4px; }
h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
     color: var(--muted-foreground); margin: 20px 0 6px;
     border-bottom: 1px solid var(--border); padding-bottom: 4px; font-weight: 600; }
table { border-collapse: collapse; width: 100%; }
th { text-align: left; font-size: 10.5px; color: var(--muted-foreground);
     text-transform: uppercase; letter-spacing: .04em;
     padding: 3px 8px; border-bottom: 1px solid var(--border); font-weight: 600; }
td { padding: 3.5px 8px; border-bottom: 1px solid var(--border); font-size: 13px; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.strong { font-weight: 600; }
.muted { color: var(--muted-foreground); }
.pill { display: inline-block; min-width: 34px; text-align: center;
        border-radius: 4px; padding: 0 5px; font-size: 11px; font-weight: 600;
        color: var(--foreground); background: var(--card); border: 1px solid var(--border); }
.barwrap { display: flex; align-items: center; gap: 6px; min-width: 130px; }
.bar { height: 8px; border-radius: 4px; background: var(--accent); }
.track { flex: 1; height: 8px; border-radius: 4px; background: var(--card);
         border: 1px solid var(--border); overflow: hidden; }
.foot { color: var(--muted-foreground); font-size: 11px; margin-top: 16px;
        border-top: 1px solid var(--border); padding-top: 8px; }
"""


def _pbar(p: float) -> str:
    pct = int(round(p * 100))
    return (f'<div class="barwrap"><div class="track"><div class="bar" '
            f'style="width:{pct}%"></div></div>'
            f'<span class="pill">{pct}%</span></div>')


def main(season: str):
    standings = pl.read_parquet(PRED_DIR / f"{season}_team_preds.parquet")
    games = pl.read_parquet(PRED_DIR / f"{season}_team_preds_games.parquet")
    players = pl.read_parquet(PRED_DIR / f"{season}_player_preds.parquet")
    asof = games["game_date"].min()
    sources = players["source"].unique().sort().to_list()

    out = [
        "<!doctype html><html><head><meta charset='utf-8'><style>", CSS, "</style></head><body>",
        f"<h1>BANG_BANG &middot; {season} live projections</h1>",
        f"<div class='sub'>asof {asof} &middot; teams: elo_538 (ratings carried across seasons) "
        f"&middot; players: {' + '.join(sources)}, 30-day windows &middot; "
        f"{len(games)} games left on schedule</div>",
    ]

    out.append("<h2>Projected final standings</h2>")
    out.append("<table><tr><th>team</th><th class='num'>record</th>"
               "<th class='num'>proj. remaining</th><th class='num'>proj. final</th>"
               "<th class='num'>proj. wins</th></tr>")
    for r in standings.iter_rows(named=True):
        out.append(
            f"<tr><td class='strong'>{r['team']}</td>"
            f"<td class='num muted'>{r['record']}</td>"
            f"<td class='num'>{r['proj_remaining_wins']:.1f} - {r['proj_remaining_losses']:.1f}</td>"
            f"<td class='num'>{r['proj_final_wins']:.1f} - {r['proj_final_losses']:.1f}</td>"
            f"<td class='num strong'>{r['proj_final_wins']:.1f}</td></tr>")
    out.append("</table>")

    out.append("<h2>Next 12 games (P(home) &middot; est. spread)</h2>")
    out.append("<table><tr><th>date</th><th>away @ home</th>"
               "<th>P(home)</th><th class='num'>est. spread</th></tr>")
    for r in games.head(12).iter_rows(named=True):
        sp = r["est_spread_pts"]
        sp_s = f"home {sp:+.1f}" if sp >= 0 else f"away {-sp:.1f}"
        out.append(
            f"<tr><td class='muted'>{r['game_date'].strftime('%a %m/%d')}</td>"
            f"<td>{r['away_team']} @ <span class='strong'>{r['home_team']}</span></td>"
            f"<td>{_pbar(r['p_home'])}</td>"
            f"<td class='num muted'>{sp_s}</td></tr>")
    out.append("</table>")

    out.append("<h2>Top 50 players &middot; projected 30-day totals</h2>")
    out.append("<table><tr><th></th><th>player</th><th>team</th>"
               "<th class='num'>gp</th><th class='num'>src</th><th class='num'>PTS</th>"
               "<th class='num'>REB</th><th class='num'>AST</th>"
               "<th class='num'>STL</th><th class='num'>BLK</th></tr>")
    for i, r in enumerate(players.head(50).iter_rows(named=True), 1):
        src = "decay" if r["source"] == "per36_decay" else "marcel"
        out.append(
            f"<tr><td class='num muted'>{i}</td>"
            f"<td class='strong'>{r['player_name']}</td>"
            f"<td class='muted'>{r['team_abbrev']}</td>"
            f"<td class='num muted'>{r['gp_2627']}</td>"
            f"<td class='num muted'>{src}</td>"
            f"<td class='num strong'>{r['pred_pts']:.0f}</td>"
            f"<td class='num'>{r['pred_reb']:.0f}</td>"
            f"<td class='num'>{r['pred_ast']:.0f}</td>"
            f"<td class='num'>{r['pred_stl']:.1f}</td>"
            f"<td class='num'>{r['pred_blk']:.1f}</td></tr>")
    out.append("</table>")

    out.append(
        "<div class='foot'>Team baseline validated walk-forward on 2025-26 holdout "
        "(see NOTES.md). Spread is a linear calibration of ELO diff on actual MoV "
        "&mdash; not a market line. Player season-start projections use Marcel "
        "(regressed 2025-26 totals); per-36 decay takes over once a player has "
        "&ge;8 games. No Vegas/balldontlie line available (no API key).</div>")
    out.append("</body></html>")

    p = PRED_DIR / f"board_{season}.html"
    p.write_text("\n".join(out))
    print(f"wrote {p}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-27")
