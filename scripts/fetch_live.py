"""Fetch a (live) season on demand. Run: uv run python scripts/fetch_live.py [season] [refresh]
Default season: 2026-27. Pass '1' as second arg to force refresh."""
import sys

from bang_bang.data.fetch import fetch_all

season = sys.argv[1] if len(sys.argv) > 1 else "2026-27"
refresh = len(sys.argv) > 2 and sys.argv[2] == "1"
fetch_all(season, refresh=refresh)
print("ALL DONE")
