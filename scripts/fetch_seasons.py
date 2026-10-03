"""Fetch all needed seasons. Run: uv run python scripts/fetch_seasons.py"""
from bang_bang.data.fetch import fetch_all

SEASONS = ["2023-24", "2024-25", "2025-26"]

for s in SEASONS:
    fetch_all(s)

print("ALL DONE")
