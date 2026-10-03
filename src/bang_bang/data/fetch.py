"""Fetch NBA data via nba_api, cached to parquet under data/raw/.

Call counts per season (kept small on purpose):
  schedule: 1
  team logs: 30 (one per team)
  player logs: 30 (one per team, returns every rostered player)
  season stats (AGE, season aggregates): 1
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def _cache_path(season: str, kind: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / f"{season}_{kind}.parquet"


def _pandas_to_pl(df) -> pl.DataFrame:
    return pl.from_pandas(df)


def _get(endpoint_cls, **kwargs):
    """Call an nba_api endpoint, return polars df of its first frame."""
    ep = endpoint_cls(**kwargs)
    frames = ep.get_data_frames()
    return _pandas_to_pl(frames[0])


def fetch_schedule(season: str, refresh: bool = False) -> pl.DataFrame:
    p = _cache_path(season, "schedule")
    if p.exists() and not refresh:
        return pl.read_parquet(p)
    from nba_api.stats.endpoints import ScheduleLeagueV2
    raw = _get(ScheduleLeagueV2, season=season)
    # normalize: keep one row per game, drop duplicates (nba_api sometimes doubles)
    out = raw.select([
        pl.col("gameId").cast(pl.String).alias("game_id"),
        pl.col("gameDate").str.slice(0, 10).str.strptime(pl.Date, "%m/%d/%Y").alias("game_date"),
        pl.col("gameStatusText").alias("status"),
        pl.col("homeTeam_teamId").cast(pl.Int32).alias("home_team_id"),
        pl.col("homeTeam_teamName").alias("home_team_name"),
        pl.col("awayTeam_teamId").cast(pl.Int32).alias("away_team_id"),
        pl.col("awayTeam_teamName").alias("away_team_name"),
        pl.col("homeTeam_score").cast(pl.Int32).alias("home_pts"),
        pl.col("awayTeam_score").cast(pl.Int32).alias("away_pts"),
    ])
    out = out.unique(subset=["game_id"])
    out.write_parquet(p)
    return out


def _all_team_ids(sched: pl.DataFrame) -> list[int]:
    ids = set(sched["home_team_id"].to_list()) | set(sched["away_team_id"].to_list())
    return sorted(ids)


def fetch_team_logs(season: str, refresh: bool = False) -> pl.DataFrame:
    p = _cache_path(season, "team_logs")
    if p.exists() and not refresh:
        return pl.read_parquet(p)
    from nba_api.stats.endpoints import TeamGameLogs
    sched = fetch_schedule(season)
    team_ids = _all_team_ids(sched)

    frames = []
    for tid in team_ids:
        df = _get(TeamGameLogs, team_id_nullable=int(tid), season_nullable=season)
        frames.append(df.filter(pl.col("AVAILABLE_FLAG").cast(pl.Int32, strict=False) == 1))
        time.sleep(0.4)
    out = pl.concat(frames, how="diagonal_relaxed")
    out.write_parquet(p)
    return out


def fetch_player_logs(season: str, refresh: bool = False) -> pl.DataFrame:
    p = _cache_path(season, "player_logs")
    if p.exists() and not refresh:
        return pl.read_parquet(p)
    from nba_api.stats.endpoints import PlayerGameLogs
    sched = fetch_schedule(season)
    team_ids = _all_team_ids(sched)

    frames = []
    for i, tid in enumerate(team_ids):
        df = _get(PlayerGameLogs, team_id_nullable=int(tid), season_nullable=season)
        frames.append(df.filter(pl.col("AVAILABLE_FLAG").cast(pl.Int32, strict=False) == 1))
        time.sleep(0.4)
        if (i + 1) % 10 == 0:
            print(f"  player logs: {i+1}/{len(team_ids)} teams done")
    out = pl.concat(frames, how="diagonal_relaxed")
    out.write_parquet(p)
    return out


def fetch_season_stats(season: str, refresh: bool = False) -> pl.DataFrame:
    """Per-player season aggregates incl. AGE (needed for Marcel)."""
    p = _cache_path(season, "season_stats")
    if p.exists() and not refresh:
        return pl.read_parquet(p)
    from nba_api.stats.endpoints import LeagueDashPlayerStats
    raw = _get(LeagueDashPlayerStats, season=season)
    out = raw.select([
        pl.col("PLAYER_ID").cast(pl.Int32).alias("player_id"),
        pl.col("PLAYER_NAME").alias("player_name"),
        pl.col("AGE").cast(pl.Float64).alias("age"),
        pl.col("GP").cast(pl.Int32).alias("gp"),
        pl.col("MIN").cast(pl.Float64).alias("min"),
        pl.col("PTS").cast(pl.Float64).alias("pts"),
        pl.col("REB").cast(pl.Float64).alias("reb"),
        pl.col("AST").cast(pl.Float64).alias("ast"),
        pl.col("STL").cast(pl.Float64).alias("stl"),
        pl.col("BLK").cast(pl.Float64).alias("blk"),
        pl.col("TOV").cast(pl.Float64).alias("tov"),
        pl.col("FG3M").cast(pl.Float64).alias("fg3m"),
    ])
    out.write_parquet(p)
    return out


def fetch_all(season: str, refresh: bool = False) -> dict[str, pl.DataFrame]:
    print(f"fetching {season}...")
    t0 = time.time()
    out = {
        "schedule": fetch_schedule(season, refresh),
        "team_logs": fetch_team_logs(season, refresh),
        "player_logs": fetch_player_logs(season, refresh),
        "season_stats": fetch_season_stats(season, refresh),
    }
    print(f"done {season} in {time.time()-t0:.0f}s")
    return out
