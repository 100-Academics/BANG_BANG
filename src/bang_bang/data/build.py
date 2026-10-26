"""Normalize raw parquets into analysis tables under data/raw/.

Builds per season:
  games.parquet: one row per completed game
    home/away: team_id, team_name, pts, fg3m, fg3a, fgm, oreb, tov, fta,
               poss, ortg, drtg (opponent's ortg), pace
  player_games.parquet: one row per player-game (from player logs)
"""

from __future__ import annotations

import polars as pl

from bang_bang.data.fetch import RAW_DIR

FT_RATIO = 0.44

TEAM_STATS = ["pts", "fgm", "fg3m", "fg3a", "oreb", "tov", "fta", "reb", "ast", "stl", "blk"]


def _with_poss(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (
            pl.col("fgm") - pl.col("oreb") + pl.col("tov") + FT_RATIO * pl.col("fta")
        ).alias("poss"),
    ).with_columns(
        (pl.col("pts") / pl.col("poss")).alias("ortg"),
    )


def build_games(season: str) -> pl.DataFrame:
    p = RAW_DIR / f"{season}_games.parquet"
    tl = pl.read_parquet(RAW_DIR / f"{season}_team_logs.parquet")
    sched = pl.read_parquet(RAW_DIR / f"{season}_schedule.parquet")

    t = tl.select([
        pl.col("GAME_ID").alias("game_id"),
        pl.col("GAME_DATE").str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d").alias("game_date"),
        pl.col("TEAM_ID").cast(pl.Int32).alias("team_id"),
        pl.col("TEAM_ABBREVIATION").alias("team_name"),
        *[pl.col(c).cast(pl.Float64).alias(c.lower()) for c in
          ["PTS", "FGM", "FG3M", "FG3A", "OREB", "TOV", "FTA", "REB", "AST", "STL", "BLK"]],
    ])
    t = _with_poss(t)
    # nba_api returns exact-duplicate rows for in-progress seasons (hit 2026-27)
    t = t.unique(subset=["game_id", "team_id"])

    s = sched.select(["game_id", "home_team_id", "away_team_id", "status"])

    joined = t.join(s, on="game_id", how="left")
    joined = joined.with_columns(
        pl.when(pl.col("team_id") == pl.col("home_team_id"))
        .then(pl.lit("home")).otherwise(pl.lit("away")).alias("side"),
    )

    home = joined.filter(pl.col("side") == "home").drop(["home_team_id", "away_team_id"])
    away = joined.filter(pl.col("side") == "away").drop(["home_team_id", "away_team_id"])

    out = home.join(away, on="game_id", how="left", suffix="_away").rename({
        "pts": "home_pts", "pts_away": "away_pts",
        "team_name": "home_team_name", "team_name_away": "away_team_name",
        "team_id": "home_team_id", "team_id_away": "away_team_id",
    }).with_columns(
        pl.col("ortg_away").alias("home_drtg"),
        pl.col("ortg").alias("away_drtg"),
        pl.max_horizontal(pl.col("poss"), pl.col("poss_away")).alias("pace"),
    )
    out = out.filter(pl.col("status").str.starts_with("Final"))
    out = out.sort("game_date")
    out.write_parquet(p)
    return out


def build_player_games(season: str) -> pl.DataFrame:
    p = RAW_DIR / f"{season}_player_games.parquet"
    plg = pl.read_parquet(RAW_DIR / f"{season}_player_logs.parquet")
    out = plg.select([
        pl.col("GAME_ID").alias("game_id"),
        pl.col("GAME_DATE").str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d").alias("game_date"),
        pl.col("PLAYER_ID").cast(pl.Int32).alias("player_id"),
        pl.col("PLAYER_NAME").alias("player_name"),
        pl.col("TEAM_ID").cast(pl.Int32).alias("team_id"),
        pl.col("TEAM_ABBREVIATION").alias("team_abbrev"),
        *[pl.col(c).cast(pl.Float64).alias(c.lower())
          for c in ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                    "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PTS"]],
    ])
    out = out.unique(subset=["game_id", "player_id"])  # in-progress seasons double rows
    out = out.sort("game_date", "player_id")
    out.write_parquet(p)
    return out
