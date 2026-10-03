"""Check player log integrity: dups, counts."""
import polars as pl
from bang_bang.data.build import build_player_games

pg = build_player_games("2025-26")
print("total rows:", len(pg))
print("unique (game_id, player_id):", pg.unique(subset=["game_id", "player_id"]).height)
dups = pg.group_by("game_id", "player_id").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
print("duplicate (game,player) pairs:", len(dups))
if len(dups):
    print(dups.head(5))
    ex = dups.head(1).row(0)
    print(pg.filter((pl.col("game_id") == ex["game_id"]) & (pl.col("player_id") == ex["player_id"])))

# per-team game counts
tl = pl.read_parquet("data/raw/2025-26_team_logs.parquet")
print("team log rows:", len(tl), "unique games:", tl["GAME_ID"].unique().count())
# schedule games
s = pl.read_parquet("data/raw/2025-26_schedule.parquet")
print("schedule finals:", s.filter(pl.col("status").str.starts_with("Final")).height)

# how many games does Brunson (1628973) have?
b = pg.filter(pl.col("player_id") == 1628973)
print("Brunson rows:", len(b), "games:", b["game_id"].unique().count())
j = pg.filter(pl.col("player_id") == 203999)
print("Jokic rows:", len(j), "games:", j["game_id"].unique().count())
dec = j.filter((pl.col("game_date") >= pl.lit(0)))
import datetime
w = j.filter((pl.col("game_date") >= datetime.date(2025,12,1)) & (pl.col("game_date") < datetime.date(2025,12,31)))
print("Jokic Dec rows:", len(w), "pts:", w["pts"].sum())
