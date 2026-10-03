"""Sanity-check player predictions on a 30-day window."""
import datetime
import polars as pl
from bang_bang.data.build import build_player_games
from bang_bang.data.fetch import RAW_DIR
from bang_bang.baselines.player_baselines import predict_per36_decay, predict_marcel

asof_d = datetime.date(2025, 12, 1)
end_d = asof_d + datetime.timedelta(days=30)
pg_all = build_player_games("2025-26")
pg = pg_all.filter(pl.col("game_date") < asof_d)
act = (pg_all.filter((pl.col("game_date") >= asof_d) & (pl.col("game_date") < end_d))
       .group_by("player_id", "player_name")
       .agg(pl.col("pts").sum().alias("actual_pts"),
            pl.col("reb").sum().alias("actual_reb"),
            pl.col("ast").sum().alias("actual_ast")))
act = act.filter(pl.col("actual_pts") >= 30).sort("actual_pts", descending=True)

prev = pl.read_parquet(RAW_DIR / "2024-25_season_stats.parquet")
mar = predict_marcel(pg, prev, asof_d)
dec = predict_per36_decay(pg, asof_d)

m = act.join(mar.select(["player_id", "pred_pts", "pred_reb", "pred_ast"]), on="player_id", how="left") \
        .join(dec.select(["player_id", pl.col("pred_pts").alias("dec_pts")]), on="player_id", how="left")
print(m.head(15).select(["player_name", "actual_pts", "actual_reb", "actual_ast", "pred_pts", "dec_pts"]))
print()
print("avg actual pts:", round(act["actual_pts"].mean(), 1),
      "| avg marcel pred:", round(mar["pred_pts"].mean(), 1),
      "| avg decay pred:", round(dec["pred_pts"].mean(), 1))
