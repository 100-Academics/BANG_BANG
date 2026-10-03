"""Check injury-return contamination in the Dec window."""
import datetime
import polars as pl
from bang_bang.data.build import build_player_games

pg = build_player_games("2025-26")
asof = datetime.date(2025, 12, 1)
end = asof + datetime.timedelta(days=30)

for name in ["Kawhi Leonard", "Nikola Jokić", "Jalen Brunson"]:
    sub = pg.filter(pl.col("player_name").str.contains(name.split()[0]))
    before = sub.filter(pl.col("game_date") < asof)
    window = sub.filter((pl.col("game_date") >= asof) & (pl.col("game_date") < end))
    print(f"{name}: before={len(before)} games (last: {before['game_date'].max() if len(before) else 'None'}), "
          f"window={len(window)} games, window pts={window['pts'].sum():.0f}, "
          f"ppg in window={window['pts'].mean() if len(window) else 0:.1f}")

# how many of the 337 actual players had <5 games before asof?
act = (pg.filter((pl.col("game_date") >= asof) & (pl.col("game_date") < end))
       .group_by("player_id").agg(pl.len().alias("wg"), pl.col("pts").sum().alias("pts")))
act = act.filter(pl.col("pts") >= 30)
before_cnt = (pg.filter(pl.col("game_date") < asof)
              .group_by("player_id").agg(pl.len().alias("bg")))
m = act.join(before_cnt, on="player_id", how="left").with_columns(
    pl.col("bg").fill_null(0))
print("\nactual players by pre-December games:")
print(m.group_by("bg").agg(pl.len().alias("n")).sort("bg"))
