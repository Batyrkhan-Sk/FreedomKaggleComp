"""Is some "new-track" value actually the SAME SONG under a different item_id?

The task description says the data contains duplicates. item_metadata carries
track_name and artist_name, and a streaming catalogue routinely holds the same
recording several times -- single, album, remaster, deluxe reissue.

If a user played item A and in the test window plays A' (same song, different
id), then:
  * our history slot on A scores 0
  * every diagnostic counts A' as a NEW track the model failed to discover
So this value has been sitting inside the 42,321 "new" points, mis-labelled as a
discovery problem, when it is really an identity problem -- and identity is far
easier to fix than taste prediction.

This measures the size of that pocket. It is NOT in the dead list: every closed
lever assumed item_id was the unit of identity.

Cheap and decisive: if disguised-replay value is small, drop it in 15 minutes.
"""
import polars as pl

from evaluate import CUT, load, truth

train, hold, meta_dur, users = load(CUT)
t = truth(hold, meta_dur, users)

meta = pl.read_csv("item_metadata.csv",
                   columns=["item_id", "track_name", "artist_name", "track_duration"])

# identity key: normalised (title, artist). Deliberately conservative -- exact
# match after casefold/strip, so this is a LOWER bound on duplication.
norm = (pl.col("track_name").str.strip_chars().str.to_lowercase(),
        pl.col("artist_name").str.strip_chars().str.to_lowercase())
meta = meta.with_columns(norm[0].alias("tn"), norm[1].alias("an")).with_columns(
    (pl.col("tn") + pl.lit("\x1f") + pl.col("an")).alias("song"))

groups = meta.group_by("song").agg(pl.len().alias("n"))
multi = groups.filter(pl.col("n") > 1)
print(f"catalogue: {meta.height:,} item_ids -> {groups.height:,} distinct (title, artist)")
print(f"  {multi.height:,} songs exist under >1 item_id, "
      f"covering {multi['n'].sum():,} item_ids "
      f"({multi['n'].sum()/meta.height:.1%} of the catalogue)\n")

i2s = meta.select("item_id", "song")

# what each test user has already heard, at SONG level and at ITEM level
hist = (train.filter(pl.col("user_id").is_in(users.implode()))
        .select("user_id", "item_id").unique()
        .join(i2s, on="item_id", how="left"))
hist_items = hist.select("user_id", "item_id").with_columns(pl.lit(True).alias("hi"))
hist_songs = hist.select("user_id", "song").unique().with_columns(pl.lit(True).alias("hs"))

tt = (t.join(i2s, on="item_id", how="left")
      .join(hist_items, on=["user_id", "item_id"], how="left")
      .join(hist_songs, on=["user_id", "song"], how="left")
      .with_columns(pl.col("hi").fill_null(False), pl.col("hs").fill_null(False)))

total_new = tt.filter(~pl.col("hi"))["frac"].sum()
disguised = tt.filter(~pl.col("hi") & pl.col("hs"))
true_new = tt.filter(~pl.col("hi") & ~pl.col("hs"))
print(f"holdout NEW-track value {total_new:9,.0f} pts")
print(f"  DISGUISED REPLAY (same song, different item_id) "
      f"{disguised['frac'].sum():9,.0f} pts = {disguised['frac'].sum()/total_new:.1%}")
print(f"  genuinely unheard songs                        "
      f"{true_new['frac'].sum():9,.0f} pts = {true_new['frac'].sum()/total_new:.1%}")

active = t["user_id"].n_unique()
SCALE = 1.29 / (active * 50)
print(f"\nperfect capture of the disguised pocket is worth "
      f"{disguised['frac'].sum()*SCALE:+.4f} leaderboard score (upper bound)")
print(f"users affected: {disguised['user_id'].n_unique():,} of {active:,} active")
