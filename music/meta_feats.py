"""The explicit-feedback and demographic columns the pipeline never loaded.

Six of eight item_metadata columns and all eight user_metadata columns were
unused: the model saw only track_duration, artist_name and genres. Everything it
knew about a track therefore came from PLAY counts.

That matters specifically for new tracks, which is where the model is weak --
7.5% of available value captured against 44% for repeats. With no user-item
history to lean on, a new track is scored from popularity and trend alone, and
those say how often a track is played, not whether it is any good or whether
this particular user would finish it:

  * like / dislike counts are explicit quality, orthogonal to play volume
  * download count is intent to keep -- a downloaded track gets listened
    THROUGH, and the metric scores fraction listened, not clicks
  * a user's own like/dislike/download counts calibrate how expressive they are
  * top_genre matching the track's genres is direct personalisation for an item
    with no interaction history at all

Caveat: these counts are static totals with no timestamp, so they may partly
reflect activity inside the evaluation window. Offline gains here could be
flattered relative to the leaderboard.
"""
import polars as pl

ITEM_FEATS = ["track_like_count", "track_dislike_count", "track_download_count",
              "like_ratio", "like_per_play", "download_ratio"]
USER_FEATS = ["u_liked", "u_disliked", "u_downloaded", "u_like_rate",
              "u_age_bin", "u_children", "u_gender", "genre_match"]
META_FEATS = ITEM_FEATS + USER_FEATS


def load_meta():
    item = pl.read_csv("item_metadata.csv",
                       columns=["item_id", "track_like_count", "track_dislike_count",
                                "track_download_count", "track_genres_list"])
    item = item.with_columns(
        (pl.col("track_like_count") /
         (pl.col("track_like_count") + pl.col("track_dislike_count") + 1)).alias("like_ratio"),
        (pl.col("track_download_count") /
         (pl.col("track_like_count") + 1)).alias("download_ratio"),
    )
    user = pl.read_csv("user_metadata.csv")
    user = user.select(
        "user_id",
        pl.col("user_liked_track_count").alias("u_liked"),
        pl.col("user_disliked_track_count").alias("u_disliked"),
        pl.col("user_downloaded_track_count").alias("u_downloaded"),
        (pl.col("user_liked_track_count") /
         (pl.col("user_liked_track_count") + pl.col("user_disliked_track_count") + 1)
         ).alias("u_like_rate"),
        pl.col("age_bin").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("u_age_bin"),
        pl.col("children").cast(pl.Int32).alias("u_children"),
        pl.col("gender").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("u_gender"),
        pl.col("top_genre").alias("_top_genre"),
    )
    return item, user


def add_meta(C, item, user):
    """Attach the unused columns to a candidate table."""
    out = C.join(item, on="item_id", how="left").join(user, on="user_id", how="left")
    out = out.with_columns(
        (pl.col("track_like_count") / (pl.col("item_pop") + 1)).alias("like_per_play"),
        pl.when(pl.col("_top_genre").is_null() | pl.col("track_genres_list").is_null())
          .then(0)
          .otherwise(pl.col("track_genres_list")
                     .str.contains(pl.col("_top_genre").fill_null(" "), literal=True)
                     .cast(pl.Int32))
          .alias("genre_match"),
    ).drop(["_top_genre", "track_genres_list"])
    return out.with_columns([pl.col(c).fill_null(0) for c in META_FEATS])
