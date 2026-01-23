from __future__ import annotations

from pathlib import Path
import polars as pl


def build_popularity_recs(
    train_path: Path,
    k: int = 12,
    max_users: int | None = 100_000,
) -> pl.DataFrame:
    """
    Popularity baseline:
    - rank items by global interaction frequency in TRAIN
    - recommend top-k to each user
    - exclude items the user already interacted with in TRAIN

    Returns DataFrame:
      customer_id | recs (list[str])
    """
    train = pl.scan_parquet(train_path)

    # Global top items by frequency
    top_items = (
        train.group_by("article_id")
        .agg(pl.len().alias("cnt"))
        .sort("cnt", descending=True)
        .select("article_id")
        .limit(5000)  # allow filtering per user
        .collect(engine="streaming")
        .get_column("article_id")
        .to_list()
    )

    # Users to evaluate
    users_lf = train.select("customer_id").unique()
    if max_users is not None:
        users = users_lf.limit(max_users).collect(engine="streaming")
    else:
        users = users_lf.collect(engine="streaming")

    # User -> set of seen items in train (as list)
    user_seen = (
        train.group_by("customer_id")
        .agg(pl.col("article_id").unique().alias("seen"))
        .collect(engine="streaming")
    )

    # Join users with seen lists
    base = users.join(user_seen, on="customer_id", how="left").with_columns(
        pl.col("seen").fill_null(pl.lit([]))
    )

    # Build recommendations by filtering top_items against each user's seen list
    # Polars doesn't have a perfect vectorized "filter list by another list" across rows in all versions,
    # so we do a small Python apply on rows (OK for max_users <= 100k).
    def recs_for_seen(seen: list[str]) -> list[str]:
        out = []
        seen_set = set(seen)
        for a in top_items:
            if a not in seen_set:
                out.append(a)
            if len(out) == k:
                break
        return out

    recs = base.with_columns(
        pl.col("seen").map_elements(recs_for_seen, return_dtype=pl.List(pl.Utf8)).alias("recs")
    ).select(["customer_id", "recs"])

    return recs


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    train_path = repo / "runs" / "day3" / "train_interactions.parquet"
    out_path = repo / "runs" / "day3" / "recs_popularity.parquet"

    df = build_popularity_recs(train_path, k=12, max_users=100_000)
    df.write_parquet(out_path)
    print(f"✅ Wrote: {out_path}")
    print(df.head(3))


if __name__ == "__main__":
    main()
