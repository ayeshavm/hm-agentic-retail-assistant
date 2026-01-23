from __future__ import annotations

from pathlib import Path
import polars as pl


def build_item_neighbors(
    train_path: Path,
    seed_items_per_user: int = 10,
    neighbors_per_item: int = 50,
) -> pl.DataFrame:
    """
    Build item->top neighbor list using co-occurrence counts on TRAIN.

    Returns DataFrame:
      item | neighbor | cnt
    """
    train = pl.scan_parquet(train_path)

    # For each user, keep top-N items to limit pair explosion
    seeds = (
        train.sort(["customer_id", "implicit_score"], descending=[False, True])
        .group_by("customer_id")
        .head(seed_items_per_user)
        .select(["customer_id", "article_id"])
    )

    # Self-join on user to generate pairs
    left = seeds.rename({"article_id": "item"})
    right = seeds.rename({"article_id": "neighbor"})

    pairs = (
        left.join(right, on="customer_id", how="inner")
        .filter(pl.col("item") != pl.col("neighbor"))
        .select(["item", "neighbor"])
    )

    # Count co-occurrences
    counts = (
        pairs.group_by(["item", "neighbor"])
        .agg(pl.len().alias("cnt"))
        .sort(["item", "cnt"], descending=[False, True])
    )

    # Keep top neighbors per item
    top_neighbors = counts.group_by("item").head(neighbors_per_item)

    return top_neighbors.collect(engine="streaming")


def build_cooccurrence_recs(
    train_path: Path,
    neighbors_df: pl.DataFrame,
    k: int = 12,
    seed_items_per_user: int = 10,
    max_users: int | None = 100_000,
) -> pl.DataFrame:
    """
    For each user:
    - take top seed_items_per_user items (train) as seeds
    - score candidate neighbors by summed co-occurrence counts
    - exclude items already seen in train
    - return top-k
    """
    train = pl.scan_parquet(train_path)

    # users subset
    users_lf = train.select("customer_id").unique()
    users = users_lf.limit(max_users).collect(engine="streaming") if max_users else users_lf.collect(engine="streaming")

    # user seeds
    seeds = (
        train.sort(["customer_id", "implicit_score"], descending=[False, True])
        .group_by("customer_id")
        .head(seed_items_per_user)
        .select(["customer_id", "article_id"])
        .collect(engine="streaming")
    )

    # seen items per user
    user_seen = (
        train.group_by("customer_id")
        .agg(pl.col("article_id").unique().alias("seen"))
        .collect(engine="streaming")
    )

    # Expand seeds with neighbors via join on item
    neigh = neighbors_df.lazy()

    seed_items = seeds.rename({"article_id": "item"}).lazy()
    scored = (
        seed_items.join(neigh, on="item", how="inner")
        .group_by(["customer_id", "neighbor"])
        .agg(pl.col("cnt").sum().alias("score"))
        .sort(["customer_id", "score"], descending=[False, True])
        .collect(engine="streaming")
    )

    # Attach seen list, filter out seen, then take top-k neighbors per user
    base = users.join(user_seen, on="customer_id", how="left").with_columns(
        pl.col("seen").fill_null(pl.lit([]))
    )

    scored2 = scored.join(base.select(["customer_id", "seen"]), on="customer_id", how="inner")

    # Filter neighbors not in seen per row; then take top-k
    def not_seen(neighbor: str, seen: list[str]) -> bool:
        return neighbor not in set(seen)

    filtered = (
        scored2.with_columns(
            pl.struct(["neighbor", "seen"])
            .map_elements(lambda s: not_seen(s["neighbor"], s["seen"]), return_dtype=pl.Boolean)
            .alias("keep")
        )
        .filter(pl.col("keep") == True)
        .drop(["keep", "seen"])
    )

    recs = (
        filtered.group_by("customer_id")
        .agg(pl.col("neighbor").head(k).alias("recs"))
        .select(["customer_id", "recs"])
    )

    return recs


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    train_path = repo / "runs" / "day3" / "train_interactions.parquet"

    neighbors_path = repo / "runs" / "day3" / "item_neighbors.parquet"
    recs_path = repo / "runs" / "day3" / "recs_cooccurrence.parquet"

    neighbors_df = build_item_neighbors(train_path, seed_items_per_user=10, neighbors_per_item=50)
    neighbors_df.write_parquet(neighbors_path)
    print(f"✅ Wrote neighbors: {neighbors_path}")

    recs_df = build_cooccurrence_recs(
        train_path,
        neighbors_df=neighbors_df,
        k=12,
        seed_items_per_user=10,
        max_users=100_000,
    )
    recs_df.write_parquet(recs_path)
    print(f"✅ Wrote recs: {recs_path}")
    print(recs_df.head(3))


if __name__ == "__main__":
    main()
