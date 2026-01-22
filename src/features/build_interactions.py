from __future__ import annotations

from pathlib import Path
import json
import polars as pl
from datetime import date


def main() -> None:
    # --- Paths ---
    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = repo_root / "data" / "raw"
    out_dir = repo_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    transactions_csv = raw_dir / "transactions_train.csv"
    interactions_parquet = out_dir / "interactions.parquet"

    # --- Sanity check: raw file exists ---
    if not transactions_csv.exists():
        raise FileNotFoundError(f"Missing raw file: {transactions_csv}")

    # --- Load + aggregate with a streaming-friendly lazy query ---
    # We only select columns we need to keep memory low.
    # H&M transactions has columns like: t_dat, customer_id, article_id, price, sales_channel_id
    lf = (
    pl.scan_csv(
        transactions_csv,
        try_parse_dates=True,
    )
    .select(["t_dat", "customer_id", "article_id"])
    .with_columns(
        [
            pl.col("customer_id").cast(pl.Utf8),
            pl.col("article_id").cast(pl.Utf8),
            pl.col("t_dat").cast(pl.Date, strict=False),
        ]
    )
    .drop_nulls(["customer_id", "article_id", "t_dat"])
    .group_by(["customer_id", "article_id"])
    .agg(
        [
            pl.len().alias("freq"),
            pl.col("t_dat").max().alias("last_seen"),
        ]
    )
    )

    # --- Compute a simple implicit score (freq + recency) ---
    # We use a stable, explainable scoring:
    #   freq_score = log1p(freq)
    #   recency_score = exp(-days_since_last / half_life)
    #   implicit_score = freq_score * recency_score
    #
    # Half-life here is a knob. 30 days is a reasonable default.
    half_life_days = 30.0

    interactions = (
    lf.collect(engine="streaming")  # or .collect()
    .with_columns(
        [
            pl.col("freq").cast(pl.Float64).log1p().alias("freq_score"),
            (pl.lit(date.today()) - pl.col("last_seen"))
            .dt.total_days()
            .cast(pl.Float64)
            .alias("days_since_last"),
        ]
    )
    # 1) create recency_score first
    .with_columns(
        [
            (-pl.col("days_since_last") / pl.lit(half_life_days))
            .exp()
            .alias("recency_score"),
        ]
    )
    # 2) now you can safely reference recency_score
    .with_columns(
        [
            (pl.col("freq_score") * pl.col("recency_score")).alias("implicit_score"),
        ]
    )
    .select(["customer_id", "article_id", "freq", "last_seen", "implicit_score"])
    .sort(["customer_id", "implicit_score"], descending=[False, True])
)


    # --- Write Parquet ---
    interactions.write_parquet(interactions_parquet)

    print(f"✅ Wrote: {interactions_parquet}")
    print(f"Rows: {interactions.height:,} | Cols: {interactions.width}")


if __name__ == "__main__":
    main()
