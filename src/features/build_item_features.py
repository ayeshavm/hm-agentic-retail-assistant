from __future__ import annotations

from pathlib import Path
import polars as pl


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = repo_root / "data" / "raw"
    out_dir = repo_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    articles_csv = raw_dir / "articles.csv"
    out_parquet = out_dir / "item_features.parquet"

    if not articles_csv.exists():
        raise FileNotFoundError(f"Missing raw file: {articles_csv}")

    # H&M articles.csv contains many columns; we’ll keep a small stable set.
    # (These column names exist in the Kaggle H&M dataset; if yours differ,
    #  adjust the list based on Day 1 schema inspection.)
    keep_cols = [
        "article_id",
        "product_type_no",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "department_name",
        "index_name",
        "index_group_name",
        "section_name",
        "garment_group_name",
    ]

    lf = pl.scan_csv(articles_csv)

    # Only select columns that exist to avoid crashes if slight schema differences
    schema_names = lf.collect_schema().names()
    existing = [c for c in keep_cols if c in schema_names]
    if "article_id" not in existing:
        raise ValueError("articles.csv must contain article_id")

    item_features = (
        lf.select(existing)
        .with_columns(pl.col("article_id").cast(pl.Utf8))
        .collect(engine="streaming")
    )

    item_features.write_parquet(out_parquet)

    print(f"✅ Wrote: {out_parquet}")
    print(f"Rows: {item_features.height:,} | Cols: {item_features.width}")


if __name__ == "__main__":
    main()
