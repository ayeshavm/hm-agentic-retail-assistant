from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import polars as pl


@dataclass(frozen=True)
class SplitPaths:
    train_path: Path
    test_path: Path


def make_time_split(
    interactions_path: Path,
    out_dir: Path,
    cutoff_date: str,
) -> SplitPaths:
    """
    Creates time-based train/test splits from interactions.parquet.

    Train: last_seen < cutoff_date
    Test:  last_seen >= cutoff_date

    Saves:
      - out_dir/train_interactions.parquet
      - out_dir/test_interactions.parquet
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_interactions.parquet"
    test_path = out_dir / "test_interactions.parquet"

    lf = pl.scan_parquet(interactions_path)

    # Ensure last_seen is a Date (should already be)
    lf = lf.with_columns(pl.col("last_seen").cast(pl.Date, strict=False))

    train_lf = lf.filter(pl.col("last_seen") < pl.lit(cutoff_date).cast(pl.Date))
    test_lf = lf.filter(pl.col("last_seen") >= pl.lit(cutoff_date).cast(pl.Date))

    train_lf.collect(engine="streaming").write_parquet(train_path)
    test_lf.collect(engine="streaming").write_parquet(test_path)

    return SplitPaths(train_path=train_path, test_path=test_path)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    interactions_path = repo / "data" / "processed" / "interactions.parquet"

    # Day 3 artifacts live here
    out_dir = repo / "runs" / "day3"
    cutoff_date = "2020-09-01"

    if not interactions_path.exists():
        raise FileNotFoundError(f"Missing: {interactions_path}")

    paths = make_time_split(interactions_path, out_dir, cutoff_date)
    print(f"✅ Train: {paths.train_path}")
    print(f"✅ Test:  {paths.test_path}")


if __name__ == "__main__":
    main()
