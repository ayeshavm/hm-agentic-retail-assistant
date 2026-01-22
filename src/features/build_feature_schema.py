from __future__ import annotations

from pathlib import Path
import json
import polars as pl


def pl_dtype_to_str(dtype: pl.DataType) -> str:
    return str(dtype)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "data" / "processed"
    schema_path = out_dir / "feature_schema.json"

    interactions_path = out_dir / "interactions.parquet"
    item_features_path = out_dir / "item_features.parquet"

    if not interactions_path.exists():
        raise FileNotFoundError(f"Missing: {interactions_path}")
    if not item_features_path.exists():
        raise FileNotFoundError(f"Missing: {item_features_path}")

    interactions = pl.read_parquet(interactions_path)
    item_features = pl.read_parquet(item_features_path)

    # Basic stats (keep it lightweight)
    interactions_stats = {
        "rows": interactions.height,
        "cols": interactions.width,
        "implicit_score_min": float(interactions["implicit_score"].min()),
        "implicit_score_max": float(interactions["implicit_score"].max()),
        "freq_min": int(interactions["freq"].min()),
        "freq_max": int(interactions["freq"].max()),
    }

    item_stats = {
        "rows": item_features.height,
        "cols": item_features.width,
    }

    schema = {
        "interactions": {
            "path": str(interactions_path),
            "dtypes": {name: pl_dtype_to_str(dtype) for name, dtype in zip(interactions.columns, interactions.dtypes)},
            "stats": interactions_stats,
        },
        "item_features": {
            "path": str(item_features_path),
            "dtypes": {name: pl_dtype_to_str(dtype) for name, dtype in zip(item_features.columns, item_features.dtypes)},
            "stats": item_stats,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"✅ Wrote: {schema_path}")


if __name__ == "__main__":
    main()
