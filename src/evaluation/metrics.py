from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import polars as pl


@dataclass(frozen=True)
class MetricResult:
    model: str
    recall_at_k: float
    map_at_k: float
    users: int


def recall_at_k(recs: list[str], truth: set[str], k: int) -> float:
    if not truth:
        return 0.0
    pred = recs[:k]
    hit = sum(1 for a in pred if a in truth)
    return hit / float(len(truth))


def ap_at_k(recs: list[str], truth: set[str], k: int) -> float:
    pred = recs[:k]
    if not truth:
        return 0.0
    hits = 0
    s = 0.0
    for i, a in enumerate(pred, start=1):
        if a in truth:
            hits += 1
            s += hits / i
    denom = min(len(truth), k)
    return s / denom if denom > 0 else 0.0


def evaluate_recs(
    model_name: str,
    recs_path: Path,
    test_path: Path,
    k: int = 12,
    max_users: int | None = 100_000,
) -> MetricResult:
    """
    Evaluate recommendation lists against test interactions.

    test ground truth: all article_id for each user in TEST (optionally limited to max_users).
    recs: per-user list column named 'recs'.
    """
    recs_df = pl.read_parquet(recs_path)
    test = pl.scan_parquet(test_path)

    # Only evaluate users in recs file
    users = recs_df.select("customer_id").unique()
    if max_users is not None:
        users = users.head(max_users)

    # Build truth per user from test
    truth_df = (
        test.join(users.lazy(), on="customer_id", how="inner")
        .group_by("customer_id")
        .agg(pl.col("article_id").unique().alias("truth"))
        .collect(engine="streaming")
    )

    eval_df = recs_df.join(truth_df, on="customer_id", how="inner")

    # Compute metrics row-wise (OK for <=100k users)
    recalls = []
    aps = []
    for row in eval_df.iter_rows(named=True):
        recs = row["recs"] or []
        truth = set(row["truth"] or [])
        recalls.append(recall_at_k(recs, truth, k))
        aps.append(ap_at_k(recs, truth, k))

    recall_mean = float(sum(recalls) / len(recalls)) if recalls else 0.0
    map_mean = float(sum(aps) / len(aps)) if aps else 0.0

    return MetricResult(
        model=model_name,
        recall_at_k=recall_mean,
        map_at_k=map_mean,
        users=len(recalls),
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    test_path = repo / "runs" / "day3" / "test_interactions.parquet"

    pop_path = repo / "runs" / "day3" / "recs_popularity.parquet"
    co_path = repo / "runs" / "day3" / "recs_cooccurrence.parquet"

    res_pop = evaluate_recs("popularity", pop_path, test_path, k=12, max_users=100_000)
    res_co = evaluate_recs("cooccurrence", co_path, test_path, k=12, max_users=100_000)

    print("Model        Users   Recall@12    MAP@12")
    print(f"{res_pop.model:<12} {res_pop.users:<7} {res_pop.recall_at_k:0.5f}    {res_pop.map_at_k:0.5f}")
    print(f"{res_co.model:<12} {res_co.users:<7} {res_co.recall_at_k:0.5f}    {res_co.map_at_k:0.5f}")


if __name__ == "__main__":
    main()
