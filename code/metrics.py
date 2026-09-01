import math


def compute_ndcg(ranked_indices: list, gt_idx: int, k: int) -> float:
    for rank, idx in enumerate(ranked_indices[:k]):
        if idx == gt_idx:
            return 1.0 / math.log2(rank + 2)
    return 0.0


def compute_method_metrics(ranked_results: list, gt_indices: list) -> dict:
    """단일 검색 메서드의 ranked_results로 recall@1, nDCG@5, nDCG@10을 계산."""
    n = len(ranked_results)
    ndcg5_vals, ndcg10_vals = [], []
    correct = 0

    for ranked, gt in zip(ranked_results, gt_indices):
        if ranked and ranked[0] == gt:
            correct += 1
        ndcg5_vals.append(compute_ndcg(ranked, gt, 5))
        ndcg10_vals.append(compute_ndcg(ranked, gt, 10))

    return {
        "recall@1":    correct / n,
        "ndcg@5":      sum(ndcg5_vals)  / n,
        "ndcg@10":     sum(ndcg10_vals) / n,
        "ndcg5_vals":  ndcg5_vals,
        "ndcg10_vals": ndcg10_vals,
    }
