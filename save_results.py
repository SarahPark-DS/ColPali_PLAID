import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


RESULTS_DIR = Path("results")


def save_benchmark_results(
    *,
    dataset: str,
    model_name: str,
    queries,
    filename_to_corpus_idx: dict,
    gt_indices: list,
    ds,
    scores: torch.Tensor,
    t_img_emb: float,
    t_naive: float,
    all_plaid_results: list,
    t_build: float,
    t_plaid: float,
    n_queries: int,
    recall_naive: float,
    recall_plaid: float,
    ndcg5_naive: float,
    ndcg10_naive: float,
    ndcg5_plaid: float,
    ndcg10_plaid: float,
    ndcg5_naive_vals: list,
    ndcg10_naive_vals: list,
    ndcg5_plaid_vals: list,
    ndcg10_plaid_vals: list,
):
    slug = dataset.split("/")[-1]
    out_dir = RESULTS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-query rows
    rows = []
    for i in range(n_queries):
        naive_pred = scores[i].argmax().item()
        plaid_pred = all_plaid_results[i][0][0]
        gt_idx = gt_indices[i]
        rows.append({
            "query_idx":       i,
            "query":           queries[i],
            "gt_idx":          gt_idx,
            "naive_pred":      naive_pred,
            "naive_score":     round(scores[i][naive_pred].item(), 4),
            "naive_correct":   naive_pred == gt_idx,
            "naive_ndcg@5":    round(ndcg5_naive_vals[i], 4),
            "naive_ndcg@10":   round(ndcg10_naive_vals[i], 4),
            "plaid_pred":      plaid_pred,
            "plaid_score":     round(all_plaid_results[i][0][1], 4),
            "plaid_correct":   plaid_pred == gt_idx,
            "plaid_ndcg@5":    round(ndcg5_plaid_vals[i], 4),
            "plaid_ndcg@10":   round(ndcg10_plaid_vals[i], 4),
        })

    # JSON summary
    summary = {
        "timestamp": ts,
        "dataset": dataset,
        "model": model_name,
        "n_queries": n_queries,
        "img_emb_time_s": round(t_img_emb, 3),
        "naive": {
            "recall@1":     round(recall_naive, 4),
            "ndcg@5":       round(ndcg5_naive, 4),
            "ndcg@10":      round(ndcg10_naive, 4),
            "search_time_s": round(t_naive, 3),
            "total_time_s":  round(t_img_emb + t_naive, 3),
        },
        "fast_plaid": {
            "recall@1":          round(recall_plaid, 4),
            "ndcg@5":            round(ndcg5_plaid, 4),
            "ndcg@10":           round(ndcg10_plaid, 4),
            "index_build_time_s": round(t_build, 3),
            "search_time_s":      round(t_plaid, 3),
            "total_time_s":       round(t_img_emb + t_build + t_plaid, 3),
        },
    }
    json_path = out_dir / f"{ts}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  [Saved] Summary JSON : {json_path}")

    # CSV
    csv_path = out_dir / f"{ts}_per_query.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [Saved] Per-query CSV: {csv_path}")

    # Chart
    png_path = out_dir / f"{ts}_comparison.png"
    _save_per_dataset_chart(
        slug=slug,
        recall_naive=recall_naive,
        recall_plaid=recall_plaid,
        ndcg5_naive=ndcg5_naive,
        ndcg10_naive=ndcg10_naive,
        ndcg5_plaid=ndcg5_plaid,
        ndcg10_plaid=ndcg10_plaid,
        t_img_emb=t_img_emb,
        t_naive=t_naive,
        t_build=t_build,
        t_plaid=t_plaid,
        png_path=png_path,
    )
    print(f"  [Saved] Chart PNG    : {png_path}")

    return summary


def save_multi_dataset_summary(all_results: list):
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    summary = {
        "timestamp": ts,
        "datasets": [
            {
                "dataset":              r["dataset"],
                "n_queries":            r["n_queries"],
                "img_emb_time_s":       round(r["t_img_emb"], 3),
                "naive_recall@1":       round(r["recall_naive"], 4),
                "naive_ndcg@5":         round(r["ndcg5_naive"], 4),
                "naive_ndcg@10":        round(r["ndcg10_naive"], 4),
                "naive_search_time_s":  round(r["t_naive"], 3),
                "naive_total_time_s":   round(r["t_img_emb"] + r["t_naive"], 3),
                "plaid_recall@1":       round(r["recall_plaid"], 4),
                "plaid_ndcg@5":         round(r["ndcg5_plaid"], 4),
                "plaid_ndcg@10":        round(r["ndcg10_plaid"], 4),
                "plaid_search_time_s":  round(r["t_plaid"], 3),
                "plaid_build_time_s":   round(r["t_build"], 3),
                "plaid_total_time_s":   round(r["t_img_emb"] + r["t_build"] + r["t_plaid"], 3),
            }
            for r in all_results
        ],
    }
    json_path = RESULTS_DIR / f"{ts}_multi_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[Saved] Multi-dataset JSON : {json_path}")

    slugs = [r["dataset"].split("/")[-1] for r in all_results]

    # Recall@1 grouped bar
    recall_path = RESULTS_DIR / f"{ts}_multi_recall.png"
    _save_grouped_bar(
        labels=slugs,
        values_a=[r["recall_naive"] for r in all_results],
        values_b=[r["recall_plaid"] for r in all_results],
        label_a="Naive",
        label_b="Fast PLAID",
        ylabel="Recall@1",
        title="Recall@1 by Dataset",
        fmt="{:.2%}",
        png_path=recall_path,
    )
    print(f"[Saved] Multi-dataset Recall chart: {recall_path}")

    # nDCG@5 grouped bar
    ndcg5_path = RESULTS_DIR / f"{ts}_multi_ndcg5.png"
    _save_grouped_bar(
        labels=slugs,
        values_a=[r["ndcg5_naive"] for r in all_results],
        values_b=[r["ndcg5_plaid"] for r in all_results],
        label_a="Naive",
        label_b="Fast PLAID",
        ylabel="nDCG@5",
        title="nDCG@5 by Dataset",
        fmt="{:.4f}",
        png_path=ndcg5_path,
    )
    print(f"[Saved] Multi-dataset nDCG@5 chart: {ndcg5_path}")

    # nDCG@10 grouped bar
    ndcg10_path = RESULTS_DIR / f"{ts}_multi_ndcg10.png"
    _save_grouped_bar(
        labels=slugs,
        values_a=[r["ndcg10_naive"] for r in all_results],
        values_b=[r["ndcg10_plaid"] for r in all_results],
        label_a="Naive",
        label_b="Fast PLAID",
        ylabel="nDCG@10",
        title="nDCG@10 by Dataset",
        fmt="{:.4f}",
        png_path=ndcg10_path,
    )
    print(f"[Saved] Multi-dataset nDCG@10 chart: {ndcg10_path}")

    # Latency stacked chart
    time_path = RESULTS_DIR / f"{ts}_multi_time.png"
    _save_multi_time_chart(all_results=all_results, png_path=time_path)
    print(f"[Saved] Multi-dataset Latency chart: {time_path}")


# ── internal helpers ──────────────────────────────────

def _save_grouped_bar(*, labels, values_a, values_b, label_a, label_b,
                      ylabel, title, fmt, png_path):
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 5))
    bars_a = ax.bar(x - w / 2, values_a, w, label=label_a, color="#4C72B0")
    bars_b = ax.bar(x + w / 2, values_b, w, label=label_b, color="#DD8452")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    all_vals = values_a + values_b
    top = max(all_vals) if all_vals else 1.0
    for bar, val in zip(list(bars_a) + list(bars_b), all_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + top * 0.01,
                fmt.format(val), ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_per_dataset_chart(*, slug, recall_naive, recall_plaid,
                            ndcg5_naive, ndcg10_naive, ndcg5_plaid, ndcg10_plaid,
                            t_img_emb, t_naive, t_build, t_plaid, png_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"ColPali Benchmark: {slug}", fontsize=13, fontweight="bold")

    # Recall@1
    ax = axes[0]
    labels = ["Naive", "Fast PLAID"]
    vals = [recall_naive, recall_plaid]
    bars = ax.bar(labels, vals, color=["#4C72B0", "#DD8452"], width=0.4, edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall@1")
    ax.set_title("Recall@1")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # nDCG@5 and nDCG@10
    x = np.arange(2)
    w = 0.35
    for ax, metric_naive, metric_plaid, title in [
        (axes[1], ndcg5_naive,  ndcg5_plaid,  "nDCG@5"),
        (axes[2], ndcg10_naive, ndcg10_plaid, "nDCG@10"),
    ]:
        bars_n = ax.bar(x - w / 2, [metric_naive, metric_plaid], w,
                        color=["#4C72B0", "#DD8452"], edgecolor="white")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["Naive", "Fast PLAID"])
        for bar, val in zip(bars_n, [metric_naive, metric_plaid]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_multi_time_chart(*, all_results, png_path):
    slugs   = [r["dataset"].split("/")[-1].replace("syntheticDocQA_", "syn_") for r in all_results]
    t_emb   = [r["t_img_emb"] for r in all_results]
    t_naive = [r["t_naive"]   for r in all_results]
    t_build = [r["t_build"]   for r in all_results]
    t_plaid = [r["t_plaid"]   for r in all_results]

    x = np.arange(len(slugs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(slugs) * 1.3), 5))
    ax.bar(x - w / 2, t_emb,   w, label="Img Embedding",    color="#4C72B0")
    ax.bar(x - w / 2, t_naive, w, label="Naive Search",      color="#55A868", bottom=t_emb)
    ax.bar(x + w / 2, t_emb,   w, label="Img Embedding (P)", color="#4C72B0", alpha=0.5)
    ax.bar(x + w / 2, t_build, w, label="PLAID Index Build", color="#C44E52",
           bottom=t_emb)
    ax.bar(x + w / 2, t_plaid, w, label="PLAID Search",      color="#DD8452",
           bottom=[e + b for e, b in zip(t_emb, t_build)])

    ax.set_ylabel("Time (s)")
    ax.set_title("Latency Breakdown by Dataset  (left=Naive, right=PLAID)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(slugs, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
