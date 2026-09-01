import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


RESULTS_DIR = Path("results")

_COLORS = ["#4C72B0", "#DD8452", "#9BBB59", "#C44E52", "#8172B2", "#CC6677"]


def _avg_range(ranges: list | None) -> list:
    if not ranges:
        return [0.0, 0.0]
    n = len(ranges)
    return [round(sum(r[0] for r in ranges) / n, 4),
            round(sum(r[1] for r in ranges) / n, 4)]


# ── 콘솔 테이블과 동일한 데이터를 CSV로 저장 ──────────────────────
def _save_summary_csv(methods: list, t_img_emb: float, path: Path) -> None:
    rows = []
    for m in methods:
        t_total = t_img_emb + m["t_build"] + m["t_search"]
        rows.append({
            "method":       m["name"],
            "recall@1":     round(m["recall@1"], 4),
            "ndcg@5":       round(m["ndcg@5"],   4),
            "ndcg@10":      round(m["ndcg@10"],  4),
            "img_emb_s":    round(t_img_emb,     3),
            "pruning_rate": round(m["pruning_rate"], 4) if m["pruning_rate"] is not None else "",
            "search_s":     round(m["t_search"], 3),
            "total_s":      round(t_total,       3),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


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
    # ── 메서드별 집계 지표 ──
    recall_naive: float,
    recall_plaid: float,
    recall_gc: float,
    recall_bc: float,
    ndcg5_naive: float,
    ndcg10_naive: float,
    ndcg5_plaid: float,
    ndcg10_plaid: float,
    ndcg5_gc: float,
    ndcg10_gc: float,
    ndcg5_bc: float,
    ndcg10_bc: float,
    ndcg5_naive_vals: list,
    ndcg10_naive_vals: list,
    ndcg5_plaid_vals: list,
    ndcg10_plaid_vals: list,
    ndcg5_gc_vals: list = None,
    ndcg10_gc_vals: list = None,
    ndcg5_bc_vals: list = None,
    ndcg10_bc_vals: list = None,
    t_gc: float = 0.0,
    pruning_rate_gc: float = 0.0,
    t_bc: float = 0.0,
    pruning_rate: float = 0.0,
    gc_ranked_results: list = None,
    bc_ranked_results: list = None,
    gc_ubound_ranges: list = None,
    gc_score_ranges: list = None,
    bc_ubound_ranges: list = None,
    bc_score_ranges: list = None,
    gc_all_ubounds: list = None,
    bc_all_ubounds: list = None,
    # ── Box Bound (L3) ──
    bb_ranked_results: list = None,
    recall_bb: float = 0.0,
    ndcg5_bb: float = 0.0,
    ndcg10_bb: float = 0.0,
    ndcg5_bb_vals: list = None,
    ndcg10_bb_vals: list = None,
    t_bb: float = 0.0,
    pruning_rate_bb: float = 0.0,
    bb_ubound_ranges: list = None,
    bb_score_ranges: list = None,
    bb_all_ubounds: list = None,
    # ── 전체 methods 리스트 (summary CSV용) ──
    methods: list = None,
):
    slug = dataset.split("/")[-1]
    out_dir = RESULTS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Per-query CSV ────────────────────────────────────────────────
    rows = []
    for i in range(n_queries):
        naive_pred = scores[i].argmax().item()
        plaid_pred = all_plaid_results[i][0][0]
        gc_pred    = gc_ranked_results[i][0]  if gc_ranked_results  and len(gc_ranked_results[i])  > 0 else -1
        bc_pred    = bc_ranked_results[i][0]  if bc_ranked_results  and len(bc_ranked_results[i])  > 0 else -1
        bb_pred    = bb_ranked_results[i][0]  if bb_ranked_results  and len(bb_ranked_results[i])  > 0 else -1
        gt_idx     = gt_indices[i]
        rows.append({
            "query_idx":       i,
            "query":           queries[i],
            "gt_idx":          gt_idx,
            # Naive
            "naive_pred":      naive_pred,
            "naive_score":     round(scores[i][naive_pred].item(), 4),
            "naive_correct":   naive_pred == gt_idx,
            "naive_ndcg@5":    round(ndcg5_naive_vals[i], 4),
            "naive_ndcg@10":   round(ndcg10_naive_vals[i], 4),
            # Fast PLAID
            "plaid_pred":      plaid_pred,
            "plaid_score":     round(all_plaid_results[i][0][1], 4),
            "plaid_correct":   plaid_pred == gt_idx,
            "plaid_ndcg@5":    round(ndcg5_plaid_vals[i], 4),
            "plaid_ndcg@10":   round(ndcg10_plaid_vals[i], 4),
            # Global Centroid (L1)
            "gc_pred":         gc_pred,
            "gc_correct":      gc_pred == gt_idx,
            "gc_ndcg@5":       round(ndcg5_gc_vals[i],  4) if ndcg5_gc_vals  else 0,
            "gc_ndcg@10":      round(ndcg10_gc_vals[i], 4) if ndcg10_gc_vals else 0,
            "gc_ubound_min":   round(gc_ubound_ranges[i][0], 4) if gc_ubound_ranges else 0,
            "gc_ubound_max":   round(gc_ubound_ranges[i][1], 4) if gc_ubound_ranges else 0,
            "gc_score_min":    round(gc_score_ranges[i][0],  4) if gc_score_ranges  else 0,
            "gc_score_max":    round(gc_score_ranges[i][1],  4) if gc_score_ranges  else 0,
            # Block Residual (L2)
            "bc_pred":         bc_pred,
            "bc_correct":      bc_pred == gt_idx,
            "bc_ndcg@5":       round(ndcg5_bc_vals[i],  4) if ndcg5_bc_vals  else 0,
            "bc_ndcg@10":      round(ndcg10_bc_vals[i], 4) if ndcg10_bc_vals else 0,
            "bc_ubound_min":   round(bc_ubound_ranges[i][0], 4) if bc_ubound_ranges else 0,
            "bc_ubound_max":   round(bc_ubound_ranges[i][1], 4) if bc_ubound_ranges else 0,
            "bc_score_min":    round(bc_score_ranges[i][0],  4) if bc_score_ranges  else 0,
            "bc_score_max":    round(bc_score_ranges[i][1],  4) if bc_score_ranges  else 0,
            # Box Bound (L3)
            "bb_pred":         bb_pred,
            "bb_correct":      bb_pred == gt_idx,
            "bb_ndcg@5":       round(ndcg5_bb_vals[i],  4) if ndcg5_bb_vals  else 0,
            "bb_ndcg@10":      round(ndcg10_bb_vals[i], 4) if ndcg10_bb_vals else 0,
            "bb_ubound_min":   round(bb_ubound_ranges[i][0], 4) if bb_ubound_ranges else 0,
            "bb_ubound_max":   round(bb_ubound_ranges[i][1], 4) if bb_ubound_ranges else 0,
            "bb_score_min":    round(bb_score_ranges[i][0],  4) if bb_score_ranges  else 0,
            "bb_score_max":    round(bb_score_ranges[i][1],  4) if bb_score_ranges  else 0,
        })

    csv_path = out_dir / f"{ts}_per_query.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [Saved] Per-query CSV : {csv_path}")

    # ── Summary CSV (콘솔 테이블과 동일) ────────────────────────────
    if methods is not None:
        summary_csv_path = out_dir / f"{ts}_summary_data.csv"
        _save_summary_csv(methods, t_img_emb, summary_csv_path)
        print(f"  [Saved] Summary CSV   : {summary_csv_path}")

    # ── JSON summary ─────────────────────────────────────────────────
    summary = {
        "timestamp": ts,
        "dataset": dataset,
        "model": model_name,
        "n_queries": n_queries,
        "img_emb_time_s": round(t_img_emb, 3),
        "naive": {
            "recall@1":      round(recall_naive, 4),
            "ndcg@5":        round(ndcg5_naive,  4),
            "ndcg@10":       round(ndcg10_naive, 4),
            "search_time_s": round(t_naive,      3),
            "total_time_s":  round(t_img_emb + t_naive, 3),
        },
        "fast_plaid": {
            "recall@1":           round(recall_plaid, 4),
            "ndcg@5":             round(ndcg5_plaid,  4),
            "ndcg@10":            round(ndcg10_plaid, 4),
            "index_build_time_s": round(t_build,      3),
            "search_time_s":      round(t_plaid,      3),
            "total_time_s":       round(t_img_emb + t_build + t_plaid, 3),
        },
        "global_centroid": {
            "recall@1":            round(recall_gc,      4),
            "ndcg@5":              round(ndcg5_gc,       4),
            "ndcg@10":             round(ndcg10_gc,      4),
            "search_time_s":       round(t_gc,           3),
            "pruning_rate":        round(pruning_rate_gc,4),
            "total_time_s":        round(t_img_emb + t_gc, 3),
            "avg_ubound_range":    _avg_range(gc_ubound_ranges),
            "avg_score_range":     _avg_range(gc_score_ranges),
        },
        "block_centroid": {
            "recall@1":            round(recall_bc,  4),
            "ndcg@5":              round(ndcg5_bc,   4),
            "ndcg@10":             round(ndcg10_bc,  4),
            "search_time_s":       round(t_bc,       3),
            "pruning_rate":        round(pruning_rate, 4),
            "total_time_s":        round(t_img_emb + t_bc, 3),
            "avg_ubound_range":    _avg_range(bc_ubound_ranges),
            "avg_score_range":     _avg_range(bc_score_ranges),
        },
        "box_bound": {
            "recall@1":            round(recall_bb,      4),
            "ndcg@5":              round(ndcg5_bb,       4),
            "ndcg@10":             round(ndcg10_bb,      4),
            "search_time_s":       round(t_bb,           3),
            "pruning_rate":        round(pruning_rate_bb,4),
            "total_time_s":        round(t_img_emb + t_bb, 3),
            "avg_ubound_range":    _avg_range(bb_ubound_ranges),
            "avg_score_range":     _avg_range(bb_score_ranges),
        },
    }
    json_path = out_dir / f"{ts}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  [Saved] Summary JSON  : {json_path}")

    # ── Comparison chart ─────────────────────────────────────────────
    png_path = out_dir / f"{ts}_comparison.png"
    if methods is not None:
        _save_per_dataset_chart(slug=slug, methods=methods,
                                t_img_emb=t_img_emb, png_path=png_path)
    else:
        _save_per_dataset_chart_legacy(
            slug=slug,
            recall_naive=recall_naive, recall_plaid=recall_plaid,
            recall_gc=recall_gc, recall_bc=recall_bc,
            ndcg5_naive=ndcg5_naive, ndcg10_naive=ndcg10_naive,
            ndcg5_plaid=ndcg5_plaid, ndcg10_plaid=ndcg10_plaid,
            ndcg5_gc=ndcg5_gc, ndcg10_gc=ndcg10_gc,
            ndcg5_bc=ndcg5_bc, ndcg10_bc=ndcg10_bc,
            t_img_emb=t_img_emb, t_naive=t_naive,
            t_build=t_build, t_plaid=t_plaid,
            t_gc=t_gc, t_bc=t_bc, png_path=png_path,
        )
    print(f"  [Saved] Chart PNG     : {png_path}")

    # ── Bound distribution histogram ─────────────────────────────────
    if gc_all_ubounds is not None and bc_all_ubounds is not None:
        dist_path = out_dir / f"{ts}_bound_distribution.png"
        _save_bound_distribution_chart(
            slug=slug,
            scores=scores,
            gc_all_ubounds=gc_all_ubounds,
            bc_all_ubounds=bc_all_ubounds,
            bb_all_ubounds=bb_all_ubounds,
            n_queries=n_queries,
            png_path=dist_path,
        )
        print(f"  [Saved] Bound Dist    : {dist_path}")

    return summary


def save_multi_dataset_summary(all_results: list):
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    recall_path = RESULTS_DIR / f"{ts}_multi_recall.png"
    _save_grouped_bar(
        labels=slugs,
        values_a=[r["recall_naive"] for r in all_results],
        values_b=[r["recall_plaid"] for r in all_results],
        label_a="Naive", label_b="Fast PLAID",
        ylabel="Recall@1", title="Recall@1 by Dataset",
        fmt="{:.2%}", png_path=recall_path,
    )
    print(f"[Saved] Multi-dataset Recall chart: {recall_path}")

    ndcg5_path = RESULTS_DIR / f"{ts}_multi_ndcg5.png"
    _save_grouped_bar(
        labels=slugs,
        values_a=[r["ndcg5_naive"] for r in all_results],
        values_b=[r["ndcg5_plaid"] for r in all_results],
        label_a="Naive", label_b="Fast PLAID",
        ylabel="nDCG@5", title="nDCG@5 by Dataset",
        fmt="{:.4f}", png_path=ndcg5_path,
    )
    print(f"[Saved] Multi-dataset nDCG@5 chart: {ndcg5_path}")

    ndcg10_path = RESULTS_DIR / f"{ts}_multi_ndcg10.png"
    _save_grouped_bar(
        labels=slugs,
        values_a=[r["ndcg10_naive"] for r in all_results],
        values_b=[r["ndcg10_plaid"] for r in all_results],
        label_a="Naive", label_b="Fast PLAID",
        ylabel="nDCG@10", title="nDCG@10 by Dataset",
        fmt="{:.4f}", png_path=ndcg10_path,
    )
    print(f"[Saved] Multi-dataset nDCG@10 chart: {ndcg10_path}")

    time_path = RESULTS_DIR / f"{ts}_multi_time.png"
    _save_multi_time_chart(all_results=all_results, png_path=time_path)
    print(f"[Saved] Multi-dataset Latency chart: {time_path}")


# ── internal helpers ──────────────────────────────────────────────────

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


def _save_per_dataset_chart(*, slug: str, methods: list, t_img_emb: float,
                             png_path: Path) -> None:
    """methods 리스트 기반으로 모든 메서드를 동적으로 차트에 표시."""
    n_m    = len(methods)
    colors = _COLORS[:n_m]
    labels = [m["name"] for m in methods]
    x      = np.arange(n_m)

    fig, axes = plt.subplots(1, 3, figsize=(6 + n_m * 1.2, 4))
    fig.suptitle(f"ColPali Benchmark: {slug}", fontsize=13, fontweight="bold")

    # Recall@1
    ax   = axes[0]
    vals = [m["recall@1"] for m in methods]
    bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor="white")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Recall@1")
    ax.set_title("Recall@1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.2%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # nDCG@5 / nDCG@10
    for ax, key, title in [
        (axes[1], "ndcg@5",  "nDCG@5"),
        (axes[2], "ndcg@10", "nDCG@10"),
    ]:
        vals = [m[key] for m in methods]
        bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor="white")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_per_dataset_chart_legacy(*, slug, recall_naive, recall_plaid,
                                    recall_gc, recall_bc,
                                    ndcg5_naive, ndcg10_naive,
                                    ndcg5_plaid, ndcg10_plaid,
                                    ndcg5_gc, ndcg10_gc,
                                    ndcg5_bc, ndcg10_bc,
                                    t_img_emb, t_naive, t_build, t_plaid,
                                    t_gc, t_bc, png_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f"ColPali Benchmark: {slug}", fontsize=13, fontweight="bold")

    method_labels = ["Naive", "Fast PLAID", "Global Centroid", "Block Centroid"]
    colors        = _COLORS[:4]

    ax   = axes[0]
    vals = [recall_naive, recall_plaid, recall_gc, recall_bc]
    bars = ax.bar(method_labels, vals, color=colors, width=0.5, edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall@1")
    ax.set_title("Recall@1")
    ax.tick_params(axis="x", labelsize=8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2%}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    x = np.arange(4)
    for ax, (m_naive, m_plaid, m_gc, m_bc), title in [
        (axes[1], (ndcg5_naive,  ndcg5_plaid,  ndcg5_gc,  ndcg5_bc),  "nDCG@5"),
        (axes[2], (ndcg10_naive, ndcg10_plaid, ndcg10_gc, ndcg10_bc), "nDCG@10"),
    ]:
        vals4    = [m_naive, m_plaid, m_gc, m_bc]
        bars_n   = ax.bar(x, vals4, 0.35, color=colors, edgecolor="white")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=8)
        for bar, val in zip(bars_n, vals4):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_bound_distribution_chart(*, slug, scores, gc_all_ubounds, bc_all_ubounds,
                                    bb_all_ubounds, n_queries, png_path):
    exact_flat = scores.cpu().numpy().flatten()
    l1_flat    = np.concatenate([np.array(q) for q in gc_all_ubounds if len(q) > 0])
    l2_flat    = np.concatenate([np.array(q) for q in bc_all_ubounds if len(q) > 0])

    has_bb     = bb_all_ubounds is not None and any(len(q) > 0 for q in bb_all_ubounds)
    n_plots    = 3 if has_bb else 2

    min_len    = min(len(l1_flat), len(l2_flat))
    gain_l1_l2 = l1_flat[:min_len] - l2_flat[:min_len]

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    fig.suptitle(
        f"{slug} — Bound Distribution ({n_queries} queries)",
        fontsize=13, fontweight="bold",
    )

    # 그래프 1: exact vs L1 vs L2 vs Box Bound 분포
    ax   = axes[0]
    bins = 60
    ax.hist(exact_flat, bins=bins, alpha=0.5, density=True,
            color="#4F81BD", label=f"exact (μ={exact_flat.mean():.3f})")
    ax.hist(l1_flat, bins=bins, alpha=0.5, density=True,
            color="#E46C0A", label=f"L1 / GC (μ={l1_flat.mean():.3f})")
    ax.hist(l2_flat, bins=bins, alpha=0.5, density=True,
            color="#9BBB59", label=f"L2 / BC (μ={l2_flat.mean():.3f})")
    if has_bb:
        bb_flat = np.concatenate([np.array(q) for q in bb_all_ubounds if len(q) > 0])
        ax.hist(bb_flat, bins=bins, alpha=0.5, density=True,
                color="#CC6677", label=f"Box Bound (μ={bb_flat.mean():.3f})")
    ax.set_xlabel("score / bound value")
    ax.set_ylabel("density")
    ax.set_title("exact vs L1 vs L2 vs Box Bound")
    ax.legend(fontsize=9)

    # 그래프 2: gain 분포 (L1 - L2)
    ax = axes[1]
    ax.hist(gain_l1_l2, bins=bins, color="#C0504D", alpha=0.8, density=True)
    ax.axvline(gain_l1_l2.mean(), color="red", linewidth=1.5,
               label=f"mean={gain_l1_l2.mean():.4f}")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("gain = L1 - L2")
    ax.set_ylabel("density")
    ax.set_title("gain distribution (L1 - L2)")
    ax.legend(fontsize=9)

    # 그래프 3: gain 분포 (L2 - Box Bound)
    if has_bb:
        min_len2    = min(len(l2_flat), len(bb_flat))
        gain_l2_bb  = l2_flat[:min_len2] - bb_flat[:min_len2]
        ax          = axes[2]
        ax.hist(gain_l2_bb, bins=bins, color="#7B2D8B", alpha=0.8, density=True)
        ax.axvline(gain_l2_bb.mean(), color="purple", linewidth=1.5,
                   label=f"mean={gain_l2_bb.mean():.4f}")
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("gain = L2 - Box Bound")
        ax.set_ylabel("density")
        ax.set_title("gain distribution (L2 - Box Bound)")
        ax.legend(fontsize=9)

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
    ax.bar(x + w / 2, t_build, w, label="PLAID Index Build", color="#C44E52", bottom=t_emb)
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
