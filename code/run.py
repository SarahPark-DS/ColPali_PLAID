from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from benchmark import run_benchmark
from config import DATASETS

_COLORS = ["#4F81BD", "#C0504D", "#9BBB59", "#FF8C00", "#7B2D8B"]


def _print_final_summary(all_results: list):
    method_names = [m["name"] for m in all_results[0]["methods"]]
    prune_names  = [
        name for name in method_names
        if all_results[0]["methods"][method_names.index(name)]["pruning_rate"] is not None
    ]

    W = 35 + len(method_names) * 24 + len(prune_names) * 12
    print(f"\n\n{'='*W}")
    print(f"{'FINAL SUMMARY':^{W}}")
    print(f"{'='*W}")

    header = f"{'Dataset':<35}"
    for name in method_names:
        header += f"  {(name[:7]+'-R@1'):>11} {'nDCG@10':>10}"
    for name in prune_names:
        header += f"  {'Prune':>10}"
    print(header)
    print(f"{'-'*W}")

    for r in all_results:
        slug = r["dataset"].split("/")[-1]
        row  = f"{slug:<35}"
        md   = {m["name"]: m for m in r["methods"]}
        for name in method_names:
            row += f"  {md[name]['recall@1']:>11.2%} {md[name]['ndcg@10']:>10.4f}"
        for name in prune_names:
            row += f"  {md[name]['pruning_rate']:>10.2%}"
        print(row)

    print(f"{'='*W}")

    avg_row = f"{'Average':<35}"
    for i, name in enumerate(method_names):
        avg_r = sum(r["methods"][i]["recall@1"] for r in all_results) / len(all_results)
        avg_n = sum(r["methods"][i]["ndcg@10"]  for r in all_results) / len(all_results)
        avg_row += f"  {avg_r:>11.2%} {avg_n:>10.4f}"
    for name in prune_names:
        idx = method_names.index(name)
        avg_p = sum(r["methods"][idx]["pruning_rate"] for r in all_results) / len(all_results)
        avg_row += f"  {avg_p:>10.2%}"
    print(avg_row)
    print(f"{'='*W}")


def _save_multi_dataset_summary(all_results: list, out_path: str = "results/final_benchmark_summary.png"):
    method_names = [m["name"] for m in all_results[0]["methods"]]
    prune_methods = [
        (i, name) for i, name in enumerate(method_names)
        if all_results[0]["methods"][i]["pruning_rate"] is not None
    ]
    dataset_labels = [
        r["dataset"].split("/")[-1].replace("_test_subsampled", "").replace("_test", "")
        for r in all_results
    ]

    x        = np.arange(len(all_results))
    n_m      = len(method_names)
    width    = 0.7 / n_m
    n_plots  = 2 + (1 if prune_methods else 0)

    _, axes = plt.subplots(n_plots, 1, figsize=(14, 6 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for ax_idx, (key, ylabel, title, scale) in enumerate([
        ("recall@1", "Recall@1 (%)", "Recall@1 Accuracy Comparison (Higher is Better)",    100),
        ("ndcg@10",  "nDCG@10",      "nDCG@10 Ranking Quality Comparison (Higher is Better)", 1),
    ]):
        ax = axes[ax_idx]
        for j, name in enumerate(method_names):
            vals   = [r["methods"][j][key] * scale for r in all_results]
            offset = (j - n_m / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=name, color=_COLORS[j % len(_COLORS)], alpha=0.9)
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title,   fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, rotation=15, ha="right", fontsize=10)
        ax.legend(loc="lower left", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    if prune_methods:
        ax = axes[2]
        p_width = width * 1.6
        for j, (method_idx, name) in enumerate(prune_methods):
            vals   = [r["methods"][method_idx]["pruning_rate"] * 100 for r in all_results]
            offset = (j - len(prune_methods) / 2 + 0.5) * p_width
            bars   = ax.bar(x + offset, vals, p_width,
                            color=_COLORS[method_idx % len(_COLORS)], alpha=0.85, label=name)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("Pruning Rate (%)", fontsize=12, fontweight="bold")
        ax.set_title("Computation Pruning Rate (ρ)",  fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, rotation=15, ha="right", fontsize=10)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[Saved] 종합 비교 차트: {out_path}")
    plt.show()


if __name__ == "__main__":
    all_results = []
    for ds_name in tqdm(DATASETS):
        result = run_benchmark(ds_name)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("실행 완료된 데이터셋 결과가 없어 시각화 및 요약본을 출력하지 못했습니다.")
    else:
        _print_final_summary(all_results)
        _save_multi_dataset_summary(all_results)
