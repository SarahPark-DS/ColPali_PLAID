"""
모든 데이터셋의 r_j vs ρ_j^(b) 통계 비교 스크립트

출력:
  - 콘솔: 데이터셋별 통계 테이블
  - cluster_index_output_4x/radius_stats_all.csv  : 전체 요약 CSV
  - cluster_index_output_4x/radius_stats_all.png  : 비교 차트
"""

import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("./cluster_index_output_4x")

DATASETS = [
    "arxivqa",
    "docvqa",
    "infovqa",
    "tabfquad",
    "tatdqa",
    "shiftproject",
    "syntheticDocQA_artificial_intelligence",
    "syntheticDocQA_energy",
    "syntheticDocQA_government_reports",
    "syntheticDocQA_healthcare_industry",
]


def _stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(arr.mean()),
        "max":  float(arr.max()),
        "p50":  float(np.percentile(arr, 50)),
        "p95":  float(np.percentile(arr, 95)),
    }


def load_and_analyze(slug: str) -> dict | None:
    save_dir = OUTPUT_DIR / slug
    npz_path = save_dir / "cluster_index_block.npz"
    meta_path = save_dir / "cluster_index_block_meta.pkl"

    if not npz_path.exists():
        print(f"  [SKIP] {slug}: index 파일 없음")
        return None

    npz  = np.load(npz_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    radii       = npz["radii"]        # (K,)
    block_radii = npz["block_radii"]  # (K, B)
    B           = meta["metadata"]["B"]
    D           = meta["metadata"].get("D", block_radii.shape[1] * (128 // B))
    block_sz    = D // B

    cluster_to_vectors = meta["cluster_to_vectors"]
    nonempty = np.array([len(c) > 0 for c in cluster_to_vectors])
    K_valid  = int(nonempty.sum())

    r       = radii[nonempty]                                    # (K',)
    blk_l2  = np.sqrt((block_radii[nonempty] ** 2).sum(axis=1)) # (K',) √(Σ ρ^(b)²)
    gain    = r - blk_l2

    violations = int(np.sum(blk_l2 > r + 1e-5))

    # 블록별 통계
    per_block = []
    for b in range(B):
        pb = block_radii[nonempty, b]
        per_block.append({
            "block": b,
            "dim_range": f"{b*block_sz}~{(b+1)*block_sz-1}",
            **_stats(pb),
        })

    return {
        "slug":       slug,
        "K_valid":    K_valid,
        "B":          B,
        "block_sz":   block_sz,
        "r":          r,
        "blk_l2":     blk_l2,
        "gain":       gain,
        "r_stats":    _stats(r),
        "blk_stats":  _stats(blk_l2),
        "gain_stats": _stats(gain),
        "improve_pct": float(gain.mean() / r.mean() * 100),
        "violations": violations,
        "per_block":  per_block,
        "block_radii_nonempty": block_radii[nonempty],
    }


def print_table(results: list):
    W = 110
    print(f"\n{'='*W}")
    print(f"{'r_j  vs  sqrt(Σ ρ_j^(b)²)  —  All Datasets':^{W}}")
    print(f"{'='*W}")

    hdr = (f"{'Dataset':<42} {'K':>5} {'B':>2}  "
           f"{'r_j mean':>9} {'r_j p50':>8} {'r_j p95':>8}  "
           f"{'blk mean':>9} {'blk p50':>8} {'blk p95':>8}  "
           f"{'gain mean':>10} {'improve%':>9}  {'viol':>5}")
    print(hdr)
    print(f"{'-'*W}")

    for r in results:
        rs = r["r_stats"]
        bs = r["blk_stats"]
        gs = r["gain_stats"]
        print(
            f"{r['slug']:<42} {r['K_valid']:>5} {r['B']:>2}  "
            f"{rs['mean']:>9.4f} {rs['p50']:>8.4f} {rs['p95']:>8.4f}  "
            f"{bs['mean']:>9.4f} {bs['p50']:>8.4f} {bs['p95']:>8.4f}  "
            f"{gs['mean']:>10.4f} {r['improve_pct']:>8.2f}%  {r['violations']:>5}"
        )

    print(f"{'='*W}")

    # 블록별 세부 통계
    for r in results:
        B = r["B"]
        print(f"\n  [{r['slug']}]  블록별 ρ_j^(b) (B={B}, block_sz={r['block_sz']})")
        print(f"    {'block':>5}  {'dim range':>12}  {'mean':>8}  {'max':>8}  {'p50':>8}  {'p95':>8}")
        print(f"    {'-'*55}")
        for pb in r["per_block"]:
            print(f"    {pb['block']:>5}  {pb['dim_range']:>12}  "
                  f"{pb['mean']:>8.4f}  {pb['max']:>8.4f}  "
                  f"{pb['p50']:>8.4f}  {pb['p95']:>8.4f}")


def save_csv(results: list, out_path: Path):
    rows = []
    for r in results:
        rs, bs, gs = r["r_stats"], r["blk_stats"], r["gain_stats"]
        base = {
            "dataset":      r["slug"],
            "K_valid":      r["K_valid"],
            "B":            r["B"],
            "r_mean":       round(rs["mean"], 4),
            "r_p50":        round(rs["p50"],  4),
            "r_p95":        round(rs["p95"],  4),
            "r_max":        round(rs["max"],  4),
            "blk_mean":     round(bs["mean"], 4),
            "blk_p50":      round(bs["p50"],  4),
            "blk_p95":      round(bs["p95"],  4),
            "blk_max":      round(bs["max"],  4),
            "gain_mean":    round(gs["mean"], 4),
            "gain_p50":     round(gs["p50"],  4),
            "improve_pct":  round(r["improve_pct"], 2),
            "violations":   r["violations"],
        }
        for pb in r["per_block"]:
            b = pb["block"]
            base[f"b{b}_mean"] = round(pb["mean"], 4)
            base[f"b{b}_p50"]  = round(pb["p50"],  4)
            base[f"b{b}_p95"]  = round(pb["p95"],  4)
        rows.append(base)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Saved] CSV: {out_path}")


def save_chart(results: list, out_path: Path):
    slugs        = [r["slug"].replace("syntheticDocQA_", "syn_") for r in results]
    r_means      = [r["r_stats"]["mean"]   for r in results]
    blk_means    = [r["blk_stats"]["mean"] for r in results]
    gain_means   = [r["gain_stats"]["mean"] for r in results]
    improve_pcts = [r["improve_pct"]        for r in results]

    x = np.arange(len(slugs))
    w = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle("r_j  vs  √(Σ ρ_j^(b)²)  —  All Datasets", fontsize=14, fontweight="bold")

    # ── (0,0) mean 비교 막대 ──
    ax = axes[0, 0]
    b1 = ax.bar(x - w/2, r_means,   w, label="r_j (level 1)",        color="#4F81BD", alpha=0.85)
    b2 = ax.bar(x + w/2, blk_means, w, label="√(Σρ²) (level 2 idx)", color="#E46C0A", alpha=0.85)
    ax.set_ylabel("mean radius")
    ax.set_title("Mean Radius: r_j vs √(Σ ρ^(b)²)")
    ax.set_xticks(x); ax.set_xticklabels(slugs, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # ── (0,1) gain mean 막대 ──
    ax = axes[0, 1]
    colors = ["#C00000" if g < 0 else "#9BBB59" for g in gain_means]
    ax.bar(x, gain_means, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("gain mean  (r_j − √(Σρ²))")
    ax.set_title("Gain (r_j − √(Σ ρ^(b)²))  —  positive = L2 tighter")
    ax.set_xticks(x); ax.set_xticklabels(slugs, rotation=25, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── (1,0) improve % 막대 ──
    ax = axes[1, 0]
    colors2 = ["#C00000" if v < 0 else "#4BACC6" for v in improve_pcts]
    bars = ax.bar(x, improve_pcts, color=colors2, alpha=0.85)
    for bar, val in zip(bars, improve_pcts):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + (0.3 if val >= 0 else -1.2),
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("improvement %  (gain / r_j)")
    ax.set_title("Tightening Rate  gain / r_j  (%)")
    ax.set_xticks(x); ax.set_xticklabels(slugs, rotation=25, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── (1,1) 데이터셋별 블록별 mean 히트맵 ──
    ax = axes[1, 1]
    B_max = max(r["B"] for r in results)
    mat = np.full((len(results), B_max), np.nan)
    for i, r in enumerate(results):
        for pb in r["per_block"]:
            mat[i, pb["block"]] = pb["mean"]
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(B_max)); ax.set_xticklabels([f"b{b}" for b in range(B_max)])
    ax.set_yticks(range(len(slugs))); ax.set_yticklabels(slugs, fontsize=8)
    ax.set_title("Per-block ρ_j^(b) mean  (heatmap)")
    plt.colorbar(im, ax=ax, label="mean ρ_j^(b)")
    for i in range(len(results)):
        for j in range(B_max):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Chart: {out_path}")


if __name__ == "__main__":
    results = []
    for slug in DATASETS:
        print(f"Loading {slug}...")
        res = load_and_analyze(slug)
        if res is not None:
            results.append(res)

    if not results:
        print("분석 가능한 데이터셋이 없습니다.")
    else:
        print_table(results)
        save_csv(results,  OUTPUT_DIR / "radius_stats_all.csv")
        save_chart(results, OUTPUT_DIR / "radius_stats_all.png")
