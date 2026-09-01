"""
Cluster Index 2D 투영 시각화
=============================
저장된 cluster_index를 로드해 벡터 공간을 2D로 투영합니다.

100만+ 벡터를 전부 투영하면 느리므로:
  1. 클러스터별로 균등 샘플링 (대표성 유지)
  2. PCA로 128 → 50차원 사전 축소 (UMAP/t-SNE 가속)
  3. UMAP(빠름) 또는 t-SNE로 2D 투영
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
OUTPUT_DIR    = Path("./cluster_index_output")
DATASET_SHORT = "arxivqa"              # 시각화할 데이터셋
METHOD        = "umap"                 # "umap" 또는 "tsne"
N_SAMPLE_PER_CLUSTER = 30              # 클러스터당 샘플 수
MAX_CLUSTERS_SHOWN   = 40              # 색으로 구분해 보여줄 클러스터 수 (큰 클러스터 우선)
PCA_DIM       = 50


def load_index(dataset_short: str):
    save_dir = OUTPUT_DIR / dataset_short
    npz = np.load(save_dir / "cluster_index.npz")
    with open(save_dir / "cluster_index_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return {
        "all_vectors"        : npz["all_vectors"],        # (N, D)
        "centroids"          : npz["centroids"],          # (K, D)
        "assignments"        : npz["assignments"],        # (N,)
        "cluster_to_vectors" : meta["cluster_to_vectors"],
        "metadata"           : meta["metadata"],
    }


def sample_vectors(idx):
    """클러스터별 균등 샘플링. 큰 클러스터 위주로 MAX_CLUSTERS_SHOWN개만 색 구분."""
    cluster_to_vectors = idx["cluster_to_vectors"]
    all_vectors        = idx["all_vectors"]

    # 클러스터를 크기순 정렬, 상위 MAX_CLUSTERS_SHOWN개만 강조
    sizes        = [(j, len(v)) for j, v in enumerate(cluster_to_vectors)]
    sizes.sort(key=lambda x: -x[1])
    top_clusters = set(j for j, _ in sizes[:MAX_CLUSTERS_SHOWN])

    sample_idx   = []
    sample_label = []   # 색칠용: top 클러스터면 클러스터 번호, 아니면 -1

    for j, members in enumerate(cluster_to_vectors):
        if len(members) == 0:
            continue
        members = np.array(members)
        take    = min(N_SAMPLE_PER_CLUSTER, len(members))
        chosen  = np.random.choice(members, size=take, replace=False)
        sample_idx.extend(chosen.tolist())
        label = j if j in top_clusters else -1
        sample_label.extend([label] * take)

    sample_idx   = np.array(sample_idx)
    sample_label = np.array(sample_label)
    vectors      = all_vectors[sample_idx]   # (S, D)

    print(f"  샘플 벡터 수: {len(sample_idx):,} (전체 {len(all_vectors):,} 중)")
    return vectors, sample_label


def project_2d(vectors):
    """PCA → UMAP/t-SNE 2D 투영."""
    from sklearn.decomposition import PCA

    print(f"  PCA {vectors.shape[1]} → {PCA_DIM} 차원...")
    vectors_pca = PCA(n_components=PCA_DIM, random_state=0).fit_transform(vectors)

    if METHOD == "umap":
        import umap  # pip install umap-learn
        print("  UMAP 2D 투영...")
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0)
        coords  = reducer.fit_transform(vectors_pca)
    else:
        from sklearn.manifold import TSNE
        print("  t-SNE 2D 투영...")
        coords = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(vectors_pca)

    return coords


def plot(coords, labels, dataset_short):
    plt.figure(figsize=(12, 10))

    # 강조되지 않은 점 (회색 배경)
    bg = labels == -1
    plt.scatter(coords[bg, 0], coords[bg, 1],
                c="lightgray", s=4, alpha=0.4, label="other clusters")

    # 강조 클러스터 (색상별)
    fg_labels = labels[~bg]
    unique    = np.unique(fg_labels)
    cmap      = plt.cm.get_cmap("tab20", len(unique))

    for i, j in enumerate(unique):
        mask = labels == j
        plt.scatter(coords[mask, 0], coords[mask, 1],
                    color=cmap(i), s=8, alpha=0.8)

    plt.title(f"Cluster Index 2D Projection — {dataset_short} ({METHOD.upper()})\n"
              f"top {MAX_CLUSTERS_SHOWN} clusters colored")
    plt.xlabel(f"{METHOD}-1")
    plt.ylabel(f"{METHOD}-2")
    plt.tight_layout()

    out_path = OUTPUT_DIR / dataset_short / f"cluster_projection_{METHOD}.png"
    plt.savefig(out_path, dpi=150)
    print(f"  저장: {out_path}")
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)

    print(f"[1] 인덱스 로드: {DATASET_SHORT}")
    idx = load_index(DATASET_SHORT)
    print(f"    K={idx['metadata']['K']}, 벡터={idx['metadata']['num_vectors']:,}")

    print("[2] 클러스터별 샘플링")
    vectors, labels = sample_vectors(idx)

    print("[3] 2D 투영")
    coords = project_2d(vectors)

    print("[4] 플롯")
    plot(coords, labels, DATASET_SHORT)

    print("✓ 완료.")