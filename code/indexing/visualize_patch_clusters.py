"""
Top-K 클러스터의 대표 패치를 원본 이미지 위에 하이라이트해서 시각화
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datasets import load_dataset
from PIL import Image

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
OUTPUT_DIR       = Path("./cluster_index_output")
DATASET_NAME     = "vidore/arxivqa_test_subsampled"
DATASET_SHORT    = "arxivqa"
TOP_N_CLUSTERS   = 10    # 보여줄 클러스터 수
PATCHES_PER_CLUSTER = 5  # 클러스터당 보여줄 패치 수
CLUSTER_RANK_START  = 11      # 0-based, 0이면 1등부터
CLUSTER_RANK_END    = 20     # 10이면 10등까지 (exclusive)
# 예: 11~20등 보려면 CLUSTER_RANK_START=10, CLUSTER_RANK_END=20


# ColPali 패치 설정 (PaliGemma 기준)
PATCH_SIZE   = 14    # 픽셀 단위 패치 크기
IMAGE_SIZE   = 448   # ColPali 입력 이미지 크기 (448x448)
GRID_SIZE    = IMAGE_SIZE // PATCH_SIZE  # 32 (32x32 = 1024 패치)


# ─────────────────────────────────────────────
# 인덱스 로드
# ─────────────────────────────────────────────
def load_index(dataset_short: str):
    save_dir = OUTPUT_DIR / dataset_short
    npz = np.load(save_dir / "cluster_index.npz")
    with open(save_dir / "cluster_index_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return {
        "assignments"        : npz["assignments"],         # (N,)
        "doc_boundaries"     : npz["doc_boundaries"],      # (M+1,)
        "cluster_to_vectors" : meta["cluster_to_vectors"], # List[List[int]]
        "doc_ids"            : meta["doc_ids"],            # List[str]
        "metadata"           : meta["metadata"],
    }


# ─────────────────────────────────────────────
# 전역 벡터 인덱스 → (문서 인덱스, 패치 번호) 변환
# ─────────────────────────────────────────────
def vec_idx_to_doc_patch(vec_idx: int, doc_boundaries: np.ndarray):
    """
    vec_idx: 전역 벡터 인덱스
    returns: (doc_idx, patch_idx_within_doc)
    """
    # doc_boundaries에서 vec_idx가 속한 문서 찾기
    doc_idx   = np.searchsorted(doc_boundaries, vec_idx, side="right") - 1
    patch_idx = vec_idx - doc_boundaries[doc_idx]
    return int(doc_idx), int(patch_idx)


# ─────────────────────────────────────────────
# 패치 번호 → 이미지 내 픽셀 좌표 변환
# ─────────────────────────────────────────────
def patch_idx_to_bbox(patch_idx: int, img_w: int, img_h: int):
    """
    patch_idx: 문서 내 패치 번호 (0 ~ 1030)
    returns: (x1, y1, x2, y2) — 원본 이미지 해상도 기준 픽셀 좌표

    PaliGemma는 이미지를 448x448로 리사이즈 후 14픽셀 단위로 분할.
    → 32x32 = 1024개 패치 + 추가 토큰 (앞쪽에 위치)
    추가 토큰(BOS 등)은 공간적 위치 없으므로 1024개 범위를 벗어난 patch_idx는 skip.
    """
    if patch_idx >= GRID_SIZE * GRID_SIZE:
        return None  # 추가 토큰 (BOS 등), 공간 위치 없음

    row = patch_idx // GRID_SIZE   # 0~31
    col = patch_idx  % GRID_SIZE   # 0~31

    # 448x448 기준 좌표
    x1_norm = col * PATCH_SIZE
    y1_norm = row * PATCH_SIZE
    x2_norm = x1_norm + PATCH_SIZE
    y2_norm = y1_norm + PATCH_SIZE

    # 원본 이미지 해상도로 스케일
    scale_x = img_w / IMAGE_SIZE
    scale_y = img_h / IMAGE_SIZE

    x1 = int(x1_norm * scale_x)
    y1 = int(y1_norm * scale_y)
    x2 = int(x2_norm * scale_x)
    y2 = int(y2_norm * scale_y)

    return x1, y1, x2, y2


# ─────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────
def visualize_top_clusters(idx, dataset):
    cluster_to_vectors = idx["cluster_to_vectors"]
    doc_boundaries     = idx["doc_boundaries"]

    # 크기순 정렬 → top N 클러스터
    sizes       = [(j, len(v)) for j, v in enumerate(cluster_to_vectors)]
    sizes.sort(key=lambda x: -x[1])
    top_clusters = sizes[CLUSTER_RANK_START:CLUSTER_RANK_END]

    n_rows = len(top_clusters)
    fig, axes = plt.subplots(
        n_rows, PATCHES_PER_CLUSTER,
        figsize=(PATCHES_PER_CLUSTER * 3, n_rows * 3)
    )
    fig.suptitle(f"Top-{CLUSTER_RANK_END - CLUSTER_RANK_START} Clusters — {DATASET_SHORT}\n"
                 f"(patch highlighted in red)", fontsize=14)

    for row_i, (cluster_j, cluster_size) in enumerate(top_clusters):
        members = np.array(cluster_to_vectors[cluster_j])

        # PATCHES_PER_CLUSTER개 랜덤 샘플
        chosen = np.random.choice(members,
                                  size=min(PATCHES_PER_CLUSTER, len(members)),
                                  replace=False)

        col_i = 0
        for vec_idx in chosen:
            ax = axes[row_i][col_i]

            doc_idx, patch_idx = vec_idx_to_doc_patch(vec_idx, doc_boundaries)

            # doc_id에서 이미지 인덱스 추출
            doc_id    = idx["doc_ids"][doc_idx]
            img_idx   = int(doc_id.split("__")[-1])
            image     = dataset[img_idx]["image"]
            img_w, img_h = image.size

            # 패치 bbox 계산
            bbox = patch_idx_to_bbox(patch_idx, img_w, img_h)

            ax.imshow(image)
            ax.axis("off")

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                rect = mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="red", facecolor="red", alpha=0.3
                )
                ax.add_patch(rect)
                ax.set_title(f"C{cluster_j} (n={cluster_size})\n"
                             f"doc={doc_idx} patch={patch_idx}",
                             fontsize=7)
            else:
                ax.set_title(f"C{cluster_j} — special token\n"
                             f"patch={patch_idx}", fontsize=7)

            col_i += 1

        # 샘플이 PATCHES_PER_CLUSTER보다 적으면 빈 칸 처리
        for empty_col in range(col_i, PATCHES_PER_CLUSTER):
            axes[row_i][empty_col].axis("off")

    plt.tight_layout()
    out_path = OUTPUT_DIR / DATASET_SHORT / f"cluster_patches_rank{CLUSTER_RANK_START+1}to{CLUSTER_RANK_END}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"저장: {out_path}")
    plt.show()


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    print("[1] 인덱스 로드...")
    idx = load_index(DATASET_SHORT)
    print(f"    K={idx['metadata']['K']}, 문서={idx['metadata']['num_docs']}")

    print("[2] 데이터셋 로드...")
    dataset = load_dataset(DATASET_NAME, split="test")

    print("[3] 시각화...")
    visualize_top_clusters(idx, dataset)

    print("✓ 완료.")