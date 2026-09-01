"""
Block Residual Bound Index Builder
===================================
기존 cluster_index.npz + cluster_index_meta.pkl 을 활용해서
블록별 반지름 block_radii (K, B) 를 추가로 계산하고 저장합니다.

저장 결과:
  cluster_index_block.npz
    - centroids     : (K, D)
    - radii         : (K,)      기존 전체 radius (비교용)
    - block_radii   : (K, B)    블록별 radius ρ_j^(b)  ← 핵심 추가
    - assignments   : (N,)
    - doc_boundaries: (M+1,)
    - all_vectors   : (N, D)

  cluster_index_block_meta.pkl
    - 기존 meta 동일 + metadata에 B, block_size 추가
"""

import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# ─────────────────────────────────────────────
# 1. 로드
# ─────────────────────────────────────────────
def load_existing_index(save_dir: Path):
    print("[1] 기존 인덱스 로드 중...")
    npz = np.load(save_dir / "cluster_index.npz")
    with open(save_dir / "cluster_index_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    data = {
        "centroids"          : npz["centroids"],
        "radii"              : npz["radii"],
        "assignments"        : npz["assignments"],
        "doc_boundaries"     : npz["doc_boundaries"],
        "all_vectors"        : npz["all_vectors"],
        "cluster_to_vectors" : meta["cluster_to_vectors"],
        "doc_cluster_sets"   : meta["doc_cluster_sets"],
        "doc_ids"            : meta["doc_ids"],
        "metadata"           : meta["metadata"],
    }
    K, D = data["centroids"].shape
    N    = len(data["assignments"])
    print(f"    K={K}, D={D}, N={N:,}")
    return data


# ─────────────────────────────────────────────
# 2. block_radii 계산
# ─────────────────────────────────────────────
def compute_block_radii(data: dict, B: int = 4, device: str = "cuda:0") -> np.ndarray:
    """
    ρ_j^(b) = max_{r in R_j} ||r^(b)||_2
    r = v - c_j  (residual),  r^(b) = b번째 블록의 residual

    Returns:
        block_radii: (K, B) float32
    """
    centroids          = data["centroids"]
    all_vectors        = data["all_vectors"]
    cluster_to_vectors = data["cluster_to_vectors"]

    K, D     = centroids.shape
    block_sz = D // B

    print(f"\n[2] block_radii 계산 중... (K={K}, B={B}, block_size={block_sz})")

    block_radii     = np.zeros((K, B), dtype=np.float32)
    centroids_t     = torch.from_numpy(centroids).to(device)

    for j in tqdm(range(K), desc="  computing block radii"):
        member_indices = cluster_to_vectors[j]
        if len(member_indices) == 0:
            continue

        members  = torch.from_numpy(
            all_vectors[np.array(member_indices)]
        ).to(device)                              # (|C_j|, D)

        c_j      = centroids_t[j]                # (D,)
        residual = members - c_j                  # (|C_j|, D)

        for b in range(B):
            start           = b * block_sz
            end             = start + block_sz
            r_b             = residual[:, start:end]   # (|C_j|, block_sz)
            block_radii[j, b] = torch.norm(r_b, dim=1).max().item()

    print(f"  완료. block_radii shape: {block_radii.shape}")
    return block_radii


# ─────────────────────────────────────────────
# 3. 저장
# ─────────────────────────────────────────────
def save_block_index(data: dict, block_radii: np.ndarray,
                     save_dir: Path, B: int):
    print(f"\n[3] 저장 중...")

    # npz
    np.savez(
        save_dir / "cluster_index_block.npz",
        centroids       = data["centroids"],
        radii           = data["radii"],
        block_radii     = block_radii,
        assignments     = data["assignments"],
        doc_boundaries  = data["doc_boundaries"],
        all_vectors     = data["all_vectors"],
    )

    # pkl — metadata에 B, block_size 추가
    meta = {
        "cluster_to_vectors" : data["cluster_to_vectors"],
        "doc_cluster_sets"   : data["doc_cluster_sets"],
        "doc_ids"            : data["doc_ids"],
        "metadata"           : {
            **data["metadata"],
            "B"          : B,
            "block_size" : data["centroids"].shape[1] // B,
        },
    }
    with open(save_dir / "cluster_index_block_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"  저장 완료: {save_dir}")
    print(f"    cluster_index_block.npz")
    print(f"    cluster_index_block_meta.pkl")


# ─────────────────────────────────────────────
# 4. 비교 및 검증
# ─────────────────────────────────────────────
def compare_and_validate(data: dict, block_radii: np.ndarray):
    """
    기존 radius vs block bound 비교.
    block bound = sum_b ρ_j^(b) 는 항상 r_j 이하여야 함.
    """
    print(f"\n[4] 기존 radius vs block bound 비교")

    radii            = data["radii"]
    cluster_to_vectors = data["cluster_to_vectors"]

    block_bound_sum = block_radii.sum(axis=1)    # (K,)
    nonempty        = np.array([len(c) > 0 for c in cluster_to_vectors])

    print(f"  {'':25} {'mean':>8} {'max':>8}")
    print(f"  {'-'*42}")
    print(f"  {'radius r_j (기존)':<25} "
          f"{radii[nonempty].mean():>8.4f} {radii[nonempty].max():>8.4f}")
    print(f"  {'block bound sum_b rho^b':<25} "
          f"{block_bound_sum[nonempty].mean():>8.4f} {block_bound_sum[nonempty].max():>8.4f}")

    improvement = (1 - block_bound_sum[nonempty].mean() / radii[nonempty].mean()) * 100
    print(f"  {'개선율 (mean 기준)':<25} {improvement:>7.2f}%")

    # 수학적 검증: block bound <= radius 항상 성립해야 함
    violations = np.sum(block_bound_sum[nonempty] > radii[nonempty] + 1e-5)
    print(f"\n  [검증] block_bound > radius 위반 케이스: {violations}개")
    if violations == 0:
        print("  ✓ 수학적으로 올바름 — block bound가 항상 radius bound 이하")
    else:
        print("  ✗ 위반 케이스 존재 — 구현 확인 필요")

    # 블록별 통계
    B = block_radii.shape[1]
    print(f"\n  블록별 radius 통계:")
    print(f"  {'block':>6} {'mean':>8} {'max':>8}")
    print(f"  {'-'*24}")
    for b in range(B):
        print(f"  {b:>6} "
              f"{block_radii[nonempty, b].mean():>8.4f} "
              f"{block_radii[nonempty, b].max():>8.4f}")


# ─────────────────────────────────────────────
# 로드 유틸리티
# ─────────────────────────────────────────────
def load_block_index(save_dir: Path):
    npz = np.load(save_dir / "cluster_index_block.npz")
    with open(save_dir / "cluster_index_block_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return {
        "centroids"          : npz["centroids"],
        "radii"              : npz["radii"],
        "block_radii"        : npz["block_radii"],
        "assignments"        : npz["assignments"],
        "doc_boundaries"     : npz["doc_boundaries"],
        "all_vectors"        : npz["all_vectors"],
        "cluster_to_vectors" : meta["cluster_to_vectors"],
        "doc_cluster_sets"   : meta["doc_cluster_sets"],
        "doc_ids"            : meta["doc_ids"],
        "metadata"           : meta["metadata"],
    }


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
if __name__ == "__main__":
    INDEX_DIR     = Path("./cluster_index_output_4x")
    B             = 4
    DEVICE        = "cuda:0"
    
    dirs = os.listdir(INDEX_DIR)
    dirs = [d for d in dirs if not d.endswith(".csv") or not d.endswith(".png")]
    
    for d in dirs:
        print(f"\n=== Processing dataset: {d} ===")
        save_dir = INDEX_DIR / d

        data        = load_existing_index(save_dir)
        block_radii = compute_block_radii(data, B=B, device=DEVICE)
        save_block_index(data, block_radii, save_dir, B=B)
        compare_and_validate(data, block_radii)

    print("\n✓ Block index 구성 완료.")
    
    
    
    