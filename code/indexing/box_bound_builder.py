"""
Box Bound Index Builder
=======================
기존 cluster_index.npz + cluster_index_meta.pkl 을 활용해서
좌표별 잔차 범위 box_lower (K, D), box_upper (K, D) 를 계산하고 저장합니다.

상한 공식:
  cluster_bound[t,j] = ⟨q_t, c_j⟩ + Σ_s max(q_{t,s}·ℓ_{j,s}, q_{t,s}·u_{j,s})
  → q_{t,s} ≥ 0 이면 u_{j,s}, q_{t,s} < 0 이면 ℓ_{j,s} 선택

저장 결과:
  cluster_index_box.npz
    - centroids     : (K, D)
    - radii         : (K,)       기존 전체 radius (비교용)
    - box_lower     : (K, D)     좌표별 잔차 최솟값  ← 핵심 추가
    - box_upper     : (K, D)     좌표별 잔차 최댓값  ← 핵심 추가
    - assignments   : (N,)
    - doc_boundaries: (M+1,)
    - all_vectors   : (N, D)

  cluster_index_box_meta.pkl
    - 기존 meta 동일
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ─────────────────────────────────────────────
# 1. 로드
# ─────────────────────────────────────────────
def load_existing_index(save_dir: Path) -> dict:
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
# 2. box_lower / box_upper 계산
# ─────────────────────────────────────────────
def compute_box_bounds(data: dict, device: str = "cuda:0") -> tuple[np.ndarray, np.ndarray]:
    """
    ℓ_{j,s} = min_{r ∈ R_j} r_s
    u_{j,s} = max_{r ∈ R_j} r_s
    r = v - c_j  (residual)

    Returns:
        box_lower : (K, D) float32
        box_upper : (K, D) float32
    """
    centroids          = data["centroids"]
    all_vectors        = data["all_vectors"]
    cluster_to_vectors = data["cluster_to_vectors"]

    K, D = centroids.shape
    print(f"\n[2] box_lower / box_upper 계산 중... (K={K}, D={D})")

    box_lower = np.zeros((K, D), dtype=np.float32)
    box_upper = np.zeros((K, D), dtype=np.float32)

    centroids_t = torch.from_numpy(centroids).to(device)

    for j in tqdm(range(K), desc="  computing box bounds"):
        member_indices = cluster_to_vectors[j]
        if len(member_indices) == 0:
            continue

        members  = torch.from_numpy(
            all_vectors[np.array(member_indices)]
        ).to(device)                          # (|C_j|, D)

        c_j      = centroids_t[j]            # (D,)
        residual = members - c_j              # (|C_j|, D)

        box_lower[j] = residual.min(dim=0).values.cpu().numpy()
        box_upper[j] = residual.max(dim=0).values.cpu().numpy()

    print(f"  완료. box_lower/box_upper shape: {box_lower.shape}")
    return box_lower, box_upper


# ─────────────────────────────────────────────
# 3. 저장
# ─────────────────────────────────────────────
def save_box_index(data: dict, box_lower: np.ndarray, box_upper: np.ndarray,
                   save_dir: Path) -> None:
    print(f"\n[3] 저장 중...")

    np.savez(
        save_dir / "cluster_index_box.npz",
        centroids       = data["centroids"],
        radii           = data["radii"],
        box_lower       = box_lower,
        box_upper       = box_upper,
        assignments     = data["assignments"],
        doc_boundaries  = data["doc_boundaries"],
        all_vectors     = data["all_vectors"],
    )

    meta = {
        "cluster_to_vectors" : data["cluster_to_vectors"],
        "doc_cluster_sets"   : data["doc_cluster_sets"],
        "doc_ids"            : data["doc_ids"],
        "metadata"           : data["metadata"],
    }
    with open(save_dir / "cluster_index_box_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"  저장 완료: {save_dir}")
    print(f"    cluster_index_box.npz")
    print(f"    cluster_index_box_meta.pkl")


# ─────────────────────────────────────────────
# 4. 비교 및 검증
# ─────────────────────────────────────────────
def compare_and_validate(data: dict, box_lower: np.ndarray, box_upper: np.ndarray) -> None:
    """
    box bound vs 기존 radius 비교.
    box bound의 잔차 범위 [ℓ, u]는 구(sphere) 안에 포함되므로
    max_s |ℓ_{j,s}|, max_s u_{j,s} <= r_j 가 성립해야 함.
    """
    print(f"\n[4] 기존 radius vs box bound 비교")

    radii              = data["radii"]
    cluster_to_vectors = data["cluster_to_vectors"]
    nonempty           = np.array([len(c) > 0 for c in cluster_to_vectors])

    # box bound 상한의 "크기": 좌표별 최대 절댓값 범위
    box_range_norm = np.linalg.norm(
        np.maximum(np.abs(box_lower), np.abs(box_upper)), axis=1
    )   # (K,) — 대략적인 box 크기

    print(f"  {'':30} {'mean':>8} {'max':>8}")
    print(f"  {'-'*48}")
    print(f"  {'radius r_j (기존 sphere)':<30} "
          f"{radii[nonempty].mean():>8.4f} {radii[nonempty].max():>8.4f}")
    print(f"  {'box range norm (참고용)':<30} "
          f"{box_range_norm[nonempty].mean():>8.4f} {box_range_norm[nonempty].max():>8.4f}")

    # 수학적 검증: ℓ_{j,s} <= 0 <= u_{j,s} 는 클러스터가 공허하지 않은 경우 성립해야 함
    lower_violations = np.sum((box_upper[nonempty] < box_lower[nonempty]).any(axis=1))
    print(f"\n  [검증] upper < lower 위반 케이스: {lower_violations}개")
    if lower_violations == 0:
        print("  ✓ box_lower <= box_upper 항상 성립")
    else:
        print("  ✗ 위반 케이스 존재 — 구현 확인 필요")

    # box 범위가 0을 포함하는지 확인 (잔차가 양수/음수 모두 존재해야 함)
    contains_zero = np.sum(
        (box_lower[nonempty] <= 0).all(axis=1) & (box_upper[nonempty] >= 0).all(axis=1)
    )
    print(f"  [참고] 모든 좌표에서 0을 포함하는 클러스터: {contains_zero}/{nonempty.sum()}")

    # 좌표별 box 폭 통계
    box_width = (box_upper - box_lower)[nonempty]
    print(f"\n  좌표별 box 폭 통계 (전 차원 평균):")
    print(f"    mean width: {box_width.mean():.4f}")
    print(f"    max  width: {box_width.max():.4f}")
    print(f"    min  width: {box_width.min():.4f}")


# ─────────────────────────────────────────────
# 로드 유틸리티
# ─────────────────────────────────────────────
def load_box_index(save_dir: Path) -> dict:
    npz = np.load(save_dir / "cluster_index_box.npz")
    with open(save_dir / "cluster_index_box_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return {
        "centroids"          : npz["centroids"],
        "radii"              : npz["radii"],
        "box_lower"          : npz["box_lower"],
        "box_upper"          : npz["box_upper"],
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
    INDEX_DIR = Path("./cluster_index_output_4x")
    DEVICE    = "cuda:0"

    dirs = [d for d in os.listdir(INDEX_DIR) if (INDEX_DIR / d).is_dir()]

    for d in dirs:
        print(f"\n{'='*55}")
        print(f"=== Processing dataset: {d} ===")
        print(f"{'='*55}")
        save_dir = INDEX_DIR / d

        data                  = load_existing_index(save_dir)
        box_lower, box_upper  = compute_box_bounds(data, device=DEVICE)
        save_box_index(data, box_lower, box_upper, save_dir)
        compare_and_validate(data, box_lower, box_upper)

    print("\n✓ Box bound index 구성 완료.")
