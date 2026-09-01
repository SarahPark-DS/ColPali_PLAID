"""
Cluster Index Builder for ColPali + ViDoRe
===========================================
오프라인 서칭을 위한 임베딩 및 클러스터 인덱스 생성
"""

#%%
import os
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from colpali_engine.models import ColPali, ColPaliProcessor
from fastkmeans import FastKMeans  # pip install fastkmeans


# %% 설정
DATASETS = [
    "vidore/arxivqa_test_subsampled",
    "vidore/docvqa_test_subsampled",
    "vidore/infovqa_test_subsampled",
    "vidore/tabfquad_test_subsampled",
    "vidore/tatdqa_test",
    "vidore/shiftproject_test",
    "vidore/syntheticDocQA_artificial_intelligence_test",
    "vidore/syntheticDocQA_energy_test",
    "vidore/syntheticDocQA_government_reports_test",
    "vidore/syntheticDocQA_healthcare_industry_test",
]
 
MODEL_NAME   = "vidore/colpali-v1.3"
 
# K = int(2 ** np.floor(np.log2(16 * np.sqrt(N))))   # 클러스터 수 (ColBERTv2 클러스터 수 결정 공식 코드 기준) 
# N = 문서 수 X  평균 토큰 수(Patch 수)

MULTIPLIER = 16 * 4            # K 결정 공식의 상수 (ColBERTv2는 16)
BATCH_SIZE   = 8             # 이미지 인코딩 배치 크기
OUTPUT_DIR   = Path("./cluster_index_output_4x")
DEVICE       = "cuda:0"
 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#%% 모델 로드
def load_model():
    print("[1] 모델 로드 중...")
    model = ColPali.from_pretrained(
        MODEL_NAME,
        torch_dtype = torch.bfloat16,
        device_map = DEVICE).eval()
    processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
    return model, processor

#%% 데이터셋 로드
def encode_dataset(ds_name, model, processor):
    """
    Returns:
        all_vectors  : (N_total, D)  float32  — 전체 patch 벡터
        doc_boundaries: (M+1,)       int32    — doc_boundaries[d]:doc_boundaries[d+1] 이 문서 d의 범위
        doc_ids      : List[str]               — 문서 식별자
    """
    print("[2] 데이터셋 인코딩 중...")
 
    all_vectors   = []   # List of (n_patches, D) tensors
    doc_ids       = []
 
    print(f"  → {ds_short_name(ds_name)}")
    dataset = load_dataset(ds_name, split="test")
    
    
    # --[변경] 벤치마크와 동일하게 이미지 중복값 있는 경우 제거--
    seen = set()
    images = []
    for i in range(len(dataset)):
        fn = dataset[i]["image_filename"]
        if fn not in seen:
            seen.add(fn)
            images.append(dataset[i]["image"])
    
    # 배치 단위로 인코딩
    for batch_start in tqdm(range(0, len(images), BATCH_SIZE),
                            desc=f"    encoding", leave=False):
        batch_images = images[batch_start : batch_start + BATCH_SIZE]

        # ColPali 전처리
        batch_inputs = processor.process_images(batch_images).to(DEVICE)

        with torch.no_grad():
            # shape: (B, n_patches, D)
            batch_embeddings = model(**batch_inputs)

        # float32로 변환 후 CPU 이동
        batch_embeddings = batch_embeddings.float().cpu()

        for i, emb in enumerate(batch_embeddings):
            # emb: (n_patches, D)
            all_vectors.append(emb.numpy().astype(np.float32))
            doc_ids.append(f"{ds_short_name(ds_name)}__{batch_start + i}")
 
    # doc_boundaries: 각 문서의 벡터 시작 인덱스 (cumsum)
    lengths = [v.shape[0] for v in all_vectors]
    doc_boundaries = np.array([0] + list(np.cumsum(lengths)), dtype=np.int32)
 
    # 전체 벡터 행렬로 합치기: (N_total, D)
    all_vectors_mat = np.concatenate(all_vectors, axis = 0) # (N, D)
 
    print(f"  총 문서 수: {len(doc_ids)}, 총 벡터 수: {all_vectors_mat.shape[0]}, 벡터 차원: {all_vectors_mat.shape[1]}")
    return all_vectors_mat, doc_boundaries, doc_ids


# K-means 클러스터링 및 반지름 계산
def run_kmeans(all_vectors_mat: np.ndarray, MULTIPLIER: int = 16):
    """
    Returns:
        centroids   : (K, D)    float32  — c_j
        assignments : (N,)      int32    — a(i)
        radii       : (K,)      float32  — r_j
    """
    print(f"[3] FastKMeans 클러스터링 수행 중...)")
    K = int(2 ** np.floor(np.log2(MULTIPLIER * np.sqrt(all_vectors_mat.shape[0]))))

    kmeans = FastKMeans(d = all_vectors_mat.shape[1], 
                        k=K, 
                        verbose=True)
    kmeans.fit(all_vectors_mat)
 
    # centroids: (K, D)
    centroids = kmeans.centroids.astype(np.float32)
 
    # assignments: (N,)  — 각 벡터의 클러스터 번호
    assignments = kmeans.predict(all_vectors_mat).astype(np.int32)
    
    # cluster_to_vectors: 클러스터 j에 속한 벡터 인덱스 목록
    cluster_to_vectors = [[] for _ in range(K)]
    for vec_idx, cluster_j in enumerate(assignments):
        cluster_to_vectors[cluster_j].append(vec_idx)
 
    # radii: r_j = max_{u in C_j} ||u - c_j||
    print("  반지름(r_j) 계산 중...")
    radii = np.zeros(K, dtype=np.float32)

    for j in range(K):
        member_indices = np.array(cluster_to_vectors[j])
        if len(member_indices) == 0:
            continue
        
        members = torch.from_numpy(all_vectors_mat[member_indices]).to(DEVICE)  # (|C_j|, D)
        c_j     = torch.from_numpy(centroids[j]).to(DEVICE)                     # (D,)
        
        dists   = torch.norm(members - c_j, dim=1)  # (|C_j|,)
        radii[j] = dists.max().item()
 
    print(f"  클러스터링 완료. centroids: {centroids.shape}, assignments: {assignments.shape}")
    return centroids, assignments, radii, cluster_to_vectors


def save_index(
    ds_name         : str,
    all_vectors_mat : np.ndarray,
    centroids       : np.ndarray,
    assignments     : np.ndarray,
    radii           : np.ndarray,
    doc_boundaries  : np.ndarray,
    doc_ids         : list,
    cluster_to_vectors: list,
):
    K = centroids.shape[0]
    M = len(doc_ids)
 
    # doc_cluster_sets: J(D)
    doc_cluster_sets = []
    for d in range(M):
        s   = doc_boundaries[d]
        e   = doc_boundaries[d + 1]
        j_d = set(assignments[s:e].tolist())
        doc_cluster_sets.append(j_d)
 
    # 저장 디렉토리
    save_dir = OUTPUT_DIR / ds_short_name(ds_name)
    save_dir.mkdir(parents=True, exist_ok=True)
 
    # npz
    npz_path = save_dir / "cluster_index.npz"
    np.savez(
        npz_path,
        centroids      = centroids,
        radii          = radii,
        assignments    = assignments,
        doc_boundaries = doc_boundaries,
        all_vectors    = all_vectors_mat,
    )
 
    # pkl
    pkl_path = save_dir / "cluster_index_meta.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "cluster_to_vectors": cluster_to_vectors,
                "doc_cluster_sets"  : doc_cluster_sets,
                "doc_ids"           : doc_ids,
                "metadata": {
                    "dataset"     : ds_name,
                    "K"           : K,
                    "D"           : centroids.shape[1],
                    "num_docs"    : M,
                    "num_vectors" : len(assignments),
                    "model_name"  : MODEL_NAME,
                },
            },
            f,
        )
 
    # 통계
    cluster_sizes = [len(c) for c in cluster_to_vectors]
    avg_jd        = np.mean([len(s) for s in doc_cluster_sets])
    print(f"  저장 완료: {save_dir}")
    print(f"  K={K}, 문서={M}, 벡터={len(assignments):,}")
    print(f"  클러스터 크기 — min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}")
    print(f"  반지름(r_j)   — min={radii.min():.4f}, max={radii.max():.4f}, mean={radii.mean():.4f}")
    print(f"  |J(D)| 평균   — {avg_jd:.1f}\n")
    
    
def load_cluster_index(ds_name: str, output_dir: Path = OUTPUT_DIR):
    save_dir = output_dir / ds_short_name(ds_name)
    npz = np.load(save_dir / "cluster_index.npz")
    with open(save_dir / "cluster_index_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return {
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
    
    
def ds_short_name(ds_name: str) -> str:
    return ds_name.split("/")[-1].replace("_test_subsampled", "").replace("_test", "")


if __name__ == "__main__":
    model, processor = load_model()
 
    for ds_name in DATASETS:
        short = ds_short_name(ds_name)
        print(f"{'='*60}")
        print(f"[데이터셋] {short}")
        print(f"{'='*60}")
 
        # 이미 처리된 경우 스킵
        save_dir = OUTPUT_DIR / short
        if (save_dir / "cluster_index.npz").exists():
            print(f"  이미 존재, 스킵: {save_dir}\n")
            continue
 
        # 2. 인코딩
        print("[2] 인코딩 중...")
        all_vectors_mat, doc_boundaries, doc_ids = encode_dataset(
            ds_name, model, processor
        )
 
        # 3. K-Means
        print("[3] K-Means 클러스터링 중...")
        centroids, assignments, radii, cluster_to_vectors = run_kmeans(all_vectors_mat, MULTIPLIER)
 
        # 4. 저장
        print("[4] 저장 중...")
        save_index(
            ds_name, all_vectors_mat,
            centroids, assignments, radii,
            doc_boundaries, doc_ids, cluster_to_vectors,
        )
 
    print("✓ 전체 완료.")
    

    
    

# %%
