import pickle
import time

import numpy as np
import torch

from config import DEVICE, INDEX_DIR, TARGET_K


def load_cluster_index(dataset_name: str) -> dict:
    slug = dataset_name.split("/")[-1].replace("_test_subsampled", "").replace("_test", "")
    save_dir = INDEX_DIR / slug

    npz = np.load(save_dir / "cluster_index_box.npz")
    with open(save_dir / "cluster_index_box_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    return {
        "centroids":        torch.from_numpy(npz["centroids"]).to(DEVICE).to(torch.float32),
        "radii":            torch.from_numpy(npz["radii"]).to(DEVICE).to(torch.float32),
        "box_lower":        torch.from_numpy(npz["box_lower"]).to(DEVICE).to(torch.float32),
        "box_upper":        torch.from_numpy(npz["box_upper"]).to(DEVICE).to(torch.float32),
        "all_vectors":      torch.from_numpy(npz["all_vectors"]).to(DEVICE).to(torch.float32),
        "doc_boundaries":   npz["doc_boundaries"],
        "doc_cluster_sets": meta["doc_cluster_sets"],
        "metadata":         meta["metadata"],
    }


def _build_doc_remap(doc_cluster_sets, image_embeddings, ds, filename_to_corpus_idx) -> dict:
    num_indexed_docs = len(doc_cluster_sets)
    if num_indexed_docs == len(image_embeddings):
        return {i: i for i in range(num_indexed_docs)}
    elif num_indexed_docs == len(ds):
        return {i: filename_to_corpus_idx[ds[i]["image_filename"]] for i in range(num_indexed_docs)}
    raise ValueError(
        f"인덱스 문서 수({num_indexed_docs})가 "
        f"이미지 수({len(image_embeddings)}) 또는 데이터셋 크기({len(ds)})와 매칭되지 않습니다."
    )


def box_bound_search(
    query_embeddings: list,
    image_embeddings: list,
    idx_assets: dict,
    filename_to_corpus_idx: dict,
    ds,
    device: str = DEVICE,
) -> tuple[list, float, float, list, list, list]:
    centroids        = idx_assets["centroids"]
    box_lower        = idx_assets["box_lower"]   # (K, D)
    box_upper        = idx_assets["box_upper"]   # (K, D)
    doc_cluster_sets = idx_assets["doc_cluster_sets"]

    num_indexed_docs = len(doc_cluster_sets)
    sehee_to_corpus  = _build_doc_remap(doc_cluster_sets, image_embeddings, ds, filename_to_corpus_idx)

    n = len(query_embeddings)
    C_doc = 0
    ranked_results = []
    ubound_ranges  = []
    score_ranges   = []
    all_ubounds    = []

    t0 = time.perf_counter()
    for q_idx in range(n):
        query_emb = query_embeddings[q_idx].to(device).to(torch.float32)  # (m, D)

        # Box Bound:
        #   cluster_bound[t,j] = ⟨q_t, c_j⟩
        #                       + Σ_s max(q_{t,s}·ℓ_{j,s}, q_{t,s}·u_{j,s})
        #
        # 벡터화:
        #   pos_part[t,j] = clamp(q_t, min=0) · u_j  — 양수 차원은 upper 선택
        #   neg_part[t,j] = clamp(q_t, max=0) · ℓ_j  — 음수 차원은 lower 선택
        q_c_dot  = torch.matmul(query_emb, centroids.T)              # (m, K)
        pos_part = torch.matmul(query_emb.clamp(min=0), box_upper.T) # (m, K)
        neg_part = torch.matmul(query_emb.clamp(max=0), box_lower.T) # (m, K)
        cluster_bounds = q_c_dot + pos_part + neg_part                # (m, K)

        upper_bounds = []
        for sehee_doc_idx in range(num_indexed_docs):
            J_D = list(doc_cluster_sets[sehee_doc_idx])
            if len(J_D) == 0:
                upper_bounds.append((sehee_doc_idx, -999.0))
                continue
            u_bound = cluster_bounds[:, J_D].max(dim=1).values.sum().item()
            upper_bounds.append((sehee_doc_idx, u_bound))

        upper_bounds.sort(key=lambda x: x[1], reverse=True)

        valid_ubounds = [u for _, u in upper_bounds if u > -999.0]
        ubound_ranges.append(
            (min(valid_ubounds), max(valid_ubounds)) if valid_ubounds else (0.0, 0.0)
        )
        all_ubounds.append(valid_ubounds)

        tau = None
        top_k_buffer    = []
        seen_corpus_ids = set()
        computed_scores = []

        for sehee_doc_idx, u_bound in upper_bounds:
            corpus_idx = sehee_to_corpus[sehee_doc_idx]

            if tau is not None and u_bound <= tau:
                break

            doc_emb = image_embeddings[corpus_idx].to(device).to(torch.float32)

            with torch.no_grad():
                score_val = torch.einsum("nd,sd->ns", query_emb, doc_emb).max(dim=1)[0].sum().item()

            C_doc += 1
            computed_scores.append(score_val)

            if corpus_idx in seen_corpus_ids:
                for idx, (old_score, c_id) in enumerate(top_k_buffer):
                    if c_id == corpus_idx and score_val > old_score:
                        top_k_buffer[idx] = (score_val, corpus_idx)
                        break
                top_k_buffer.sort(key=lambda x: x[0], reverse=True)
            else:
                top_k_buffer.append((score_val, corpus_idx))
                top_k_buffer.sort(key=lambda x: x[0], reverse=True)
                seen_corpus_ids.add(corpus_idx)
                if len(top_k_buffer) > TARGET_K:
                    popped = top_k_buffer.pop()
                    seen_corpus_ids.remove(popped[1])

            if len(top_k_buffer) == TARGET_K:
                tau = top_k_buffer[-1][0]

        score_ranges.append(
            (min(computed_scores), max(computed_scores)) if computed_scores else (0.0, 0.0)
        )
        ranked_results.append([corpus_idx for _, corpus_idx in top_k_buffer])

    t_our = time.perf_counter() - t0

    total_naive  = n * len(image_embeddings)
    if C_doc > total_naive:
        print(f"⚠️ [WARNING] C_doc({C_doc})이 이론상 최대({total_naive})를 초과했습니다.")
    pruning_rate = 1.0 - (C_doc / total_naive)

    print(f"  [결과] 문서 단위 비교 횟수 축소: {total_naive}회 -> {C_doc}회")
    print(f"  [결과] 최종 Pruning Rate (ρ): {pruning_rate:.2%}")

    return ranked_results, t_our, pruning_rate, ubound_ranges, score_ranges, all_ubounds
