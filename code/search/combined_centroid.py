"""
Combined Centroid Pruning (Level 1+2)
======================================
상한선 공식:
  residual[t, j]  = min( ||q_t|| · r_j,  Σ_b ||q_t^(b)|| · ρ_j^(b) )
  cluster_bound[t, j] = ⟨q_t, c_j⟩ + residual[t, j]
  doc_bound[d]        = Σ_t  max_{j∈J(D)}  cluster_bound[t, j]

L1(global)과 L2(block) 잔차 중 더 작은 값(tighter bound)을 per-(token, cluster)로 선택합니다.
단독 L1 또는 L2보다 항상 같거나 더 타이트한 상한선을 보장합니다.
"""

import time

import torch

from config import DEVICE, TARGET_K
from search.block_centroid import _build_doc_remap


def combined_centroid_search(
    query_embeddings: list,
    image_embeddings: list,
    idx_assets: dict,
    filename_to_corpus_idx: dict,
    ds,
    device: str = DEVICE,
) -> tuple[list, float, float, list, list, list]:
    centroids        = idx_assets["centroids"]        # (K, D)
    radii            = idx_assets["radii"]            # (K,)
    block_radii      = idx_assets["block_radii"]      # (K, B)
    doc_cluster_sets = idx_assets["doc_cluster_sets"]

    num_indexed_docs = len(doc_cluster_sets)
    sehee_to_corpus  = _build_doc_remap(
        doc_cluster_sets, image_embeddings, ds, filename_to_corpus_idx
    )

    K, D     = centroids.shape
    B        = block_radii.shape[1]
    block_sz = D // B

    n     = len(query_embeddings)
    C_doc = 0
    ranked_results = []
    ubound_ranges  = []
    score_ranges   = []
    all_ubounds    = []

    t0 = time.perf_counter()
    for q_idx in range(n):
        query_emb = query_embeddings[q_idx].to(device).to(torch.float32)  # (m, D)
        m = query_emb.shape[0]

        q_c_dot       = torch.matmul(query_emb, centroids.T)              # (m, K)
        q_norm        = torch.norm(query_emb, p=2, dim=-1, keepdim=True)  # (m, 1)
        q_block_norms = torch.norm(
            query_emb.reshape(m, B, block_sz), p=2, dim=-1
        )                                                                   # (m, B)

        l1_residual = q_norm * radii.unsqueeze(0)                          # (m, K)
        l2_residual = torch.matmul(q_block_norms, block_radii.T)           # (m, K)

        cluster_bounds = q_c_dot + torch.minimum(l1_residual, l2_residual) # (m, K)

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

        tau             = None
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

    t_search = time.perf_counter() - t0

    total_naive  = n * len(image_embeddings)
    pruning_rate = 1.0 - (C_doc / total_naive)

    print(f"  [결과] 문서 단위 비교 횟수 축소: {total_naive}회 -> {C_doc}회")
    print(f"  [결과] 최종 Pruning Rate (ρ): {pruning_rate:.2%}")

    return ranked_results, t_search, pruning_rate, ubound_ranges, score_ranges, all_ubounds
