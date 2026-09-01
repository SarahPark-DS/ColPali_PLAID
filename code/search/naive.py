import time

import torch

from config import DEVICE


def naive_search(query_embeddings: list, image_embeddings: list) -> tuple:
    # Stack docs once: (n_p, n_patches, D) in float32
    docs = torch.stack([p.to(DEVICE).to(torch.float32) for p in image_embeddings])

    t0 = time.perf_counter()
    n_q = len(query_embeddings)
    scores = torch.zeros(n_q, len(image_embeddings))
    for i, q in enumerate(query_embeddings):
        q = q.to(DEVICE).to(torch.float32)  # (n_q_tokens, D)
        # scores per doc: max over patches then sum over query tokens
        scores[i] = torch.einsum("nd,psd->nps", q, docs).max(dim=2)[0].sum(dim=0)
    t_naive = time.perf_counter() - t0
    return scores, t_naive
