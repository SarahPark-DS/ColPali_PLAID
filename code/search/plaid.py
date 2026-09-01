import importlib
import time

import torch

from config import DEVICE, TARGET_K


def create_plaid_index(image_embeddings: list, device: str = DEVICE):
    if not importlib.util.find_spec("fast_plaid"):
        raise ImportError("pip install --no-deps fast-plaid fastkmeans")
    from fast_plaid import search
    index = search.FastPlaid(index="index")
    index.create(documents_embeddings=[d.to(device).to(torch.float32) for d in image_embeddings])
    return index


def _get_topk_batched(query_embeddings: list, plaid_index, k: int = TARGET_K,
                      batch_size: int = 128, device: str = DEVICE) -> list:
    scores_list = []
    for i in range(0, len(query_embeddings), batch_size):
        qs_batch = torch.nn.utils.rnn.pad_sequence(
            query_embeddings[i:i + batch_size], batch_first=True, padding_value=0
        ).to(device)
        scores_list.append(
            plaid_index.search(queries_embeddings=qs_batch.to(torch.float32), top_k=k)
        )
    return scores_list


def plaid_search(query_embeddings: list, image_embeddings: list,
                 k: int = TARGET_K, device: str = DEVICE) -> tuple[list, float, float]:
    t0 = time.perf_counter()
    index = create_plaid_index(image_embeddings, device=device)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    results_batched = _get_topk_batched(query_embeddings, index, k=k, device=device)
    t_plaid = time.perf_counter() - t0

    all_results = [q[:k] for batch in results_batched for q in batch]
    return all_results, t_build, t_plaid
