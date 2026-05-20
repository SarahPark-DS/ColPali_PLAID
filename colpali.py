#%%
import importlib
import math
import pickle
import time
from pathlib import Path

import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from datasets import load_dataset
from tqdm import tqdm

# ── 벤치마크 대상 데이터셋 ────────────────────────────
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

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ── 모델 로드 (한 번만) ───────────────────────────────
model_name = "vidore/colpali-v1.3"
model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
).eval()
processor = ColPaliProcessor.from_pretrained(model_name)

# ── PLAID 헬퍼 ────────────────────────────────────────
def create_plaid_index(ps, device="cuda:0"):
    if not importlib.util.find_spec("fast_plaid"):
        raise ImportError("pip install --no-deps fast-plaid fastkmeans")
    from fast_plaid import search
    index = search.FastPlaid(index="index")
    index.create(documents_embeddings=[d.to(device).to(torch.float32) for d in ps])
    return index

def compute_ndcg(ranked_indices, gt_idx, k):
    for rank, idx in enumerate(ranked_indices[:k]):
        if idx == gt_idx:
            return 1.0 / math.log2(rank + 2)
    return 0.0

def get_topk_plaid(qs, plaid_index, k=10, batch_size=128, device="cuda:0"):
    scores_list = []
    for i in range(0, len(qs), batch_size):
        qs_batch = torch.nn.utils.rnn.pad_sequence(
            qs[i : i + batch_size], batch_first=True, padding_value=0
        ).to(device)
        scores_list.append(
            plaid_index.search(queries_embeddings=qs_batch.to(torch.float32), top_k=k)
        )
    return scores_list

# ── 데이터셋별 벤치마크 함수 ──────────────────────────
def run_benchmark(dataset_name: str) -> dict:
    slug = dataset_name.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    try:
        ds = load_dataset(dataset_name, split="test")
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    queries = ds["query"]
    # None 쿼리 필터링
    valid_mask = [q is not None for q in queries]
    n_skipped = valid_mask.count(False)
    if n_skipped:
        print(f"  Warning: skipping {n_skipped} samples with None query")
    print(f"  Queries: {len(ds)} (valid: {sum(valid_mask)})")

    # 중복 제거 이미지 corpus 구축
    # 여러 쿼리가 같은 이미지를 공유하는 데이터셋(tabfquad, tatdqa 등)에 대응
    filename_to_corpus_idx = {}
    corpus_images = []
    for i in range(len(ds)):
        fn = ds[i]["image_filename"]
        if fn not in filename_to_corpus_idx:
            filename_to_corpus_idx[fn] = len(corpus_images)
            corpus_images.append(ds["image"][i])

    print(f"  Unique images (corpus): {len(corpus_images)}")

    # 각 쿼리의 정답 corpus 인덱스 (None 쿼리 제외)
    gt_indices = [
        filename_to_corpus_idx[ds[i]["image_filename"]]
        for i in range(len(ds)) if valid_mask[i]
    ]
    queries = [q for q, v in zip(queries, valid_mask) if v]

    # 이미지 임베딩 (캐시)
    cache_path = CACHE_DIR / f"{slug}_image_embeddings.pkl"
    t0 = time.perf_counter()
    if cache_path.exists():
        print("  Loading cached image embeddings...")
        with open(cache_path, "rb") as f:
            image_embeddings = pickle.load(f)
        if len(image_embeddings) != len(corpus_images):
            # corpus 크기가 다르면 캐시 무효화
            print(f"  Cache size mismatch ({len(image_embeddings)} vs {len(corpus_images)}), recomputing...")
            cache_path.unlink()
            image_embeddings = None
    else:
        image_embeddings = None

    if image_embeddings is None:
        print("  Generating image embeddings...")
        image_embeddings = []
        for img in tqdm(corpus_images, desc="  Image Embedding"):
            batch = processor.process_images([img.convert("RGB")]).to("cuda:0")
            with torch.no_grad():
                emb = model(**batch)
            image_embeddings.append(emb[0].cpu())
        with open(cache_path, "wb") as f:
            pickle.dump(image_embeddings, f)
    t_img_emb = time.perf_counter() - t0

    # 쿼리 임베딩
    print("  Generating query embeddings...")
    query_embeddings = []
    for q in tqdm(queries, desc="  Query Embedding"):
        batch = processor.process_queries([q]).to("cuda:0")
        with torch.no_grad():
            emb = model(**batch)
        query_embeddings.append(emb[0].cpu())

    n = len(queries)

    # [1] Naive
    t0 = time.perf_counter()
    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    t_naive = time.perf_counter() - t0

    correct_naive = sum(
        1 for i in range(n)
        if scores[i].argmax().item() == gt_indices[i]
    )

    # [2] Fast PLAID
    t0 = time.perf_counter()
    plaid_index = create_plaid_index(image_embeddings, device="cuda:0")
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    plaid_results_batched = get_topk_plaid(query_embeddings, plaid_index, k=10, device="cuda:0")
    t_plaid = time.perf_counter() - t0

    all_plaid_results = [q for batch in plaid_results_batched for q in batch]
    correct_plaid = sum(
        1 for i in range(n)
        if all_plaid_results[i][0][0] == gt_indices[i]
    )

    # nDCG@5 and nDCG@10
    ndcg5_naive_vals, ndcg10_naive_vals = [], []
    ndcg5_plaid_vals, ndcg10_plaid_vals = [], []
    for i in range(n):
        ranked_naive = scores[i].argsort(descending=True).tolist()
        ndcg5_naive_vals.append(compute_ndcg(ranked_naive, gt_indices[i], 5))
        ndcg10_naive_vals.append(compute_ndcg(ranked_naive, gt_indices[i], 10))
        ranked_plaid = [r[0] for r in all_plaid_results[i]]
        ndcg5_plaid_vals.append(compute_ndcg(ranked_plaid, gt_indices[i], 5))
        ndcg10_plaid_vals.append(compute_ndcg(ranked_plaid, gt_indices[i], 10))

    ndcg5_naive  = sum(ndcg5_naive_vals)  / n
    ndcg10_naive = sum(ndcg10_naive_vals) / n
    ndcg5_plaid  = sum(ndcg5_plaid_vals)  / n
    ndcg10_plaid = sum(ndcg10_plaid_vals) / n

    print(f"\n  {'Method':<26} {'R@1':>7} {'nDCG@5':>8} {'nDCG@10':>9} {'Img Emb(s)':>11} {'Search(s)':>10} {'Total(s)':>9}")
    print(f"  {'-'*82}")
    print(f"  {'Naive (score_multi_vector)':<26} {correct_naive/n:>6.2%} {ndcg5_naive:>8.4f} {ndcg10_naive:>9.4f} {t_img_emb:>11.2f} {t_naive:>10.2f} {t_img_emb+t_naive:>9.2f}")
    print(f"  {'Fast PLAID (search only)':<26} {correct_plaid/n:>6.2%} {ndcg5_plaid:>8.4f} {ndcg10_plaid:>9.4f} {t_img_emb:>11.2f} {t_plaid:>10.2f} {t_img_emb+t_plaid:>9.2f}")
    print(f"  {'Fast PLAID (build+search)':<26} {correct_plaid/n:>6.2%} {ndcg5_plaid:>8.4f} {ndcg10_plaid:>9.4f} {t_img_emb:>11.2f} {t_build+t_plaid:>10.2f} {t_img_emb+t_build+t_plaid:>9.2f}")

    result = {
        "dataset": dataset_name,
        "n_queries": n,
        "scores": scores,
        "all_plaid_results": all_plaid_results,
        "filename_to_corpus_idx": filename_to_corpus_idx,
        "gt_indices": gt_indices,
        "ds": ds,
        "queries": queries,
        "t_img_emb": t_img_emb,
        "t_naive": t_naive,
        "t_build": t_build,
        "t_plaid": t_plaid,
        "recall_naive": correct_naive / n,
        "recall_plaid": correct_plaid / n,
        "ndcg5_naive":  ndcg5_naive,
        "ndcg10_naive": ndcg10_naive,
        "ndcg5_plaid":  ndcg5_plaid,
        "ndcg10_plaid": ndcg10_plaid,
        "ndcg5_naive_vals":  ndcg5_naive_vals,
        "ndcg10_naive_vals": ndcg10_naive_vals,
        "ndcg5_plaid_vals":  ndcg5_plaid_vals,
        "ndcg10_plaid_vals": ndcg10_plaid_vals,
    }

    # 데이터셋별 결과 저장
    from save_results import save_benchmark_results
    save_benchmark_results(model_name=model_name, **result)

    return result

#%%
# ── 전체 데이터셋 실행 ────────────────────────────────
all_results = []
for ds_name in DATASETS:
    result = run_benchmark(ds_name)
    if result is not None:
        all_results.append(result)

# ── 최종 집계 요약 ────────────────────────────────────
W = 96
print(f"\n\n{'='*W}")
print(f"{'FINAL SUMMARY':^{W}}")
print(f"{'='*W}")
print(f"{'Dataset':<40} {'N-R@1':>7} {'N-nDCG@5':>9} {'N-nDCG@10':>10} {'P-R@1':>7} {'P-nDCG@5':>9} {'P-nDCG@10':>10}")
print(f"{'-'*W}")
for r in all_results:
    slug = r["dataset"].split("/")[-1]
    print(
        f"{slug:<40}"
        f" {r['recall_naive']:>6.2%} {r['ndcg5_naive']:>9.4f} {r['ndcg10_naive']:>10.4f}"
        f" {r['recall_plaid']:>6.2%} {r['ndcg5_plaid']:>9.4f} {r['ndcg10_plaid']:>10.4f}"
    )
print(f"{'='*W}")

avg = lambda key: sum(r[key] for r in all_results) / len(all_results)
print(
    f"{'Average':<40}"
    f" {avg('recall_naive'):>6.2%} {avg('ndcg5_naive'):>9.4f} {avg('ndcg10_naive'):>10.4f}"
    f" {avg('recall_plaid'):>6.2%} {avg('ndcg5_plaid'):>9.4f} {avg('ndcg10_plaid'):>10.4f}"
)
print(f"{'='*W}")

# 전체 집계 차트 저장
from save_results import save_multi_dataset_summary
save_multi_dataset_summary(all_results)
