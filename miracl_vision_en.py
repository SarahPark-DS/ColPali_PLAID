#%%
import csv
import importlib
import json
import math
import pickle
import time
from datetime import datetime
from pathlib import Path
from fast_plaid import search

import matplotlib.pyplot as plt
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from datasets import load_dataset
from tqdm import tqdm

# ── Config ────────────────────────────────────────────
DATASET_PATH = "nvidia/miracl-vision"
LANG         = "en"
MODEL_NAME   = "vidore/colpali-v1.3"
DEVICE       = "cuda:0"
PLAID_DEVICE = "cuda:1"  # PLAID index/search는 cuda:1 — cuda:0 OOM 방지

CACHE_DIR   = Path("cache")
RESULTS_DIR = Path("results/miracl_vision_en")
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 모델 로드 ──────────────────────────────────────────
model = ColPali.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
).eval()
processor = ColPaliProcessor.from_pretrained(MODEL_NAME)

# ── nDCG 헬퍼 ────────────────────────────────────────
def compute_ndcg_set(ranked_indices, gt_set, k):
    """nDCG@k for a query with potentially multiple relevant documents."""
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, idx in enumerate(ranked_indices[:k])
        if idx in gt_set
    )
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(gt_set), k)))
    return dcg / idcg if idcg > 0 else 0.0

# ── PLAID 헬퍼 ────────────────────────────────────────
def create_plaid_index(ps, device=DEVICE):
    if not importlib.util.find_spec("fast_plaid"):
        raise ImportError("pip install --no-deps fast-plaid fastkmeans")

    index_dir = Path("index")
    meta_path = index_dir / "metadata.json"

    # 기존 index가 있고 corpus 크기가 일치하면 재빌드 없이 로드
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("num_documents") == len(ps):
            print(f"  Loading existing PLAID index ({meta['num_documents']:,} docs)...")
            return search.FastPlaid(index=str(index_dir), device=device)
        else:
            print(f"  Index size mismatch ({meta.get('num_documents')} vs {len(ps)}), rebuilding...")

    torch.cuda.empty_cache()
    try:
        index = search.FastPlaid(index=str(index_dir), device=device, low_memory=False)
        index.create(documents_embeddings=[d.to(device).to(torch.float16) for d in ps])
    except torch.cuda.OutOfMemoryError:
        print(f"  WARNING: {device} OOM during index build — retrying on CPU...")
        torch.cuda.empty_cache()
        index = search.FastPlaid(index=str(index_dir))
        index.create(documents_embeddings=[d.cpu().to(torch.float16) for d in ps])
    return index

def get_topk_plaid(qs, plaid_index, k=10, batch_size=128, device=DEVICE, profile=False):
    scores_list = []
    t_pad = t_transfer = t_search = 0.0
    for i in range(0, len(qs), batch_size):
        if profile:
            torch.cuda.synchronize()
            _t = time.perf_counter()
        qs_batch = torch.nn.utils.rnn.pad_sequence(
            qs[i : i + batch_size], batch_first=True, padding_value=0
        )
        if profile:
            torch.cuda.synchronize(); t_pad += time.perf_counter() - _t; _t = time.perf_counter()
        qs_batch = qs_batch.to(device).to(torch.float16)
        if profile:
            torch.cuda.synchronize(); t_transfer += time.perf_counter() - _t; _t = time.perf_counter()
        scores_list.append(plaid_index.search(queries_embeddings=qs_batch, top_k=k, n_ivf_probe=8, n_full_scores=256, show_progress=False))
        if profile:
            torch.cuda.synchronize(); t_search += time.perf_counter() - _t
    if profile:
        n_batches = math.ceil(len(qs) / batch_size)
        print(f"  [PLAID profile] batches={n_batches}  pad={t_pad:.3f}s  transfer={t_transfer:.3f}s  search={t_search:.3f}s")
    return scores_list

#%%
# ── 데이터 로드 ───────────────────────────────────────
print(f"Loading MIRACL-VISION [{LANG}]...")
queries_ds = load_dataset(DATASET_PATH, f"queries-{LANG}", split="default")
corpus_ds  = load_dataset(DATASET_PATH, f"corpus-{LANG}",  split="default")
qrels_ds   = load_dataset(DATASET_PATH, f"qrels-{LANG}",   split="default")
images_ds  = load_dataset(DATASET_PATH, f"images-{LANG}",  split="default")

# corpus _id → corpus index (= image index, 두 데이터셋이 정렬돼 있음)
corpus_id_to_idx = {doc["_id"]: i for i, doc in enumerate(corpus_ds)}

# qrels: {query_id: set of relevant corpus indices}
qrels_dict: dict[str, set] = {}
for rel in qrels_ds:
    if rel["score"] > 0:
        qid = str(rel["query-id"])
        cid = str(rel["corpus-id"])
        qrels_dict.setdefault(qid, set()).add(corpus_id_to_idx[cid])

# qrels가 있는 쿼리만 사용
query_ids    = [q["_id"] for q in queries_ds if q["_id"] in qrels_dict]
query_texts  = [q["text"] for q in queries_ds if q["_id"] in qrels_dict]
gt_sets      = [qrels_dict[qid] for qid in query_ids]

print(f"  Corpus size   : {len(corpus_ds):,} images")
print(f"  Queries total : {len(queries_ds):,}")
print(f"  Queries w/ GT : {len(query_ids):,}")

#%%
# ── 이미지 임베딩 (캐시) ──────────────────────────────
cache_path = CACHE_DIR / f"miracl_vision_{LANG}_image_embeddings.pkl"
t0 = time.perf_counter()

if cache_path.exists():
    print("Loading cached image embeddings...")
    with open(cache_path, "rb") as f:
        image_embeddings = pickle.load(f)
    if len(image_embeddings) != len(corpus_ds):
        print(f"  Cache mismatch ({len(image_embeddings)} vs {len(corpus_ds)}), recomputing...")
        cache_path.unlink()
        image_embeddings = None
else:
    image_embeddings = None

if image_embeddings is None:
    print(f"Generating image embeddings ({len(corpus_ds):,} images)...")
    image_embeddings = []
    for item in tqdm(images_ds, desc="Image Embedding"):
        batch = processor.process_images([item["image"].convert("RGB")]).to(DEVICE)
        with torch.no_grad():
            emb = model(**batch)
        image_embeddings.append(emb[0].cpu())
    with open(cache_path, "wb") as f:
        pickle.dump(image_embeddings, f)
    print(f"  Saved: {cache_path}")

t_img_emb = time.perf_counter() - t0
print(f"  Image embedding time: {t_img_emb:.1f}s")

#%%
# ── 쿼리 임베딩 ───────────────────────────────────────
print("Generating query embeddings...")
query_embeddings = []
for q in tqdm(query_texts, desc="Query Embedding"):
    batch = processor.process_queries([q]).to(DEVICE)
    with torch.no_grad():
        emb = model(**batch)
    query_embeddings.append(emb[0].cpu())

n = len(query_ids)

# 임베딩 생성 완료 — 모델 해제해서 VRAM 확보
del model
torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f"  GPU memory after model release: {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB")

#%%
# ── 공통 전처리: GPU로 올리기 ──────────────────────────
print("Moving embeddings to GPU (float16)...")
image_embeddings_gpu = [e.to(DEVICE).to(torch.float16) for e in image_embeddings]
query_embeddings_gpu  = [e.to(DEVICE).to(torch.float16) for e in query_embeddings]
torch.cuda.synchronize()
print(f"  GPU memory allocated : {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB")

# ── PLAID 인덱스 빌드 (타이밍 측정) ──────────────────
print("\n[2] Fast PLAID — Index Build...")
torch.cuda.synchronize()
t0 = time.perf_counter()
plaid_index = create_plaid_index(image_embeddings, device=PLAID_DEVICE)
torch.cuda.synchronize()
t_build = time.perf_counter() - t0
print(f"  Index build time : {t_build:.2f}s")

# ── Warmup (cold start 제외) ──────────────────────────
print("  Warming up...")
_ = processor.score_multi_vector(query_embeddings_gpu[:2], image_embeddings_gpu[:2])
_ = get_topk_plaid(query_embeddings_gpu[:2], plaid_index, k=10, device=PLAID_DEVICE)
torch.cuda.synchronize()

#%%
# ── [1] Naive: score_multi_vector ────────────────────
print("\n[1] Naive (score_multi_vector)...")
torch.cuda.synchronize()
t0 = time.perf_counter()
scores = processor.score_multi_vector(query_embeddings_gpu, image_embeddings_gpu)
torch.cuda.synchronize()
t_naive = time.perf_counter() - t0

# Recall@1: top-1 예측이 relevant set에 포함되면 정답
correct_naive = sum(1 for i in range(n) if scores[i].argmax().item() in gt_sets[i])
recall_naive  = correct_naive / n

ndcg5_naive_vals  = [compute_ndcg_set(scores[i].argsort(descending=True).tolist(), gt_sets[i], 5)  for i in range(n)]
ndcg10_naive_vals = [compute_ndcg_set(scores[i].argsort(descending=True).tolist(), gt_sets[i], 10) for i in range(n)]
ndcg5_naive  = sum(ndcg5_naive_vals)  / n
ndcg10_naive = sum(ndcg10_naive_vals) / n

qps_naive = n / t_naive

print(f"  Recall@1 : {recall_naive:.2%} ({correct_naive}/{n})")
print(f"  nDCG@5   : {ndcg5_naive:.4f}")
print(f"  nDCG@10  : {ndcg10_naive:.4f}")
print(f"  Time     : {t_naive:.2f}s  |  QPS: {qps_naive:.1f}")

#%%
# ── [2] Fast PLAID — Search ───────────────────────────
print("\n[2] Fast PLAID — Search...")
torch.cuda.synchronize()
t0 = time.perf_counter()
plaid_results_batched = get_topk_plaid(query_embeddings_gpu, plaid_index, k=10, device=PLAID_DEVICE, profile=True)
torch.cuda.synchronize()
t_plaid = time.perf_counter() - t0

all_plaid_results = [q for batch in plaid_results_batched for q in batch]
correct_plaid = sum(1 for i in range(n) if all_plaid_results[i][0][0] in gt_sets[i])
recall_plaid  = correct_plaid / n

ndcg5_plaid_vals  = [compute_ndcg_set([r[0] for r in all_plaid_results[i]], gt_sets[i], 5)  for i in range(n)]
ndcg10_plaid_vals = [compute_ndcg_set([r[0] for r in all_plaid_results[i]], gt_sets[i], 10) for i in range(n)]
ndcg5_plaid  = sum(ndcg5_plaid_vals)  / n
ndcg10_plaid = sum(ndcg10_plaid_vals) / n

qps_plaid = n / t_plaid

print(f"  Recall@1 : {recall_plaid:.2%} ({correct_plaid}/{n})")
print(f"  nDCG@5   : {ndcg5_plaid:.4f}")
print(f"  nDCG@10  : {ndcg10_plaid:.4f}")
print(f"  Search time      : {t_plaid:.2f}s  |  QPS: {qps_plaid:.1f}")
print(f"  Build+search time: {t_build + t_plaid:.2f}s")

#%%
# ── 결과 요약 출력 ────────────────────────────────────
W = 99
print(f"\n{'='*W}")
print(f"  MIRACL-VISION [{LANG.upper()}]  |  Corpus: {len(corpus_ds):,}  |  Queries: {n}")
print(f"{'='*W}")
print(f"  {'Method':<26} {'R@1':>7} {'nDCG@5':>8} {'nDCG@10':>9} {'Img Emb(s)':>11} {'Search(s)':>10} {'Total(s)':>9} {'QPS':>8}")
print(f"  {'-'*W}")
print(f"  {'Naive (score_multi_vector)':<26} {recall_naive:>6.2%} {ndcg5_naive:>8.4f} {ndcg10_naive:>9.4f} {t_img_emb:>11.1f} {t_naive:>10.2f} {t_img_emb+t_naive:>9.1f} {qps_naive:>8.1f}")
print(f"  {'Fast PLAID (search only)':<26} {recall_plaid:>6.2%} {ndcg5_plaid:>8.4f} {ndcg10_plaid:>9.4f} {t_img_emb:>11.1f} {t_plaid:>10.2f} {t_img_emb+t_plaid:>9.1f} {qps_plaid:>8.1f}")
print(f"  {'Fast PLAID (build+search)':<26} {recall_plaid:>6.2%} {ndcg5_plaid:>8.4f} {ndcg10_plaid:>9.4f} {t_img_emb:>11.1f} {t_build+t_plaid:>10.2f} {t_img_emb+t_build+t_plaid:>9.1f} {qps_plaid:>8.1f}")
print(f"{'='*W}")

#%%
# ── 결과 저장 ─────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# JSON summary
summary = {
    "timestamp": ts,
    "dataset": DATASET_PATH,
    "language": LANG,
    "model": MODEL_NAME,
    "n_corpus": len(corpus_ds),
    "n_queries": n,
    "img_emb_time_s": round(t_img_emb, 2),
    "naive": {
        "recall@1":      round(recall_naive, 4),
        "ndcg@5":        round(ndcg5_naive, 4),
        "ndcg@10":       round(ndcg10_naive, 4),
        "search_time_s": round(t_naive, 3),
        "qps":           round(qps_naive, 2),
        "total_time_s":  round(t_img_emb + t_naive, 2),
    },
    "fast_plaid": {
        "recall@1":           round(recall_plaid, 4),
        "ndcg@5":             round(ndcg5_plaid, 4),
        "ndcg@10":            round(ndcg10_plaid, 4),
        "index_build_time_s": round(t_build, 3),
        "search_time_s":      round(t_plaid, 3),
        "qps":                round(qps_plaid, 2),
        "total_time_s":       round(t_img_emb + t_build + t_plaid, 2),
    },
}
json_path = RESULTS_DIR / f"{ts}_summary.json"
json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"[Saved] Summary JSON : {json_path}")

# CSV per-query
csv_path = RESULTS_DIR / f"{ts}_per_query.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "query_idx", "query_id", "query_text",
        "gt_corpus_indices", "n_relevant",
        "naive_pred", "naive_score", "naive_correct", "naive_ndcg@5", "naive_ndcg@10",
        "plaid_pred", "plaid_score", "plaid_correct", "plaid_ndcg@5", "plaid_ndcg@10",
    ])
    writer.writeheader()
    for i in range(n):
        naive_pred  = scores[i].argmax().item()
        plaid_pred  = all_plaid_results[i][0][0]
        plaid_score = all_plaid_results[i][0][1]
        writer.writerow({
            "query_idx":         i,
            "query_id":          query_ids[i],
            "query_text":        query_texts[i],
            "gt_corpus_indices": sorted(gt_sets[i]),
            "n_relevant":        len(gt_sets[i]),
            "naive_pred":        naive_pred,
            "naive_score":       round(scores[i][naive_pred].item(), 4),
            "naive_correct":     naive_pred in gt_sets[i],
            "naive_ndcg@5":      round(ndcg5_naive_vals[i], 4),
            "naive_ndcg@10":     round(ndcg10_naive_vals[i], 4),
            "plaid_pred":        plaid_pred,
            "plaid_score":       round(plaid_score, 4),
            "plaid_correct":     plaid_pred in gt_sets[i],
            "plaid_ndcg@5":      round(ndcg5_plaid_vals[i], 4),
            "plaid_ndcg@10":     round(ndcg10_plaid_vals[i], 4),
        })
print(f"[Saved] Per-query CSV: {csv_path}")

# Comparison chart
png_path = RESULTS_DIR / f"{ts}_comparison.png"
labels   = ["Naive", "PLAID\n(search only)", "PLAID\n(build+search)"]
t_emb_v  = [t_img_emb, t_img_emb,  t_img_emb]
t_bld_v  = [0,         0,           t_build]
t_sch_v  = [t_naive,   t_plaid,     t_plaid]
totals   = [e + b + s for e, b, s in zip(t_emb_v, t_bld_v, t_sch_v)]

fig, (ax_r, ax_t) = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(f"MIRACL-VISION [{LANG.upper()}]  |  Corpus {len(corpus_ds):,} images",
             fontsize=13, fontweight="bold")

# Recall@1
recall_bars = ax_r.bar(["Naive", "Fast PLAID"], [recall_naive, recall_plaid],
                        color=["#4C72B0", "#DD8452"], width=0.4, edgecolor="white")
ax_r.set_ylim(0, 1.0)
ax_r.set_ylabel("Recall@1")
ax_r.set_title("Recall@1 Comparison")
for bar, val in zip(recall_bars, [recall_naive, recall_plaid]):
    ax_r.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2%}",
              ha="center", va="bottom", fontsize=10, fontweight="bold")

# Latency stacked
x = range(len(labels))
ax_t.bar(x, t_emb_v, 0.45, label="Img Embedding", color="#4C72B0")
ax_t.bar(x, t_bld_v, 0.45, label="Index Build",   color="#C44E52", bottom=t_emb_v)
ax_t.bar(x, t_sch_v, 0.45, label="Search",        color="#55A868",
         bottom=[e + b for e, b in zip(t_emb_v, t_bld_v)])
ax_t.set_xticks(list(x))
ax_t.set_xticklabels(labels)
ax_t.set_ylabel("Time (s)")
ax_t.set_title("Latency Breakdown")
ax_t.legend(fontsize=8)
for i, total in enumerate(totals):
    ax_t.text(i, total + max(totals) * 0.01, f"{total:.0f}s",
              ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Saved] Chart PNG    : {png_path}")
