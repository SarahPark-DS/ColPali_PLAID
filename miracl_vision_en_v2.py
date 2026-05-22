#%%
import csv
import json
import math
import pickle
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from datasets import load_dataset
from fast_plaid import search
from tqdm import tqdm

# ── Config ────────────────────────────────────────────
DATASET_PATH = "nvidia/miracl-vision"
LANG         = "en"
MODEL_NAME   = "vidore/colpali-v1.3"
DEVICE       = "cuda:0"
PLAID_DEVICE = "cuda:1"

CACHE_DIR   = Path("cache")
RESULTS_DIR = Path("results/miracl_vision_en")
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 모델 로드 ──────────────────────────────────────────
model = ColPali.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE
).eval()
processor = ColPaliProcessor.from_pretrained(MODEL_NAME)

# ── nDCG 헬퍼 ─────────────────────────────────────────
def compute_ndcg_set(ranked_indices, gt_set, k):
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, idx in enumerate(ranked_indices[:k])
        if idx in gt_set
    )
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(gt_set), k)))
    return dcg / idcg if idcg > 0 else 0.0

#%%
# ── 데이터 로드 ───────────────────────────────────────
print(f"Loading MIRACL-VISION [{LANG}]...")
queries_ds = load_dataset(DATASET_PATH, f"queries-{LANG}", split="default")
corpus_ds  = load_dataset(DATASET_PATH, f"corpus-{LANG}",  split="default")
qrels_ds   = load_dataset(DATASET_PATH, f"qrels-{LANG}",   split="default")
images_ds  = load_dataset(DATASET_PATH, f"images-{LANG}",  split="default")

corpus_id_to_idx = {doc["_id"]: i for i, doc in enumerate(corpus_ds)}

qrels_dict: dict[str, set] = {}
for rel in qrels_ds:
    if rel["score"] > 0:
        qid = str(rel["query-id"])
        cid = str(rel["corpus-id"])
        qrels_dict.setdefault(qid, set()).add(corpus_id_to_idx[cid])

query_ids   = [q["_id"]  for q in queries_ds if q["_id"] in qrels_dict]
query_texts = [q["text"] for q in queries_ds if q["_id"] in qrels_dict]
gt_sets     = [qrels_dict[qid] for qid in query_ids]

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
# ── GPU 전처리 (Naive용) ──────────────────────────────
print("Moving embeddings to GPU (float16) for Naive...")
image_embeddings_gpu = [e.to(DEVICE).to(torch.float16) for e in image_embeddings]
query_embeddings_gpu  = [e.to(DEVICE).to(torch.float16) for e in query_embeddings]
torch.cuda.synchronize()
print(f"  GPU memory allocated : {torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB")
print(f"  image_embeddings_gpu : {len(image_embeddings_gpu)} tensors, each {tuple(image_embeddings_gpu[0].shape)}, dtype={image_embeddings_gpu[0].dtype}, device={image_embeddings_gpu[0].device}")
print(f"  query_embeddings_gpu : {len(query_embeddings_gpu)} tensors, each {tuple(query_embeddings_gpu[0].shape)}, dtype={query_embeddings_gpu[0].dtype}, device={query_embeddings_gpu[0].device}")

# ── PLAID 인덱스 빌드 (캐시) ──────────────────────────
PLAID_INDEX_DIR = Path(f"cache/plaid_index_{LANG}").resolve()  # 절대경로로 경로 문제 방지
print("\n[2] Fast PLAID — Index Build...")
torch.cuda.synchronize()
t0 = time.perf_counter()

plaid_index = search.FastPlaid(index=str(PLAID_INDEX_DIR), device=PLAID_DEVICE, low_memory=False)

# 파일 존재 여부가 아닌 실제 인덱스 로드 성공 여부로 판단
# (_reload_index 실패 시 indices[device]=None으로 조용히 실패하는 경우 대비)
index_loaded = any(v is not None for v in plaid_index.indices.values())

if index_loaded:
    print(f"  Loading cached PLAID index from {PLAID_INDEX_DIR}")
    t_build = 0.0
else:
    print(f"  Building PLAID index → {PLAID_INDEX_DIR}")
    plaid_index.create(
        documents_embeddings=image_embeddings,
        nbits=4,
        kmeans_niters=4,
    )
    torch.cuda.synchronize()
    t_build = time.perf_counter() - t0
    print(f"  Index build time : {t_build:.2f}s")
    print(f"  Saved: {PLAID_INDEX_DIR}")

# ── Warmup ────────────────────────────────────────────
print("  Warming up...")
_ = processor.score_multi_vector(query_embeddings_gpu[:2], image_embeddings_gpu[:2])
_ = plaid_index.search(queries_embeddings=query_embeddings[:2], top_k=10, show_progress=False)
torch.cuda.synchronize()

#%%
# ── [1] Naive: score_multi_vector ────────────────────
print("\n[1] Naive (score_multi_vector)...")
print(f"  Input  queries : {len(query_embeddings_gpu)} x {tuple(query_embeddings_gpu[0].shape)}  ({query_embeddings_gpu[0].dtype}, {query_embeddings_gpu[0].device})")
print(f"  Input  docs    : {len(image_embeddings_gpu)} x {tuple(image_embeddings_gpu[0].shape)}  ({image_embeddings_gpu[0].dtype}, {image_embeddings_gpu[0].device})")
torch.cuda.synchronize()
t0 = time.perf_counter()
scores = processor.score_multi_vector(query_embeddings_gpu, image_embeddings_gpu)
print(f"  Output scores  : {tuple(scores.shape)}  ({scores.dtype}, {scores.device})")
torch.cuda.synchronize()
t_naive = time.perf_counter() - t0

correct_naive = sum(1 for i in range(n) if scores[i].argmax().item() in gt_sets[i])
recall_naive  = correct_naive / n
ndcg5_naive_vals  = [compute_ndcg_set(scores[i].argsort(descending=True).tolist(), gt_sets[i], 5)  for i in range(n)]
ndcg10_naive_vals = [compute_ndcg_set(scores[i].argsort(descending=True).tolist(), gt_sets[i], 10) for i in range(n)]
ndcg5_naive  = sum(ndcg5_naive_vals)  / n
ndcg10_naive = sum(ndcg10_naive_vals) / n
qps_naive    = n / t_naive

print(f"  Recall@1 : {recall_naive:.2%} ({correct_naive}/{n})")
print(f"  nDCG@5   : {ndcg5_naive:.4f}")
print(f"  nDCG@10  : {ndcg10_naive:.4f}")
print(f"  Time     : {t_naive:.2f}s  |  QPS: {qps_naive:.1f}")

#%%
# ── [2] Fast PLAID — Search ───────────────────────────
# fast-plaid 공식 방식: 쿼리 리스트를 그대로 전달, 배치/패딩 불필요
print("\n[2] Fast PLAID — Search...")
print(f"  Input  queries : {len(query_embeddings)} x {tuple(query_embeddings[0].shape)}  ({query_embeddings[0].dtype}, {query_embeddings[0].device})")
print(f"  Index  device  : {PLAID_DEVICE},  low_memory=False,  n_ivf_probe=1,  n_full_scores=256")
torch.cuda.synchronize()
t0 = time.perf_counter()

all_plaid_results = plaid_index.search(
    queries_embeddings=query_embeddings,  # list[torch.Tensor], CPU bfloat16
    top_k=10,
    n_ivf_probe=1,
    n_full_scores=256,
    show_progress=False,
)

torch.cuda.synchronize()
t_plaid = time.perf_counter() - t0

correct_plaid = sum(1 for i in range(n) if all_plaid_results[i][0][0] in gt_sets[i])
recall_plaid  = correct_plaid / n
ndcg5_plaid_vals  = [compute_ndcg_set([r[0] for r in all_plaid_results[i]], gt_sets[i], 5)  for i in range(n)]
ndcg10_plaid_vals = [compute_ndcg_set([r[0] for r in all_plaid_results[i]], gt_sets[i], 10) for i in range(n)]
ndcg5_plaid  = sum(ndcg5_plaid_vals)  / n
ndcg10_plaid = sum(ndcg10_plaid_vals) / n
qps_plaid    = n / t_plaid

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
json_path = RESULTS_DIR / f"{ts}_v2_summary.json"
json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"[Saved] Summary JSON : {json_path}")

# CSV per-query
csv_path = RESULTS_DIR / f"{ts}_v2_per_query.csv"
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
png_path = RESULTS_DIR / f"{ts}_v2_comparison.png"
labels  = ["Naive", "PLAID\n(search only)", "PLAID\n(build+search)"]
t_emb_v = [t_img_emb, t_img_emb, t_img_emb]
t_bld_v = [0,         0,          t_build]
t_sch_v = [t_naive,   t_plaid,    t_plaid]
totals  = [e + b + s for e, b, s in zip(t_emb_v, t_bld_v, t_sch_v)]

fig, (ax_r, ax_t) = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(f"MIRACL-VISION [{LANG.upper()}]  |  Corpus {len(corpus_ds):,} images",
             fontsize=13, fontweight="bold")

recall_bars = ax_r.bar(["Naive", "Fast PLAID"], [recall_naive, recall_plaid],
                        color=["#4C72B0", "#DD8452"], width=0.4, edgecolor="white")
ax_r.set_ylim(0, 1.0)
ax_r.set_ylabel("Recall@1")
ax_r.set_title("Recall@1 Comparison")
for bar, val in zip(recall_bars, [recall_naive, recall_plaid]):
    ax_r.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2%}",
              ha="center", va="bottom", fontsize=10, fontweight="bold")

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
