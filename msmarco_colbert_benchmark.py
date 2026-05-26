#%%
import importlib
import math
import pickle
import time
import json
import csv
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

# ── Config ────────────────────────────────────────────
MODEL_NAME        = "colbert-ir/colbertv2.0"
DEVICE            = "cuda:0"
PLAID_DEVICE      = "cuda:1"
CORPUS_SIZE       = None      # 전체 8.8M (None) 또는 정수로 서브셋 지정
NAIVE_CORPUS_LIMIT = None  # Naive는 이 크기까지만 실행 (full 8.8M은 ~20시간 소요)
TOP_K             = 1000      # R@100, R@1000 지원을 위해 1000으로 설정
PLAID_NBITS       = 2         # 논문과 동일한 2-bit 압축 (~19GB for 8.8M, cuda:1 적합)

CACHE_DIR   = Path("cache/msmarco")
RESULTS_DIR = Path("results/msmarco_colbert")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 모델 로드 ──────────────────────────────────────────
print("Loading ColBERT v2.0...")
from pylate import models as pylate_models
colbert = pylate_models.ColBERT(
    model_name_or_path=MODEL_NAME,
    document_length=180,
    query_length=32,
)
colbert = colbert.to(DEVICE)
print("  ColBERT v2.0 loaded.")

# ── 헬퍼 함수 ─────────────────────────────────────────
def compute_ndcg(ranked_indices, gt_set, k):
    for rank, idx in enumerate(ranked_indices[:k]):
        if idx in gt_set:
            return 1.0 / math.log2(rank + 2)
    return 0.0

def compute_mrr(ranked_indices, gt_set, k=10):
    for rank, idx in enumerate(ranked_indices[:k]):
        if idx in gt_set:
            return 1.0 / (rank + 1)
    return 0.0

def compute_recall_at_k(ranked_indices, gt_set, k):
    retrieved = set(ranked_indices[:k])
    return len(gt_set & retrieved) / len(gt_set)

def create_plaid_index(ps, device=PLAID_DEVICE, nbits=PLAID_NBITS):
    if not importlib.util.find_spec("fast_plaid"):
        raise ImportError("pip install --no-deps fast-plaid fastkmeans")
    import json as _json
    from fast_plaid import search

    index_dir = Path("index_msmarco")
    meta_path = index_dir / "metadata.json"
    nbits_path = index_dir / "nbits.json"

    # corpus 크기와 nbits 모두 일치할 때만 기존 index 재사용
    if meta_path.exists() and nbits_path.exists():
        meta  = _json.loads(meta_path.read_text())
        saved = _json.loads(nbits_path.read_text())
        if meta.get("num_documents") == len(ps) and saved.get("nbits") == nbits:
            print(f"  Loading existing PLAID index ({meta['num_documents']:,} docs, nbits={nbits})...")
            return search.FastPlaid(index=str(index_dir), device=device)
        else:
            print(f"  Index mismatch (docs {meta.get('num_documents')} vs {len(ps)}, "
                  f"nbits {saved.get('nbits')} vs {nbits}), rebuilding...")

    index = search.FastPlaid(index=str(index_dir), device=device)
    index.create(documents_embeddings=[d.cpu().to(torch.float32) for d in ps], nbits=nbits)
    nbits_path.write_text(_json.dumps({"nbits": nbits}))
    return index

def get_topk_plaid(qs, plaid_index, k=TOP_K, batch_size=2000):
    scores_list = []
    for i in range(0, len(qs), batch_size):
        qs_batch = torch.nn.utils.rnn.pad_sequence(
            qs[i : i + batch_size], batch_first=True, padding_value=0
        ).cpu().to(torch.float32)
        scores_list.append(
            plaid_index.search(queries_embeddings=qs_batch, top_k=k)
        )
    return [q for batch in scores_list for q in batch]

def score_multi_vector_naive(qs, ps, batch_size=256, device=DEVICE):
    """Exact MaxSim scoring (ColBERT/ColPali 공통 연산)."""
    scores_list = []
    for i in tqdm(range(0, len(qs), batch_size), desc="  query batch"):
        qs_batch = torch.nn.utils.rnn.pad_sequence(
            qs[i : i + batch_size], batch_first=True, padding_value=0
        ).to(device)
        scores_batch = []
        for j in range(0, len(ps), batch_size):
            ps_batch = torch.nn.utils.rnn.pad_sequence(
                ps[j : j + batch_size], batch_first=True, padding_value=0
            ).to(device)
            scores_batch.append(
                torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                .max(dim=3)[0].sum(dim=2)
            )
        scores_list.append(torch.cat(scores_batch, dim=1).cpu())
    return torch.cat(scores_list, dim=0)

#%%
# ── 데이터 로드 ───────────────────────────────────────
print("\nLoading MS MARCO corpus...")
corpus_raw = load_dataset("BeIR/msmarco", "corpus", split="corpus")
n_full = len(corpus_raw)
print(f"  Full corpus: {n_full:,} passages")

# qrels 로드 (dev set)
print("Loading qrels (dev)...")
qrels_raw = load_dataset("BeIR/msmarco-qrels", split="validation")
# {query_id: set of relevant corpus _id strings}
qrels_full: dict[str, set] = {}
for row in qrels_raw:
    if row["score"] > 0:
        qrels_full.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))

# queries 로드
print("Loading queries...")
queries_raw = load_dataset("BeIR/msmarco", "queries", split="queries")
query_id_to_text = {str(row["_id"]): row["text"] for row in queries_raw}

# ── corpus 서브셋: 관련 문서 먼저 포함 ───────────────
# 관련 문서가 subset에 없으면 해당 쿼리를 평가할 수 없으므로,
# 먼저 dev qrels에 등장하는 문서를 포함하고 나머지를 random으로 채움
print("\nBuilding corpus subset...")
relevant_pids = {pid for pids in qrels_full.values() for pid in pids}
corpus_ids_all = [row["_id"] for row in corpus_raw]
corpus_texts_all = [row["text"] for row in corpus_raw]

# 관련 문서를 우선 포함
relevant_indices = [i for i, pid in enumerate(corpus_ids_all) if pid in relevant_pids]
other_indices = [i for i, pid in enumerate(corpus_ids_all) if pid not in relevant_pids]

if CORPUS_SIZE is not None:
    n_fill = max(0, CORPUS_SIZE - len(relevant_indices))
    import random; random.seed(42)
    selected = relevant_indices + random.sample(other_indices, min(n_fill, len(other_indices)))
    selected.sort()
else:
    selected = list(range(n_full))

corpus_texts  = [corpus_texts_all[i] for i in selected]
corpus_pids   = [corpus_ids_all[i]   for i in selected]
pid_to_idx    = {pid: idx for idx, pid in enumerate(corpus_pids)}
print(f"  Corpus subset : {len(corpus_texts):,} passages")
print(f"  (relevant docs: {len(relevant_indices):,} / other: {len(selected)-len(relevant_indices):,})")

# 평가 가능한 쿼리 (GT 문서가 subset 안에 있는 것)
query_ids, query_texts, gt_sets = [], [], []
for qid, rel_pids in qrels_full.items():
    gt_idx_set = {pid_to_idx[pid] for pid in rel_pids if pid in pid_to_idx}
    if gt_idx_set and qid in query_id_to_text:
        query_ids.append(qid)
        query_texts.append(query_id_to_text[qid])
        gt_sets.append(gt_idx_set)

n = len(query_ids)
print(f"  Evaluable queries: {n:,}")

#%%
# ── 패시지 임베딩 (캐시) ─────────────────────────────
cache_path = CACHE_DIR / f"passage_embeddings_{len(corpus_texts)}.pkl"
t0 = time.perf_counter()

if cache_path.exists():
    print(f"\nLoading cached passage embeddings...")
    with open(cache_path, "rb") as f:
        passage_embeddings = pickle.load(f)
    if len(passage_embeddings) != len(corpus_texts):
        print(f"  Cache mismatch, recomputing...")
        cache_path.unlink()
        passage_embeddings = None
else:
    passage_embeddings = None

if passage_embeddings is None:
    print(f"\nGenerating passage embeddings ({len(corpus_texts):,} passages)...")
    passage_embeddings = colbert.encode(
        corpus_texts,
        batch_size=512,
        is_query=False,
        show_progress_bar=True,
        convert_to_tensor=False,
    )
    # CPU 텐서 리스트로 정규화
    passage_embeddings = [
        torch.tensor(e) if not isinstance(e, torch.Tensor) else e.cpu()
        for e in passage_embeddings
    ]
    with open(cache_path, "wb") as f:
        pickle.dump(passage_embeddings, f)
    print(f"  Saved: {cache_path}")

t_pass_emb = time.perf_counter() - t0
print(f"  Passage embedding time: {t_pass_emb:.1f}s")

#%%
# ── 쿼리 임베딩 ───────────────────────────────────────
print(f"\nGenerating query embeddings ({n:,} queries)...")
t0 = time.perf_counter()
query_embeddings_raw = colbert.encode(
    query_texts,
    batch_size=512,
    is_query=True,
    show_progress_bar=True,
    convert_to_tensor=False,
)
query_embeddings = [
    torch.tensor(e) if not isinstance(e, torch.Tensor) else e.cpu()
    for e in query_embeddings_raw
]
t_query_emb = time.perf_counter() - t0
print(f"  Query embedding time: {t_query_emb:.1f}s")

#%%
# ── [1] Naive: Exact MaxSim (NAIVE_CORPUS_LIMIT까지만) ──
naive_n_corpus = len(passage_embeddings) if NAIVE_CORPUS_LIMIT is None else min(len(passage_embeddings), NAIVE_CORPUS_LIMIT)
naive_ps = passage_embeddings[:naive_n_corpus]

# Naive는 GT가 naive_n_corpus 범위 안에 있는 쿼리만 평가
naive_query_mask = [bool(gt & set(range(naive_n_corpus))) for gt in gt_sets]
naive_qs   = [query_embeddings[i] for i, ok in enumerate(naive_query_mask) if ok]
naive_gts  = [gt_sets[i] for i, ok in enumerate(naive_query_mask) if ok]
naive_n    = len(naive_qs)

print(f"\n[1] Naive (Exact MaxSim) — corpus {naive_n_corpus:,} / queries {naive_n:,}...")
t0 = time.perf_counter()
scores = score_multi_vector_naive(naive_qs, naive_ps)
t_naive = time.perf_counter() - t0

naive_idx_map = [i for i, ok in enumerate(naive_query_mask) if ok]
naive_r1, naive_ndcg5, naive_ndcg10, naive_mrr10 = [], [], [], []
naive_r100, naive_r1000 = [], []
for i in range(naive_n):
    ranked = scores[i].argsort(descending=True).tolist()
    naive_r1.append(1.0 if ranked[0] in naive_gts[i] else 0.0)
    naive_ndcg5.append(compute_ndcg(ranked, naive_gts[i], 5))
    naive_ndcg10.append(compute_ndcg(ranked, naive_gts[i], 10))
    naive_mrr10.append(compute_mrr(ranked, naive_gts[i], 10))
    naive_r100.append(compute_recall_at_k(ranked, naive_gts[i], 100))
    naive_r1000.append(compute_recall_at_k(ranked, naive_gts[i], 1000))

recall1_naive  = sum(naive_r1) / naive_n
ndcg5_naive    = sum(naive_ndcg5) / naive_n
ndcg10_naive   = sum(naive_ndcg10) / naive_n
mrr10_naive    = sum(naive_mrr10) / naive_n
r100_naive     = sum(naive_r100) / naive_n
r1000_naive    = sum(naive_r1000) / naive_n

print(f"  Recall@1   : {recall1_naive:.4f}")
print(f"  Recall@100 : {r100_naive:.4f}")
print(f"  Recall@1000: {r1000_naive:.4f}")
print(f"  nDCG@5     : {ndcg5_naive:.4f}")
print(f"  nDCG@10    : {ndcg10_naive:.4f}")
print(f"  MRR@10     : {mrr10_naive:.4f}")
print(f"  Time       : {t_naive:.2f}s")

#%%
# ── [2] Fast PLAID ────────────────────────────────────
print("\n[2] Fast PLAID...")
t0 = time.perf_counter()
plaid_index = create_plaid_index(passage_embeddings, device=PLAID_DEVICE)
t_build = time.perf_counter() - t0
print(f"  Index build time: {t_build:.2f}s")

t0 = time.perf_counter()
plaid_results = get_topk_plaid(query_embeddings, plaid_index, k=TOP_K)
t_plaid = time.perf_counter() - t0

plaid_r1, plaid_ndcg5, plaid_ndcg10, plaid_mrr10 = [], [], [], []
plaid_r100, plaid_r1000 = [], []
for i in range(n):
    ranked = [r[0] for r in plaid_results[i]]
    plaid_r1.append(1.0 if ranked[0] in gt_sets[i] else 0.0)
    plaid_ndcg5.append(compute_ndcg(ranked, gt_sets[i], 5))
    plaid_ndcg10.append(compute_ndcg(ranked, gt_sets[i], 10))
    plaid_mrr10.append(compute_mrr(ranked, gt_sets[i], 10))
    plaid_r100.append(compute_recall_at_k(ranked, gt_sets[i], 100))
    plaid_r1000.append(compute_recall_at_k(ranked, gt_sets[i], 1000))

recall1_plaid  = sum(plaid_r1) / n
ndcg5_plaid    = sum(plaid_ndcg5) / n
ndcg10_plaid   = sum(plaid_ndcg10) / n
mrr10_plaid    = sum(plaid_mrr10) / n
r100_plaid     = sum(plaid_r100) / n
r1000_plaid    = sum(plaid_r1000) / n

print(f"  Recall@1   : {recall1_plaid:.4f}")
print(f"  Recall@100 : {r100_plaid:.4f}")
print(f"  Recall@1000: {r1000_plaid:.4f}")
print(f"  nDCG@5     : {ndcg5_plaid:.4f}")
print(f"  nDCG@10    : {ndcg10_plaid:.4f}")
print(f"  MRR@10     : {mrr10_plaid:.4f}")
print(f"  Search time      : {t_plaid:.2f}s")
print(f"  Build+search time: {t_build + t_plaid:.2f}s")

#%%
# ── 결과 요약 ─────────────────────────────────────────
W = 90
print(f"\n{'='*W}")
print(f"  MS MARCO Passage  |  Model: ColBERTv2.0  |  PLAID nbits={PLAID_NBITS}")
print(f"  Full corpus: {len(corpus_texts):,}  |  PLAID queries: {n:,}  |  Naive corpus: {naive_n_corpus:,} / queries: {naive_n:,}")
print(f"{'='*W}")
print(f"  {'Method':<30} {'R@1':>7} {'R@100':>7} {'R@1000':>8} {'nDCG@5':>8} {'nDCG@10':>9} {'MRR@10':>8} {'Search(s)':>10}")
print(f"  {'-'*W}")
naive_label = f"Naive (corpus={naive_n_corpus//1000}k)"
print(f"  {naive_label:<30} {recall1_naive:>6.4f} {r100_naive:>7.4f} {r1000_naive:>8.4f} {ndcg5_naive:>8.4f} {ndcg10_naive:>9.4f} {mrr10_naive:>8.4f} {t_naive:>10.2f}")
print(f"  {'Fast PLAID (search only)':<30} {recall1_plaid:>6.4f} {r100_plaid:>7.4f} {r1000_plaid:>8.4f} {ndcg5_plaid:>8.4f} {ndcg10_plaid:>9.4f} {mrr10_plaid:>8.4f} {t_plaid:>10.2f}")
print(f"  {'Fast PLAID (build+search)':<30} {recall1_plaid:>6.4f} {r100_plaid:>7.4f} {r1000_plaid:>8.4f} {ndcg5_plaid:>8.4f} {ndcg10_plaid:>9.4f} {mrr10_plaid:>8.4f} {t_build+t_plaid:>10.2f}")
if naive_n_corpus == len(corpus_texts):
    print(f"  Speedup (search): {t_naive/t_plaid:.1f}x faster with PLAID")
else:
    print(f"  Note: Naive ran on {naive_n_corpus:,} passages, PLAID on {len(corpus_texts):,} passages")
print(f"{'='*W}")

#%%
# ── 결과 저장 ─────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

summary = {
    "timestamp": ts,
    "model": MODEL_NAME,
    "plaid_nbits": PLAID_NBITS,
    "full_corpus_size": len(corpus_texts),
    "pass_emb_time_s": round(t_pass_emb, 2),
    "naive": {
        "corpus_size":   naive_n_corpus,
        "n_queries":     naive_n,
        "recall@1":      round(recall1_naive, 4),
        "recall@100":    round(r100_naive, 4),
        "recall@1000":   round(r1000_naive, 4),
        "ndcg@5":        round(ndcg5_naive, 4),
        "ndcg@10":       round(ndcg10_naive, 4),
        "mrr@10":        round(mrr10_naive, 4),
        "search_time_s": round(t_naive, 3),
    },
    "fast_plaid": {
        "corpus_size":        len(corpus_texts),
        "n_queries":          n,
        "recall@1":           round(recall1_plaid, 4),
        "recall@100":         round(r100_plaid, 4),
        "recall@1000":        round(r1000_plaid, 4),
        "ndcg@5":             round(ndcg5_plaid, 4),
        "ndcg@10":            round(ndcg10_plaid, 4),
        "mrr@10":             round(mrr10_plaid, 4),
        "index_build_time_s": round(t_build, 3),
        "search_time_s":      round(t_plaid, 3),
    },
}
json_path = RESULTS_DIR / f"{ts}_summary.json"
json_path.write_text(json.dumps(summary, indent=2))
print(f"[Saved] Summary JSON : {json_path}")

# Per-query CSV — naive_idx_map maps naive-list index → original query index
naive_by_orig = {}
for j, orig_i in enumerate(naive_idx_map):
    naive_by_orig[orig_i] = {
        "r@1": naive_r1[j], "r@100": naive_r100[j], "r@1000": naive_r1000[j],
        "ndcg@5": naive_ndcg5[j], "ndcg@10": naive_ndcg10[j], "mrr@10": naive_mrr10[j],
    }

csv_path = RESULTS_DIR / f"{ts}_per_query.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "query_idx", "query_id", "query_text", "n_relevant",
        "naive_r@1", "naive_r@100", "naive_r@1000", "naive_ndcg@5", "naive_ndcg@10", "naive_mrr@10",
        "plaid_r@1", "plaid_r@100", "plaid_r@1000", "plaid_ndcg@5", "plaid_ndcg@10", "plaid_mrr@10",
    ])
    writer.writeheader()
    for i in range(n):
        nr = naive_by_orig.get(i, {})
        writer.writerow({
            "query_idx":      i,
            "query_id":       query_ids[i],
            "query_text":     query_texts[i],
            "n_relevant":     len(gt_sets[i]),
            "naive_r@1":      round(nr.get("r@1", float("nan")), 4),
            "naive_r@100":    round(nr.get("r@100", float("nan")), 4),
            "naive_r@1000":   round(nr.get("r@1000", float("nan")), 4),
            "naive_ndcg@5":   round(nr.get("ndcg@5", float("nan")), 4),
            "naive_ndcg@10":  round(nr.get("ndcg@10", float("nan")), 4),
            "naive_mrr@10":   round(nr.get("mrr@10", float("nan")), 4),
            "plaid_r@1":      round(plaid_r1[i], 4),
            "plaid_r@100":    round(plaid_r100[i], 4),
            "plaid_r@1000":   round(plaid_r1000[i], 4),
            "plaid_ndcg@5":   round(plaid_ndcg5[i], 4),
            "plaid_ndcg@10":  round(plaid_ndcg10[i], 4),
            "plaid_mrr@10":   round(plaid_mrr10[i], 4),
        })
print(f"[Saved] Per-query CSV: {csv_path}")
