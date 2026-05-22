#%%
"""
fast-plaid 공식 벤치마크 조건 재현 + ColPali 조건 비교

[비교 목적]
  텍스트 검색 모델 : Naive vs Fast-PLAID QPS 변화
  이미지 검색 모델 : Naive vs Fast-PLAID QPS 변화
  → 토큰 길이(180 vs 1030)가 PLAID 효과에 미치는 영향 정량화

fast-plaid 공식 벤치마크 (H100, FIQA 57K docs):
  - 모델     : ColBERT (text), 문서 ~180토큰, 쿼리 ~32토큰
  - QPS      : ~146 (n_ivf_probe=8, n_full_scores=4096)
"""

import shutil
import time
from pathlib import Path

import torch
from fast_plaid import search

DEVICE = "cuda:1"
RESULTS = []


def naive_maxsim(query_embs_gpu: list, doc_embs_gpu: list) -> torch.Tensor:
    """GPU MaxSim: score(q,d) = Σ_i max_j (q_i · d_j)"""
    D = torch.stack(doc_embs_gpu)           # [n_docs, d_tok, dim]
    n_docs, d_tok, dim = D.shape
    D_flat = D.reshape(n_docs * d_tok, dim) # [n_docs*d_tok, dim]

    # 쿼리 배치 크기: sim 중간 텐서가 4GB 이하가 되도록 자동 조정
    q_tok = query_embs_gpu[0].shape[0]
    bytes_per_q = n_docs * d_tok * q_tok * 2
    q_batch = max(1, int(4e9 / bytes_per_q))

    all_scores = []
    for i in range(0, len(query_embs_gpu), q_batch):
        Q = torch.stack(query_embs_gpu[i : i + q_batch])   # [b, q_tok, dim]
        b = Q.shape[0]
        sim = (Q.reshape(b * q_tok, dim) @ D_flat.T)       # [b*q_tok, n_docs*d_tok]
        sim = sim.view(b, q_tok, n_docs, d_tok)
        score = sim.max(dim=-1).values.sum(dim=1)           # [b, n_docs]
        all_scores.append(score)
        del sim
    return torch.cat(all_scores, dim=0)                    # [n_queries, n_docs]


def run_bench(
    tag: str,
    n_docs: int,
    n_queries: int,
    doc_tokens: int,
    query_tokens: int,
    dim: int = 128,
    n_ivf_probe: int = 8,
    n_full_scores: int = 4096,
    index_dir: str = "bench_index",
    n_warmup: int = 3,
    n_repeat: int = 5,
):
    print(f"\n{'='*64}")
    print(f"  {tag}")
    print(f"  docs={n_docs:,}  queries={n_queries}  "
          f"doc_tok={doc_tokens}  q_tok={query_tokens}")
    print(f"  n_ivf_probe={n_ivf_probe}  n_full_scores={n_full_scores}")
    print(f"{'='*64}")

    # ── 합성 임베딩 생성 (L2 정규화) ─────────────────────────
    print("  Generating synthetic embeddings...")
    doc_embs  = [torch.nn.functional.normalize(torch.randn(doc_tokens,   dim), dim=-1).to(torch.float16) for _ in range(n_docs)]
    query_embs = [torch.nn.functional.normalize(torch.randn(query_tokens, dim), dim=-1).to(torch.float16) for _ in range(n_queries)]

    # GPU로 올리기
    doc_embs_gpu   = [e.to(DEVICE) for e in doc_embs]
    query_embs_gpu = [e.to(DEVICE) for e in query_embs]
    torch.cuda.synchronize()

    # ── [Naive] MaxSim ────────────────────────────────────────
    print("  [Naive] Warming up...")
    _ = naive_maxsim(query_embs_gpu[:4], doc_embs_gpu)
    torch.cuda.synchronize()

    print(f"  [Naive] Measuring ({n_repeat} runs)...")
    naive_times = []
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = naive_maxsim(query_embs_gpu, doc_embs_gpu)
        torch.cuda.synchronize()
        naive_times.append(time.perf_counter() - t0)

    t_naive_avg = sum(naive_times) / len(naive_times)
    t_naive_min = min(naive_times)
    qps_naive_avg = n_queries / t_naive_avg
    qps_naive_max = n_queries / t_naive_min
    print(f"  [Naive] avg={t_naive_avg:.3f}s  QPS_avg={qps_naive_avg:.1f}  QPS_max={qps_naive_max:.1f}")

    # GPU 임베딩 해제 후 PLAID용 메모리 확보
    del doc_embs_gpu, query_embs_gpu
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ── [PLAID] 인덱스 빌드 ───────────────────────────────────
    index_path = Path(index_dir).resolve()
    if index_path.exists():
        shutil.rmtree(index_path)

    print("  [PLAID] Building index...")
    t0 = time.perf_counter()
    plaid_index = search.FastPlaid(index=str(index_path), device=DEVICE, low_memory=False)
    plaid_index.create(documents_embeddings=doc_embs, nbits=4, kmeans_niters=4)
    torch.cuda.synchronize()
    t_build = time.perf_counter() - t0
    print(f"  [PLAID] Build time: {t_build:.2f}s")

    print(f"  [PLAID] Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        _ = plaid_index.search(queries_embeddings=query_embs[:10], top_k=10,
                               n_ivf_probe=n_ivf_probe, n_full_scores=n_full_scores,
                               show_progress=False)
    torch.cuda.synchronize()

    print(f"  [PLAID] Measuring ({n_repeat} runs)...")
    plaid_times = []
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = plaid_index.search(queries_embeddings=query_embs, top_k=10,
                               n_ivf_probe=n_ivf_probe, n_full_scores=n_full_scores,
                               show_progress=False)
        torch.cuda.synchronize()
        plaid_times.append(time.perf_counter() - t0)

    t_plaid_avg = sum(plaid_times) / len(plaid_times)
    t_plaid_min = min(plaid_times)
    qps_plaid_avg = n_queries / t_plaid_avg
    qps_plaid_max = n_queries / t_plaid_min
    print(f"  [PLAID] avg={t_plaid_avg:.3f}s  QPS_avg={qps_plaid_avg:.1f}  QPS_max={qps_plaid_max:.1f}")
    print(f"  [PLAID/Naive QPS ratio] {qps_plaid_avg/qps_naive_avg:.2f}x")

    shutil.rmtree(index_path)
    torch.cuda.empty_cache()

    RESULTS.append({
        "tag":            tag,
        "n_docs":         n_docs,
        "doc_tokens":     doc_tokens,
        "query_tokens":   query_tokens,
        "t_build_s":      round(t_build, 1),
        "qps_naive":      round(qps_naive_avg, 1),
        "qps_plaid":      round(qps_plaid_avg, 1),
        "plaid_vs_naive": round(qps_plaid_avg / qps_naive_avg, 2),
    })


#%%
# ── [A] 텍스트 ColBERT — 공식 벤치마크 조건 ─────────────────
run_bench(
    tag="[A] Text ColBERT  (57K docs, 180 tok) — FIQA scale",
    n_docs=57_638, n_queries=648,
    doc_tokens=180, query_tokens=32,
    n_ivf_probe=8, n_full_scores=4096,
    index_dir="bench_index_A",
)

#%%
# ── [B] 텍스트 ColBERT — MIRACL-Vision 규모 ─────────────────
run_bench(
    tag="[B] Text ColBERT  (42K docs, 180 tok) — MIRACL scale",
    n_docs=42_971, n_queries=447,
    doc_tokens=180, query_tokens=32,
    n_ivf_probe=8, n_full_scores=4096,
    index_dir="bench_index_B",
)

#%%
# ── [C] ColPali 토큰 길이 — 기본 파라미터 ────────────────────
run_bench(
    tag="[C] ColPali tokens (42K docs, 1030 tok) — default params",
    n_docs=42_971, n_queries=447,
    doc_tokens=1030, query_tokens=24,
    n_ivf_probe=8, n_full_scores=4096,
    index_dir="bench_index_C",
)

#%%
# ── [D] ColPali 토큰 길이 — 축소 파라미터 ────────────────────
run_bench(
    tag="[D] ColPali tokens (42K docs, 1030 tok) — fast params",
    n_docs=42_971, n_queries=447,
    doc_tokens=1030, query_tokens=24,
    n_ivf_probe=1, n_full_scores=256,
    index_dir="bench_index_D",
)

#%%
# ── 결과 요약 ─────────────────────────────────────────────────
W = 95
print(f"\n{'='*W}")
print(f"  Naive vs Fast-PLAID QPS  |  device={DEVICE}  |  synthetic fp16 embeddings")
print(f"{'='*W}")
print(f"  {'Tag':<50} {'n_docs':>7} {'d_tok':>5} {'Build(s)':>9} "
      f"{'Naive QPS':>10} {'PLAID QPS':>10} {'ratio':>7}")
print(f"  {'-'*W}")
for r in RESULTS:
    ratio_str = f"{r['plaid_vs_naive']:.2f}x"
    arrow = "↑" if r['plaid_vs_naive'] >= 1 else "↓"
    print(f"  {r['tag']:<50} {r['n_docs']:>7,} {r['doc_tokens']:>5} {r['t_build_s']:>9.1f} "
          f"{r['qps_naive']:>10.1f} {r['qps_plaid']:>10.1f} {arrow}{ratio_str:>6}")
print(f"{'='*W}")
print("""
비교 포인트:
  [A][B] ratio > 1.0 → 텍스트 ColBERT에서 PLAID가 Naive보다 빠름
  [C][D] ratio < 1.0 → ColPali 1030토큰에서 PLAID가 Naive보다 느림
  [B]→[C] Naive 변화 : 토큰 길이가 Naive 속도에 미치는 영향
  [B]→[C] PLAID 변화 : 토큰 길이가 PLAID 속도에 미치는 영향 (더 큰 하락 예상)
""")
