from datasets import load_dataset

from config import DEVICE, MODEL_NAME, TARGET_K
from embeddings import compute_query_embeddings, load_or_compute_image_embeddings
from metrics import compute_method_metrics
from save_results import save_benchmark_results
from search.block_centroid import block_centroid_search, load_cluster_index
from search.box_bound import box_bound_search
from search.box_bound import load_cluster_index as load_box_index
from search.combined_centroid import combined_centroid_search
from search.global_centroid import global_centroid_search
from search.naive import naive_search
from search.plaid import plaid_search


def _make_method(name: str, ranked_results: list, gt_indices: list,
                 t_search: float, t_build: float = 0.0,
                 pruning_rate: float | None = None,
                 ubound_ranges: list | None = None,
                 score_ranges: list | None = None) -> dict:
    return {
        "name": name,
        "ranked_results": ranked_results,
        "t_search": t_search,
        "t_build": t_build,
        "pruning_rate": pruning_rate,
        "ubound_ranges": ubound_ranges,
        "score_ranges": score_ranges,
        **compute_method_metrics(ranked_results, gt_indices),
    }


def run_benchmark(dataset_name: str) -> dict | None:
    slug = dataset_name.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    try:
        ds = load_dataset(dataset_name, split="test")
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    # ── 쿼리 / corpus 준비 ──────────────────────────────
    queries = ds["query"]
    valid_mask = [q is not None for q in queries]
    n_skipped = valid_mask.count(False)
    if n_skipped:
        print(f"  Warning: skipping {n_skipped} samples with None query")
    print(f"  Queries: {len(ds)} (valid: {sum(valid_mask)})")

    filename_to_corpus_idx: dict = {}
    corpus_images: list = []
    for i in range(len(ds)):
        fn = ds[i]["image_filename"]
        if fn not in filename_to_corpus_idx:
            filename_to_corpus_idx[fn] = len(corpus_images)
            corpus_images.append(ds["image"][i])
    print(f"  Unique images (corpus): {len(corpus_images)}")

    gt_indices = [
        filename_to_corpus_idx[ds[i]["image_filename"]]
        for i in range(len(ds)) if valid_mask[i]
    ]
    queries = [q for q, v in zip(queries, valid_mask) if v]
    n = len(queries)

    # ── 임베딩 ─────────────────────────────────────────
    image_embeddings, t_img_emb = load_or_compute_image_embeddings(slug, corpus_images)
    query_embeddings = compute_query_embeddings(queries)

    # ── 검색 ───────────────────────────────────────────
    scores, t_naive = naive_search(query_embeddings, image_embeddings)
    all_plaid_results, t_build, t_plaid = plaid_search(
        query_embeddings, image_embeddings, k=TARGET_K, device=DEVICE
    )
    idx_assets = load_cluster_index(dataset_name)

    print("  Running Global Centroid Pruning (Level 1)...")
    gc_ranked_results, t_gc, pruning_rate_gc, gc_ubound_ranges, gc_score_ranges, gc_all_ubounds = global_centroid_search(
        query_embeddings, image_embeddings, idx_assets, filename_to_corpus_idx, ds
    )

    print("  Running Block Residual Pruning (Level 2)...")
    bc_ranked_results, t_bc, pruning_rate, bc_ubound_ranges, bc_score_ranges, bc_all_ubounds = block_centroid_search(
        query_embeddings, image_embeddings, idx_assets, filename_to_corpus_idx, ds
    )

    print("  Running Combined Centroid Pruning (Level 1+2)...")
    cb_ranked_results, t_cb, pruning_rate_cb, cb_ubound_ranges, cb_score_ranges, _ = combined_centroid_search(
        query_embeddings, image_embeddings, idx_assets, filename_to_corpus_idx, ds
    )

    print("  Running Box Bound Pruning (L3)...")
    box_idx_assets = load_box_index(dataset_name)
    bb_ranked_results, t_bb, pruning_rate_bb, bb_ubound_ranges, bb_score_ranges, bb_all_ubounds = box_bound_search(
        query_embeddings, image_embeddings, box_idx_assets, filename_to_corpus_idx, ds
    )

    # ── 메서드별 표준 구조로 집계 ─────────────────────
    naive_ranked = [scores[i].argsort(descending=True).tolist() for i in range(n)]
    plaid_ranked = [[r[0] for r in all_plaid_results[i]] for i in range(n)]

    methods = [
        _make_method("Naive",          naive_ranked,        gt_indices, t_naive),
        _make_method("Fast PLAID",     plaid_ranked,        gt_indices, t_plaid,  t_build=t_build),
        _make_method("Global Centroid (L1)", gc_ranked_results,  gt_indices, t_gc,  pruning_rate=pruning_rate_gc,
                     ubound_ranges=gc_ubound_ranges, score_ranges=gc_score_ranges),
        _make_method("Block Residual (L2)", bc_ranked_results, gt_indices, t_bc, pruning_rate=pruning_rate,
                     ubound_ranges=bc_ubound_ranges, score_ranges=bc_score_ranges),
        _make_method("Combined (L1+L2)",   cb_ranked_results, gt_indices, t_cb, pruning_rate=pruning_rate_cb,
                     ubound_ranges=cb_ubound_ranges, score_ranges=cb_score_ranges),
        _make_method("Box Bound (L3)",     bb_ranked_results, gt_indices, t_bb, pruning_rate=pruning_rate_bb,
                     ubound_ranges=bb_ubound_ranges, score_ranges=bb_score_ranges),
    ]

    # ── 결과 출력 ───────────────────────────────────────
    print(f"\n  {'Method':<26} {'R@1':>7} {'nDCG@5':>8} {'nDCG@10':>9} "
          f"{'Img Emb(s)':>11} {'Prune Rate':>11} {'Search(s)':>10} {'Total(s)':>9}")
    print(f"  {'-'*98}")
    for m in methods:
        prune_str = f"{m['pruning_rate']:.2%}" if m['pruning_rate'] is not None else "-"
        t_total   = t_img_emb + m["t_build"] + m["t_search"]
        print(f"  {m['name']:<26} {m['recall@1']:>6.2%} {m['ndcg@5']:>8.4f} {m['ndcg@10']:>9.4f} "
              f"{t_img_emb:>11.2f} {prune_str:>11} {m['t_search']:>10.2f} {t_total:>9.2f}")

    # ── 결과 저장 ───────────────────────────────────────
    m_naive = next(m for m in methods if m["name"] == "Naive")
    m_plaid = next(m for m in methods if m["name"] == "Fast PLAID")
    m_gc    = next(m for m in methods if m["name"] == "Global Centroid (L1)")
    m_bc    = next(m for m in methods if m["name"] == "Block Residual (L2)")
    m_bb    = next(m for m in methods if m["name"] == "Box Bound (L3)")
    save_benchmark_results(
        model_name=MODEL_NAME, ds=ds,
        dataset=dataset_name, n_queries=n,
        queries=queries, filename_to_corpus_idx=filename_to_corpus_idx, gt_indices=gt_indices,
        scores=scores, t_img_emb=t_img_emb,
        all_plaid_results=all_plaid_results,
        methods=methods,
        t_naive=m_naive["t_search"], t_build=m_plaid["t_build"], t_plaid=m_plaid["t_search"],
        t_gc=m_gc["t_search"],   pruning_rate_gc=m_gc["pruning_rate"],
        t_bc=m_bc["t_search"],   pruning_rate=m_bc["pruning_rate"],
        t_bb=m_bb["t_search"],   pruning_rate_bb=m_bb["pruning_rate"],
        recall_naive=m_naive["recall@1"], recall_plaid=m_plaid["recall@1"],
        recall_gc=m_gc["recall@1"],       recall_bc=m_bc["recall@1"],
        recall_bb=m_bb["recall@1"],
        ndcg5_naive=m_naive["ndcg@5"],   ndcg10_naive=m_naive["ndcg@10"],
        ndcg5_plaid=m_plaid["ndcg@5"],   ndcg10_plaid=m_plaid["ndcg@10"],
        ndcg5_gc=m_gc["ndcg@5"],         ndcg10_gc=m_gc["ndcg@10"],
        ndcg5_bc=m_bc["ndcg@5"],         ndcg10_bc=m_bc["ndcg@10"],
        ndcg5_bb=m_bb["ndcg@5"],         ndcg10_bb=m_bb["ndcg@10"],
        ndcg5_naive_vals=m_naive["ndcg5_vals"], ndcg10_naive_vals=m_naive["ndcg10_vals"],
        ndcg5_plaid_vals=m_plaid["ndcg5_vals"], ndcg10_plaid_vals=m_plaid["ndcg10_vals"],
        ndcg5_gc_vals=m_gc["ndcg5_vals"],       ndcg10_gc_vals=m_gc["ndcg10_vals"],
        ndcg5_bc_vals=m_bc["ndcg5_vals"],       ndcg10_bc_vals=m_bc["ndcg10_vals"],
        ndcg5_bb_vals=m_bb["ndcg5_vals"],       ndcg10_bb_vals=m_bb["ndcg10_vals"],
        gc_ranked_results=m_gc["ranked_results"],
        bc_ranked_results=m_bc["ranked_results"],
        bb_ranked_results=m_bb["ranked_results"],
        gc_ubound_ranges=m_gc["ubound_ranges"],  gc_score_ranges=m_gc["score_ranges"],
        bc_ubound_ranges=m_bc["ubound_ranges"],  bc_score_ranges=m_bc["score_ranges"],
        bb_ubound_ranges=m_bb["ubound_ranges"],  bb_score_ranges=m_bb["score_ranges"],
        gc_all_ubounds=gc_all_ubounds,
        bc_all_ubounds=bc_all_ubounds,
        bb_all_ubounds=bb_all_ubounds,
    )

    return {
        "dataset": dataset_name,
        "n_queries": n,
        "t_img_emb": t_img_emb,
        "gt_indices": gt_indices,
        "queries": queries,
        "filename_to_corpus_idx": filename_to_corpus_idx,
        "ds": ds,
        "methods": methods,
        # save_results.py 호환용
        "scores": scores,
        "all_plaid_results": all_plaid_results,
    }
