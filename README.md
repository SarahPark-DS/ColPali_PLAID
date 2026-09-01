# ColPali + PLAID: Document-Level Pruning Benchmark

Benchmark of centroid-/box-based **document-level pruning** for late-interaction
(ColPali / ColBERT-style) retrieval, evaluated on the
[ViDoRe](https://huggingface.co/vidore) visual document retrieval datasets.

Every method returns a top-10 ranking per query; we compare retrieval quality
(Recall@1, nDCG@5, nDCG@10), search latency, and the fraction of full
query–document MaxSim computations that pruning avoids (**pruning rate ρ**).

## Methods

| Method | Description |
| --- | --- |
| **Naive** | Exhaustive MaxSim over every query–document pair (ground-truth ranking). |
| **Fast PLAID** | [`fast-plaid`](https://pypi.org/project/fast-plaid/) centroid-based ANN index. |
| **Global Centroid (L1)** | Upper bound `⟨q_t, c_j⟩ + ‖q_t‖·r_j` per cluster; prune documents whose bound falls below the running top-k threshold. |
| **Block Residual (L2)** | Tighter bound using per-block residual radii: `⟨q_t, c_j⟩ + Σ_b ‖q_t^(b)‖·ρ_j^(b)`. |
| **Combined (L1+L2)** | Per-(token, cluster) minimum of the L1 and L2 residuals — always at least as tight as either. |
| **Box Bound (L3)** | Coordinate-wise residual box `[ℓ_j, u_j]`: `⟨q_t, c_j⟩ + Σ_s max(q_{t,s}·ℓ_{j,s}, q_{t,s}·u_{j,s})`. |

All bound methods share one k-means cluster index per dataset and differ only in
the residual term stored alongside the centroids.

## Repository layout

```
.
├── code/
│   ├── run.py                  # entry point — runs the full benchmark over all datasets
│   ├── benchmark.py            # single-dataset pipeline (embed → search → score → save)
│   ├── config.py               # datasets, model name, device, paths
│   ├── model.py                # ColPali model / processor loader (cached)
│   ├── embeddings.py           # image & query embedding (image embeddings cached to disk)
│   ├── metrics.py              # Recall@1, nDCG@k
│   ├── save_results.py         # per-query CSV, summary JSON, comparison / bound-distribution charts
│   ├── requirements.txt
│   ├── search/
│   │   ├── naive.py            # exhaustive MaxSim
│   │   ├── plaid.py            # Fast PLAID index + search
│   │   ├── global_centroid.py  # L1
│   │   ├── block_centroid.py   # L2
│   │   ├── combined_centroid.py# L1+L2
│   │   └── box_bound.py        # L3
│   └── indexing/
│       ├── create_centroid.py         # step 1: k-means cluster index (cluster_index.npz)
│       ├── block_residual_builder.py  # step 2: add per-block residual radii
│       ├── box_bound_builder.py        # step 3: add coordinate-wise residual box
│       ├── radius_stats_all_datasets.py
│       ├── visualize_indexing.py
│       └── visualize_patch_clusters.py
└── result/                     # committed snapshot of benchmark outputs (see below)
```

## Setup

Requires an NVIDIA GPU (CUDA). Tested with Python 3.11, PyTorch 2.11 / CUDA 12.8.

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# fast-plaid pulls heavy deps; if you hit conflicts:
#   pip install --no-deps fast-plaid fastkmeans
```

Datasets are pulled automatically from the Hugging Face Hub on first run
(`vidore/*`, see `code/config.py`). The model is `vidore/colpali-v1.3`.

## Usage

All commands assume you are in `code/`.

### 1. Build the cluster indices

```bash
cd indexing
python create_centroid.py          # -> ./cluster_index_output_4x/<dataset>/cluster_index.npz
python block_residual_builder.py   # adds block_radii  -> cluster_index_block.npz
python box_bound_builder.py         # adds box bounds   -> cluster_index_box.npz
cd ..
```

This encodes every corpus image once and runs FastKMeans
(`K = 2^floor(log2(64·√N))`, N = total patch vectors). Output lands in
`code/indexing/cluster_index_output_4x/`, which is where `config.INDEX_DIR`
expects it.

### 2. Run the benchmark

```bash
python run.py
```

For each dataset in `config.DATASETS` this:

1. loads/creates ColPali image embeddings (cached to `code/cache/*.pkl`),
2. embeds queries,
3. runs Naive, Fast PLAID, and the four pruning methods,
4. writes results to `code/results/<dataset>/`,
5. prints a final cross-dataset summary table and saves
   `code/results/final_benchmark_summary.png`.

To restrict the run, edit the `DATASETS` list in `code/config.py`.
Device is set by `config.DEVICE` (default `cuda:0`).

## Outputs

Per dataset (`result/<dataset>/`, timestamped `YYYYMMDD_HHMMSS_*`):

| File | Contents |
| --- | --- |
| `*_summary.json` | Aggregate Recall@1 / nDCG / latency / pruning rate per method. |
| `*_summary_data.csv` | Same numbers in the console-table layout. |
| `*_per_query.csv` | Per-query prediction, correctness, nDCG, and bound/score ranges. |
| `*_comparison.png` | Recall@1 / nDCG@5 / nDCG@10 bar charts. |
| `*_bound_distribution.png` | Exact-score vs L1/L2/box upper-bound distributions and gap histograms. |

`result/final_benchmark_summary.png` is the cross-dataset roll-up.

## Datasets

`arxivqa`, `docvqa`, `infovqa`, `tabfquad`, `tatdqa`, `shiftproject`, and the four
`syntheticDocQA` splits (`artificial_intelligence`, `energy`,
`government_reports`, `healthcare_industry`) — all `vidore/*_test[_subsampled]`.

## Acknowledgement

This repository was developed with support from the 서울시립대학교 데이터 사이언스 플러스 차세대 융합인재 양성사업단 - http://dsplus.uos.ac.kr/