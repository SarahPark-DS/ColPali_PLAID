from pathlib import Path

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

MODEL_NAME = "vidore/colpali-v1.3"
DEVICE = "cuda:0"

CACHE_DIR = Path("cache")
INDEX_DIR = Path("./indexing/cluster_index_output_4x")

RESULTS_DIR = Path("results")

TARGET_K = 10
