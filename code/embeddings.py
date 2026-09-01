import pickle
import time

import torch
from tqdm import tqdm

from config import CACHE_DIR, DEVICE
from model import get_model


def load_or_compute_image_embeddings(slug: str, corpus_images: list) -> tuple[list, float]:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{slug}_image_embeddings.pkl"

    t0 = time.perf_counter()
    image_embeddings = None

    if cache_path.exists():
        print("  Loading cached image embeddings...")
        with open(cache_path, "rb") as f:
            image_embeddings = pickle.load(f)
        if len(image_embeddings) != len(corpus_images):
            print(f"  Cache size mismatch ({len(image_embeddings)} vs {len(corpus_images)}), recomputing...")
            cache_path.unlink()
            image_embeddings = None

    if image_embeddings is None:
        model, processor = get_model()
        print("  Generating image embeddings...")
        image_embeddings = []
        for img in tqdm(corpus_images, desc="  Image Embedding"):
            batch = processor.process_images([img.convert("RGB")]).to(DEVICE)
            with torch.no_grad():
                emb = model(**batch)
            image_embeddings.append(emb[0].cpu())
        with open(cache_path, "wb") as f:
            pickle.dump(image_embeddings, f)

    t_img_emb = time.perf_counter() - t0
    return image_embeddings, t_img_emb


def compute_query_embeddings(queries: list) -> list:
    model, processor = get_model()
    print("  Generating query embeddings...")
    query_embeddings = []
    for q in tqdm(queries, desc="  Query Embedding"):
        batch = processor.process_queries([q]).to(DEVICE)
        with torch.no_grad():
            emb = model(**batch)
        query_embeddings.append(emb[0].cpu())
    return query_embeddings
