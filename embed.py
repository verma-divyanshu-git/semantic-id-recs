"""Turn each item's text, and optionally its product image, into embeddings with frozen pretrained models.

Run:
    uv run python embed.py            # writes data/text_emb.npy, one 768-number row per item id - 1
    uv run python embed.py --images   # downloads product images, writes data/image_emb.npy, 512 numbers per item
"""

import argparse
import html
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HOME", ".hf_cache")  # keep downloaded models inside the project

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from data import DATA, load

MODEL = "sentence-transformers/sentence-t5-base"  # TIGER used Sentence-T5
IMAGE_MODEL = "sentence-transformers/clip-ViT-B-32"
OUT = DATA / "text_emb.npy"
IMAGE_OUT = DATA / "image_emb.npy"
IMAGES = DATA / "images"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def item_text(m):
    categories = "; ".join(" > ".join(path) for path in m.get("categories") or [])
    text = f"Title: {m.get('title') or ''}. Brand: {m.get('brand') or ''}. Categories: {categories}. Price: {m.get('price') or ''}."
    return html.unescape(text)  # titles contain HTML codes such as &quot; and &amp;


def fetch_image(item, url):
    path = IMAGES / f"{item}.jpg"
    if not path.exists():
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(url, part)
        part.rename(path)
    return path


def embed_images(meta, batch=512):
    """CLIP embedding of each item's product image. The 7 items without an image get a row of zeros."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    have = [(i, m["imUrl"]) for i, m in enumerate(meta) if m.get("imUrl")]
    with ThreadPoolExecutor(32) as pool:
        paths = list(pool.map(lambda a: fetch_image(*a), have))
    model = SentenceTransformer(IMAGE_MODEL, device=DEVICE)
    emb = np.zeros((len(meta), model.get_sentence_embedding_dimension()), dtype=np.float32)
    for start in range(0, len(have), batch):  # open images in batches so they don't all sit in memory
        images = [Image.open(p).convert("RGB") for p in paths[start : start + batch]]
        rows = [i for i, _ in have[start : start + batch]]
        emb[rows] = model.encode(images, batch_size=64, normalize_embeddings=True)
        print(f"{start + len(images)} of {len(have)} images", flush=True)
    return emb


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", action="store_true")
    args = p.parse_args()
    _, _, _, meta = load()
    if args.images:
        out, emb = IMAGE_OUT, embed_images(meta)
    else:
        texts = [item_text(m) for m in meta]
        print(texts[0])
        out = OUT
        emb = SentenceTransformer(MODEL, device=DEVICE).encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    np.save(out, emb.astype(np.float32))
    print(f"saved {emb.shape} to {out}")
