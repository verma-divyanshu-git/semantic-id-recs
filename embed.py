"""Turn each item's title, brand, categories and price into a text embedding with a frozen pretrained model.

Run: uv run python embed.py   # writes data/text_emb.npy, one 768-number row per item id - 1
"""

import html
import os

os.environ.setdefault("HF_HOME", ".hf_cache")  # keep downloaded models inside the project

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from data import DATA, load

MODEL = "sentence-transformers/sentence-t5-base"  # TIGER used Sentence-T5
OUT = DATA / "text_emb.npy"


def item_text(m):
    categories = "; ".join(" > ".join(path) for path in m.get("categories") or [])
    text = f"Title: {m.get('title') or ''}. Brand: {m.get('brand') or ''}. Categories: {categories}. Price: {m.get('price') or ''}."
    return html.unescape(text)  # titles contain HTML codes such as &quot; and &amp;


if __name__ == "__main__":
    _, _, _, meta = load()
    texts = [item_text(m) for m in meta]
    print(texts[0])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    emb = SentenceTransformer(MODEL, device=device).encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    np.save(OUT, emb.astype(np.float32))
    print(f"saved {emb.shape} to {OUT}")
