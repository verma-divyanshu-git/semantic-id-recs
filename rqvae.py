"""RQ-VAE: squeeze each item's text embedding into 3 small numbers, its semantic ID.

Each of the 3 levels has a codebook of 256 vectors.
Level 1 picks the code nearest to the item, level 2 picks the code nearest to what level 1 missed, and so on.
Similar items end up sharing their first codes.
A 4th number is added so that items with the same 3 codes still get different IDs.

Run: uv run python rqvae.py --split loo   # fits on training items only, writes data/semantic_ids_loo.json
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from data import DATA, get_split, load
from sasrec import DEVICE


def mlp(sizes):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])  # no ReLU after the last layer


def residual_quantize(z, codebooks):
    """At each level, pick the code nearest to what the earlier levels left unexplained."""
    codes, residuals, chosen = [], [], []
    r = z
    for cb in codebooks:
        idx = torch.cdist(r, cb).argmin(1)
        codes.append(idx)
        residuals.append(r)
        chosen.append(cb[idx])
        r = r - cb[idx].detach()
    return torch.stack(codes, 1), residuals, chosen


def add_collision_token(codes):
    """Append a counter so items with the same codes get IDs like (1, 2, 3, 0) and (1, 2, 3, 1)."""
    seen = Counter()
    out = []
    for c in map(tuple, codes):
        out.append(c + (seen[c],))
        seen[c] += 1
    return out


class RQVAE(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256, 128), latent=32, levels=3, codebook_size=256, beta=0.25):
        super().__init__()
        self.encoder = mlp([in_dim, *hidden, latent])
        self.decoder = mlp([latent, *reversed(hidden), in_dim])
        self.codebooks = nn.Parameter(torch.zeros(levels, codebook_size, latent))
        self.beta = beta

    @torch.no_grad()
    def init_codebooks(self, x):
        """Start each codebook at the k-means centres of its level's residuals, so no code starts unused."""
        r = self.encoder(x)
        for level, cb in enumerate(self.codebooks):
            km = KMeans(len(cb), n_init=1, random_state=0).fit(r.cpu().numpy())
            self.codebooks[level] = torch.tensor(km.cluster_centers_, dtype=r.dtype, device=r.device)
            r = r - self.codebooks[level][torch.tensor(km.labels_, device=r.device)]

    def forward(self, x):
        z = self.encoder(x)
        codes, residuals, chosen = residual_quantize(z, self.codebooks)
        # Pull each residual and its chosen code toward each other. beta sets how hard the encoder is pulled.
        quant_loss = sum(self.beta * (r - e.detach()).pow(2).sum(-1) + (r.detach() - e).pow(2).sum(-1) for r, e in zip(residuals, chosen))
        z_q = z + (sum(chosen) - z).detach()  # straight-through: decoder gradients reach the encoder
        recon_loss = (self.decoder(z_q) - x).pow(2).sum(-1)
        return codes, recon_loss.mean(), quant_loss.mean()


def id_stats(codes):
    ids = [tuple(c) for c in codes.tolist()]
    counts = Counter(ids)
    return {
        "codes_used_per_level": [len(set(level)) for level in zip(*ids)],
        "unique_3_code_ids": len(counts),
        "share_of_items_sharing_3_codes": round(sum(n for n in counts.values() if n > 1) / len(ids), 4),
        "max_items_per_3_codes": max(counts.values()),
    }


def train_rqvae(x, epochs, batch=1024, lr=1e-3):
    torch.manual_seed(0)
    model = RQVAE(x.shape[1]).to(DEVICE)
    model.init_codebooks(x)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(x), device=DEVICE).split(batch):
            _, recon, quant = model(x[idx])
            opt.zero_grad()
            (recon + quant).backward()
            opt.step()
        if epoch % 500 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                codes, recon, quant = model(x)
            print(f"epoch {epoch:5d}  recon {recon.item():.4f}  quant {quant.item():.4f}  {id_stats(codes)}", flush=True)
    return model.eval()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["loo", "time"], default="loo")
    p.add_argument("--epochs", type=int, default=3000)
    args = p.parse_args()

    seqs, times, items, _ = load()
    train = get_split(args.split, seqs, times)[0]
    train_items = sorted({i for s in train for i in s})
    emb = torch.tensor(np.load(DATA / "text_emb.npy"), device=DEVICE)

    start = time.time()
    model = train_rqvae(emb[torch.tensor(train_items, device=DEVICE) - 1], args.epochs)
    with torch.no_grad():
        codes, recon, _ = model(emb)
    ids = add_collision_token(codes.tolist())
    stats = {
        "train_items": len(train_items),
        "all_items": len(items),
        "recon_loss_all_items": round(recon.item(), 4),
        **id_stats(codes),
        "max_collision_token": max(i[-1] for i in ids),
        "minutes": round((time.time() - start) / 60, 1),
    }
    (DATA / f"semantic_ids_{args.split}.json").write_text(json.dumps(ids))
    Path(f"results/rqvae_{args.split}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
