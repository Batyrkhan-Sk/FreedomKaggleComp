"""Hit@3 measurement for a fine-tuned or zero-shot model.

The competition gives no labels, so the metric is reconstructed the same way the
training pairs are: degrade a known image, embed it, and check whether the
untouched original returns in the top 3 of the whole gallery.

**On circularity.** The model is trained on this very degradation family, so
evaluating at the training strength would largely measure memorisation of our
own augmentations. `--strength` therefore defaults to a value held out from
training (1.25, the level calibrated against the real leaderboard in the
zero-shot phase), and the leaderboard remains the final arbiter. A fine-tuned
model that wins here but not there means our augmentations do not match the
host's -- which is exactly the failure mode worth detecting early.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from augment import query_view
from data import InferenceDataset
from model import RetrievalNet
from train import DATA_DIR

Image.MAX_IMAGE_PIXELS = None


def l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def whitened_pca_fit(gallery: np.ndarray, dim: int):
    """Fit on the gallery, return a transform applied identically to both sides."""
    mu = gallery.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(gallery - mu, full_matrices=False)
    dim = min(dim, gallery.shape[1])
    comps, scale = vt[:dim], s[:dim] / np.sqrt(len(gallery) - 1)
    return lambda z: l2((z - mu) @ comps.T / (scale + 1e-8))


def load_model(checkpoint: str | None, backbone: str, device: str, fp16: bool):
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = RetrievalNet(ckpt["args"]["backbone"], ckpt["args"]["trainable_blocks"])
        model.load_state_dict(ckpt["model"])
        label = f"fine-tuned ({ckpt['args']['backbone']}, epoch {ckpt['epoch']})"
    else:
        model = RetrievalNet(backbone, trainable_blocks=0)
        label = f"zero-shot ({backbone})"
    model = model.to(device).eval()
    if fp16:
        model = model.half()
    return model, label


def embed(model, dataset, device, batch, workers, fp16, n_total, feature_dim=None):
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers)
    out = None
    with torch.no_grad():
        for imgs, idxs in loader:
            imgs = imgs.to(device)
            if fp16:
                imgs = imgs.half()
            v = model.features(imgs).float().cpu().numpy()
            if out is None:
                out = np.zeros((n_total, v.shape[1]), dtype=np.float32)
            out[idxs.numpy()] = v
    return out


class QueryDataset(torch.utils.data.Dataset):
    """Degraded views of a chosen subset, for use as synthetic queries."""

    def __init__(self, paths, size, strength, seed):
        self.paths, self.size = paths, size
        self.tf = query_view(size, strength)
        self.seed = seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        np.random.seed(self.seed + i)      # reproducible degradation per query
        torch.manual_seed(self.seed + i)
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="", help="empty = evaluate zero-shot")
    ap.add_argument("--backbone", default="dinov2_vitb14", help="used when no checkpoint")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--gallery", type=int, default=4000, help="first N images form the gallery")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--strength", type=float, default=1.25, help="held out from training")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    paths = sorted(Path(args.data_dir).glob("*.png"))[: args.gallery]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, label = load_model(args.checkpoint or None, args.backbone, device, args.fp16)

    gallery = embed(model, InferenceDataset(paths, args.size), device,
                    args.batch, args.workers, args.fp16, len(paths))

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(paths), size=min(args.queries, len(paths)), replace=False)
    qpaths = [paths[i] for i in picks]
    queries = embed(model, QueryDataset(qpaths, args.size, args.strength, args.seed),
                    device, args.batch, args.workers, args.fp16, len(qpaths))

    g, q = l2(gallery.astype(np.float64)), l2(queries.astype(np.float64))
    if args.dim and args.dim < g.shape[1]:
        tf = whitened_pca_fit(g, args.dim)
        g, q = tf(g), tf(q)

    order = np.argsort(-(q @ g.T), axis=1)
    hit1 = float(np.mean([picks[i] == order[i, 0] for i in range(len(picks))]))
    hit3 = float(np.mean([picks[i] in order[i, :3] for i in range(len(picks))]))

    print(f"{label}  @{args.size}  gallery {len(paths)}  strength {args.strength}")
    print(f"  Hit@1 {hit1:.3f}")
    print(f"  Hit@3 {hit3:.3f}")


if __name__ == "__main__":
    main()
