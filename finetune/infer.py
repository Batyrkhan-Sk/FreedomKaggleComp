"""Embed the full corpus with a fine-tuned model and write the submission.

The projection head is deliberately discarded here: contrastive training warps
the head's output space to suit the loss, while the representation just beneath
it transfers better to retrieval. This is the standard SimCLR/MoCo finding and
it is why `RetrievalNet.features` exists separately from `forward`.

Post-processing is unchanged from the zero-shot pipeline that scored 0.8285:
L2-normalise, whitened PCA, L2-normalise. Whitening was the single biggest
lever in the zero-shot setting -- with 9,000 artwork distractors, the dominant
variance directions encode "this is a painting" and drown out the detail that
distinguishes one from another.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import InferenceDataset
from model import RetrievalNet
from train import DATA_DIR


def l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def whitened_pca(x: np.ndarray, dim: int) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x - mu, full_matrices=False)  # exact, deterministic
    dim = min(dim, x.shape[1])
    scale = s[:dim] / np.sqrt(len(x) - 1)
    explained = (s[:dim] ** 2).sum() / (s ** 2).sum()
    print(f"PCA {x.shape[1]} -> {dim}   explains {explained:.1%} of variance")
    return (x - mu) @ vt[:dim].T / (scale + 1e-8)


def embed_all(model, paths, size, device, batch, workers, fp16=False) -> np.ndarray:
    loader = DataLoader(InferenceDataset(paths, size), batch_size=batch,
                        shuffle=False, num_workers=workers)
    feats = None
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for imgs, idxs in loader:
            imgs = imgs.to(device)
            if fp16:
                imgs = imgs.half()
            vec = model.features(imgs).float().cpu().numpy()
            if feats is None:
                feats = np.zeros((len(paths), vec.shape[1]), dtype=np.float32)
            feats[idxs.numpy()] = vec
            done = int(idxs[-1]) + 1
            if done % (batch * 50) < batch:
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(paths)}  {rate:.1f} img/s  "
                      f"ETA {(len(paths)-done)/rate/60:.0f} min", flush=True)
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="finetuned.pt")
    ap.add_argument("--size", type=int, default=392)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--save-features", default="")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trained = ckpt["args"]
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model = RetrievalNet(trained["backbone"], trained["trainable_blocks"]).to(device)
    model.load_state_dict(ckpt["model"])
    if args.fp16:
        model = model.half()
    print(f"loaded {args.checkpoint} (epoch {ckpt['epoch']}, {trained['backbone']})  device {device}")

    paths = sorted(Path(args.data_dir).glob("*.png"))
    feats = embed_all(model, paths, args.size, device, args.batch, args.workers, args.fp16)
    print(f"embedded {feats.shape}")

    if args.save_features:
        np.save(args.save_features, feats)
        np.save(args.save_features.replace(".npy", "_names.npy"),
                np.array([p.name for p in paths]))

    x = l2(feats.astype(np.float64))
    x = l2(whitened_pca(x, args.dim)).astype(np.float32)

    df = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(x.shape[1])])
    df.insert(0, "image_name", [p.name for p in paths])
    df["ID"] = df["image_name"]
    df.to_csv(args.out, index=False, float_format="%.6f")
    print(f"wrote {args.out}: {len(df)} rows, {df.shape[1]} cols, "
          f"norms ~{np.linalg.norm(x, axis=1).mean():.4f}")


if __name__ == "__main__":
    main()
