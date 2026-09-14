"""Turn raw DINOv2 features into a competition-format submission.csv.

Pipeline follows standard image-retrieval practice:
    L2-normalise -> PCA (optionally whitened) -> L2-normalise
Whitening tends to help here because the gallery contains 9,000 artwork
distractors, and the dominant "this is a painting" directions otherwise
soak up most of the variance and crowd the true match out of the top-3.
"""

import argparse

import numpy as np
import pandas as pd


def l2(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def pca_reduce(x, dim, whiten):
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    # Economy SVD: 20000 x D is small enough to do exactly, no randomisation.
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:dim]
    out = xc @ comps.T
    if whiten:
        scale = s[:dim] / np.sqrt(max(len(x) - 1, 1))
        out = out / (scale + 1e-8)
    explained = (s[:dim] ** 2).sum() / (s**2).sum()
    print(f"PCA {x.shape[1]} -> {dim}  (whiten={whiten})  explains {explained:.1%} of variance")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_l224")
    ap.add_argument("--dim", type=int, default=128, help="output dimensionality (0 = keep full)")
    ap.add_argument("--whiten", action="store_true", default=True)
    ap.add_argument("--no-whiten", dest="whiten", action="store_false")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    feats = np.load(f"{args.features}.npy")
    names = np.load(f"{args.features}_names.npy", allow_pickle=True)
    print(f"Loaded {feats.shape} from {args.features}.npy")

    n_dead = int((np.abs(feats).sum(axis=1) == 0).sum())
    if n_dead:
        print(f"WARNING: {n_dead} all-zero rows (failed image loads)")

    x = l2(feats.astype(np.float64))
    if args.dim and args.dim < x.shape[1]:
        x = pca_reduce(x, args.dim, args.whiten)
    x = l2(x).astype(np.float32)

    cols = [f"feature_{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=cols)
    df.insert(0, "image_name", names)
    df["ID"] = df["image_name"]

    df.to_csv(args.out, index=False, float_format="%.6f")

    print(f"\nWrote {args.out}")
    print(f"  rows: {len(df)}  cols: {len(df.columns)}")
    print(f"  norms: min={np.linalg.norm(x, axis=1).min():.4f} max={np.linalg.norm(x, axis=1).max():.4f}")
    print(f"  first: {df['image_name'].iloc[0]}   last: {df['image_name'].iloc[-1]}")


if __name__ == "__main__":
    main()
