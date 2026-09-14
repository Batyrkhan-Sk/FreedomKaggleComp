"""VLAD over RootSIFT -- a global descriptor built from local features.

Every approach tried so far used deep semantic embeddings, which encode *what a
picture depicts*. That is the wrong axis for this task: the gallery holds two
different Monet paintings of the same bridge, and any semantic model places them
almost on top of each other. Fine-tuning for augmentation-invariance made it
worse (0.768 vs 0.792) and de-rotating the queries changed nothing (0.8289 vs
0.8285), which together rule out the "make the model tolerate the transform"
family.

Local features attack it differently. SIFT keypoints describe specific corners,
craquelure and brushstroke junctions rather than subject matter, and they are
constructed to survive rotation, scale and illumination change. VLAD then
aggregates those descriptors into one fixed-length vector, so the result still
fits the competition's "submit an embedding per image" format.

Two design points that matter for accuracy:

* **RootSIFT.** L1-normalising then square-rooting a SIFT descriptor makes
  Euclidean distance behave like the Hellinger kernel, which is a consistent and
  free improvement on raw SIFT.
* **Power normalisation.** VLAD vectors are bursty -- a repeated texture floods
  one cluster. Signed square root followed by L2 suppresses that.

Runs entirely on CPU.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 640          # SIFT is scale-invariant; full resolution is wasted time
MAX_KEYPOINTS = 400
CODEBOOK_SIZE = 64      # VLAD dim = CODEBOOK_SIZE * 128

_sift = None


def _detector():
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=MAX_KEYPOINTS)
    return _sift


def root_sift(path: str | Path) -> np.ndarray:
    """RootSIFT descriptors for one image, or an empty array if none found."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((0, 128), dtype=np.float32)
    h, w = img.shape
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    _, desc = _detector().detectAndCompute(img, None)
    if desc is None or len(desc) == 0:
        return np.zeros((0, 128), dtype=np.float32)
    desc = desc.astype(np.float32)
    desc /= (desc.sum(axis=1, keepdims=True) + 1e-7)     # L1
    return np.sqrt(desc)                                  # Hellinger


def encode_vlad(desc: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Sum of residuals to the nearest centroid, per cluster."""
    k, d = centroids.shape
    out = np.zeros((k, d), dtype=np.float32)
    if len(desc) == 0:
        return out.ravel()
    # nearest centroid via squared distance, expanded to avoid a big temporary
    d2 = (desc**2).sum(1)[:, None] - 2 * desc @ centroids.T + (centroids**2).sum(1)[None, :]
    assign = d2.argmin(axis=1)
    for j in range(k):
        m = assign == j
        if m.any():
            out[j] = (desc[m] - centroids[j]).sum(axis=0)
    v = out.ravel()
    v = np.sign(v) * np.sqrt(np.abs(v))                   # power normalisation
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def train_codebook(paths, n_images: int = 1500, seed: int = 0) -> np.ndarray:
    """K-means centroids from descriptors sampled across the corpus."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(paths), size=min(n_images, len(paths)), replace=False)
    pool = []
    for i in pick:
        d = root_sift(paths[i])
        if len(d):
            take = rng.choice(len(d), size=min(60, len(d)), replace=False)
            pool.append(d[take])
    pool = np.concatenate(pool) if pool else np.zeros((1, 128), np.float32)
    print(f"  codebook pool: {pool.shape[0]:,} descriptors")

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    _, _, centers = cv2.kmeans(pool, CODEBOOK_SIZE, None, crit, 3,
                               cv2.KMEANS_PP_CENTERS)
    return centers.astype(np.float32)
