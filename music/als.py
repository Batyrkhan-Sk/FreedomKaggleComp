"""Implicit-feedback ALS (Hu-Koren-Volinsky), numpy/scipy only.

Deliberately NOT the `implicit` package: this has to run in a Kaggle notebook
with internet off, and numpy/scipy are the only linear-algebra dependencies
guaranteed to be there. ~60 lines is a fair price for removing that risk.

Why this instead of the existing TruncatedSVD block (latent.py):

  1. SVD treats every unobserved (user, item) as a hard 0. ALS treats it as
     *low confidence* -- c_ui = 1 + alpha*r_ui, with p_ui = 1 only where played.
     That difference is the whole reason ALS exists for implicit feedback.
  2. latent.py places a user at the weighted CENTROID of their item factors
     (`R @ item_f`). That is a projection, not a fit -- nothing solves for the
     user vector that best reconstructs their listening. ALS solves it exactly.

Both sides are alternating ridge solves, so there is no learning rate and no
early-stopping decision to get wrong.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _solve_side(Xt: sp.csr_matrix, Y: np.ndarray, reg: float, alpha: float) -> np.ndarray:
    """One ALS half-step: solve every row of X given fixed factors Y.

    For row u with observed items I_u and raw weights r:
        A = YtY + Y[I_u]^T diag(alpha*r) Y[I_u] + reg*I
        b = Y[I_u]^T (1 + alpha*r)          (p_ui = 1 on observed entries)
    The YtY precompute is what makes this linear in nnz rather than in n_items.
    """
    n, f = Xt.shape[0], Y.shape[1]
    YtY = Y.T @ Y
    base = YtY + reg * np.eye(f, dtype=np.float64)
    out = np.zeros((n, f), dtype=np.float32)
    indptr, indices, data = Xt.indptr, Xt.indices, Xt.data
    for u in range(n):
        s, e = indptr[u], indptr[u + 1]
        if s == e:
            continue
        Yu = Y[indices[s:e]].astype(np.float64)
        cu = alpha * data[s:e].astype(np.float64)
        A = base + (Yu * cu[:, None]).T @ Yu
        b = Yu.T @ (1.0 + cu)
        out[u] = np.linalg.solve(A, b)
    return out


def als(R: sp.csr_matrix, factors: int = 64, iters: int = 12, reg: float = 0.05,
        alpha: float = 40.0, seed: int = 0, verbose: bool = True):
    """R: users x items, values = raw affinity (we use log1p of decayed plays)."""
    rng = np.random.default_rng(seed)
    Rc, Rt = R.tocsr(), R.T.tocsr()
    X = (rng.standard_normal((R.shape[0], factors)) * 0.01).astype(np.float32)
    Y = (rng.standard_normal((R.shape[1], factors)) * 0.01).astype(np.float32)
    for it in range(iters):
        X = _solve_side(Rc, Y, reg, alpha)
        Y = _solve_side(Rt, X, reg, alpha)
        if verbose:
            print(f"    als iter {it + 1}/{iters}", flush=True)
    return X, Y
