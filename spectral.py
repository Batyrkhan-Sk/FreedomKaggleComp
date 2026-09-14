"""Fast Spectral Ranking (Iscen et al., CVPR 2018) as a SUBMITTABLE embedding.

Diffusion / manifold ranking is the largest documented re-ranking win in this
problem family, but it normally produces a RANKING and this competition only
accepts VECTORS. FSR is the bridge: eigendecompose the kNN graph offline and
manifold ranking becomes a dot product in the spectral space, so the host's own
cosine search performs the diffusion.

  S  = D^-1/2 W D^-1/2          (symmetrically normalised kNN affinity)
  f  = (I - a S)^-1 y           (diffusion; y = query indicator)
     = U (I - a L)^-1 U^T y     (S = U L U^T)

so score(d | q) = u_d^T (I - a L)^-1 u_q, which is the dot product of

  phi_i = (I - a L)^-1/2 u_i

Two honest deviations from the paper:
  * the host computes COSINE, not dot product, so each row gets normalised.
    Within a query that only rescales by |phi_d|, which acts as a mild hub
    penalty -- a deviation, not a bug, but it is not exactly the paper's rule.
  * alpha-QE failed on this task; that says little here, since aQE averages raw
    neighbours while diffusion propagates over the whole manifold.

The real risk is the 9,000 artwork DISTRACTORS: diffusion spreads similarity
along dense clusters, and distractor clusters are where it can spread wrongly.
"""
import argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from task1_concat import l2, whiten
from bake_verified import build_base


def build_graph(X, k, gamma):
    """Top-k mutual affinity with similarities raised to gamma."""
    n = len(X)
    rows, cols, vals = [], [], []
    for i in range(0, n, 512):
        S = X[i:i + 512] @ X.T
        np.put_along_axis(S, np.arange(i, min(i + 512, n))[:, None], -1.0, axis=1)
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        v = np.take_along_axis(S, idx, axis=1)
        rows.append(np.repeat(np.arange(i, i + len(S)), k))
        cols.append(idx.ravel())
        vals.append(v.ravel())
    r, c = np.concatenate(rows), np.concatenate(cols)
    v = np.clip(np.concatenate(vals), 0, None) ** gamma
    W = sp.csr_matrix((v, (r, c)), shape=(n, n))
    W = W.maximum(W.T)                       # symmetrise, keeping the stronger edge
    d = np.asarray(W.sum(1)).ravel() + 1e-12
    Dm = sp.diags(d ** -0.5)
    return (Dm @ W @ Dm).tocsr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=50, help='neighbours per node')
    ap.add_argument('--gamma', type=float, default=3.0, help='similarity sharpening')
    ap.add_argument('--alpha', type=float, default=0.99, help='diffusion strength')
    ap.add_argument('--rank', type=int, default=1024, help='eigenvectors kept')
    ap.add_argument('--margin', type=float, default=0.15)
    ap.add_argument('--blend', type=float, default=0.0,
                    help='append the ORIGINAL block at this weight. 0 = pure '
                         'spectral. A hedge: with no offline validation, a pure '
                         'geometry swap is the highest-variance thing we can ship.')
    ap.add_argument('--precision', type=int, default=6)
    ap.add_argument('--out', default='submission_spectral.csv')
    args = ap.parse_args()

    base, names = build_base(args.margin)
    X = whiten(base, 5120)                                  # the 0.86912 space
    print(f'base {X.shape}')

    S = build_graph(X.astype(np.float32), args.k, args.gamma)
    print(f'graph nnz {S.nnz:,} ({S.nnz/len(X):.1f} edges/node)')

    r = min(args.rank, len(X) - 2)
    lam, U = eigsh(S, k=r, which='LA')
    order = np.argsort(-lam)
    lam, U = lam[order], U[:, order]
    print(f'eigenvalues: max {lam[0]:.4f}  min kept {lam[-1]:.4f}')

    scale = 1.0 / np.sqrt(np.maximum(1.0 - args.alpha * lam, 1e-6))
    phi = l2(U * scale[None, :])
    print(f'spectral embedding {phi.shape}  scale range '
          f'{scale.min():.2f}..{scale.max():.2f}')

    out = phi if args.blend <= 0 else np.hstack([l2(phi), args.blend * l2(X)])
    out = l2(out).astype(np.float32)
    np.save(args.out.replace('.csv', '.npy'), out)

    import pandas as pd
    df = pd.DataFrame(np.round(out, args.precision),
                      columns=[f'feature_{i}' for i in range(out.shape[1])])
    df.insert(0, 'image_name', names)
    df['ID'] = names
    df.to_csv(args.out, index=False)
    print(f'wrote {args.out}: {len(df)} rows x {df.shape[1]} cols')


if __name__ == '__main__':
    main()
