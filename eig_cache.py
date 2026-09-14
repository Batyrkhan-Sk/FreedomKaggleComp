"""One eigendecomposition, then every (rank, alpha, blend) is free.

The graph is fully connected at every k tested, so eigenvalue 1 has multiplicity
1 -- no component-indicator degeneracy. But the spectrum decays slowly (256
eigenvalues still above 0.966), so a low rank captures almost none of the
ranking-relevant structure. Compute a deep decomposition once and truncate.
"""
import argparse, os, numpy as np
from scipy.sparse.linalg import eigsh
from task1_concat import whiten
from bake_verified import build_base
from spectral import build_graph

ap = argparse.ArgumentParser()
ap.add_argument('--k', type=int, default=50)
ap.add_argument('--gamma', type=float, default=3.0)
ap.add_argument('--rank', type=int, default=2048)
a = ap.parse_args()

CACHE = 'base5120.npy'
if os.path.exists(CACHE):
    X = np.load(CACHE)
else:
    base, _ = build_base(0.15); X = whiten(base, 5120).astype(np.float32); np.save(CACHE, X)
print('base', X.shape, flush=True)

S = build_graph(X, a.k, a.gamma)
print(f'graph nnz {S.nnz:,}', flush=True)
lam, U = eigsh(S, k=a.rank, which='LA')
o = np.argsort(-lam); lam, U = lam[o], U[:, o]
np.savez(f'eigs_k{a.k}_g{int(a.gamma)}.npz', lam=lam, U=U.astype(np.float32))
print(f'saved eigs_k{a.k}_g{int(a.gamma)}.npz  rank {a.rank}')
for r in (1, 64, 256, 1024, 2047):
    if r < len(lam): print(f'   lambda[{r}] = {lam[r]:.5f}')
