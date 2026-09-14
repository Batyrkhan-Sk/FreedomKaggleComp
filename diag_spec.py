"""Is the spectral embedding actually RANKING, or just clustering?

If the kNN graph fragments, eigenvalue 1 has multiplicity = #components and the
top eigenvectors are component indicators. Everything in a component then gets a
near-identical vector, cosine saturates, and the host's top-3 becomes arbitrary.
That failure would look fine in every log and only show up on the leaderboard.
"""
import numpy as np, scipy.sparse as sp, os
from scipy.sparse.csgraph import connected_components
from task1_concat import l2, whiten
from bake_verified import build_base
from spectral import build_graph

CACHE = 'base5120.npy'
if os.path.exists(CACHE):
    X = np.load(CACHE); names = np.load('kout/lost-in-the-museum/feature_names.npy', allow_pickle=True)
    print('base from cache', X.shape)
else:
    base, names = build_base(0.15)
    X = whiten(base, 5120).astype(np.float32)
    np.save(CACHE, X); print('base built + cached', X.shape)

cand = np.load('rot_candidates.npy')
for k in (10, 25, 50, 100):
    S = build_graph(X, k, 3.0)
    n_comp, lab = connected_components(S, directed=False)
    sizes = np.bincount(lab)
    print(f'k={k:3d}: {n_comp:5d} components  largest {sizes.max():6d}  '
          f'singletons {(sizes==1).sum():5d}  '
          f'-> eigenvalue 1 has multiplicity {n_comp}')
