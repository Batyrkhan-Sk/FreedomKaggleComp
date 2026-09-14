"""Feed LoFTR's verified pairs into the diffusion graph instead of baking them.

Baking could only overwrite a query's own top-1, so 137 confident verifications
moved 10 rows and the score did not budge. But diffusion PROPAGATES: a verified
edge (q,g) reshapes the manifold around both nodes, so it can also correct
queries LoFTR could not verify, through their neighbours. That is the difference
between using 137 facts as 137 overrides and using them as 137 constraints.

The submission stays legal because Fast Spectral Ranking turns any affinity
matrix into vectors whose cosine reproduces its manifold ranking -- so an
arbitrary similarity source becomes submittable.
"""
import argparse, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from task1_concat import l2
from spectral import build_graph

ap = argparse.ArgumentParser()
ap.add_argument('--k', type=int, default=50)
ap.add_argument('--gamma', type=float, default=3.0)
ap.add_argument('--gate', type=int, default=100, help='min LoFTR inliers to trust')
ap.add_argument('--lw', type=float, default=1.0, help='weight of a verified edge')
ap.add_argument('--rank', type=int, default=2048)
ap.add_argument('--alpha', type=float, default=0.99)
ap.add_argument('--blend', type=float, default=4.0)
ap.add_argument('--precision', type=int, default=5)
ap.add_argument('--out', default='')
a = ap.parse_args()

X = np.load('base5120.npy')
names = np.load('kout/lost-in-the-museum/feature_names.npy', allow_pickle=True)
d = np.load('verified_full.npz'); cand, bg, bn = d['cand'], d['best_g'], d['best_n']
sel = (bn >= a.gate)
q, g = cand[sel], bg[sel]
print(f'LoFTR edges at gate {a.gate}: {sel.sum()}')

S = build_graph(X, a.k, a.gamma)          # normalised already; rebuild raw below
# rebuild the UNnormalised affinity so LoFTR edges are added on a comparable scale
W = sp.csr_matrix(S.shape)
Wb = build_graph.__wrapped__ if hasattr(build_graph, '__wrapped__') else None
# simplest correct route: recompute the raw kNN affinity here
n = len(X); rows, cols, vals = [], [], []
for i in range(0, n, 512):
    Sb = X[i:i+512] @ X.T
    np.put_along_axis(Sb, np.arange(i, min(i+512, n))[:, None], -1.0, axis=1)
    idx = np.argpartition(-Sb, a.k, axis=1)[:, :a.k]
    v = np.take_along_axis(Sb, idx, axis=1)
    rows.append(np.repeat(np.arange(i, i+len(Sb)), a.k)); cols.append(idx.ravel()); vals.append(v.ravel())
r_, c_ = np.concatenate(rows), np.concatenate(cols)
v_ = np.clip(np.concatenate(vals), 0, None) ** a.gamma
W = sp.csr_matrix((v_, (r_, c_)), shape=(n, n))
W = W.maximum(W.T)
print(f'base affinity: nnz {W.nnz:,}  mean edge {W.data.mean():.5f}  max {W.data.max():.5f}')

L = sp.csr_matrix((np.full(len(q), a.lw), (q, g)), shape=(n, n))
W = W.maximum(L.maximum(L.T))
print(f'after LoFTR:   nnz {W.nnz:,}  (verified edge weight {a.lw})')

dg = np.asarray(W.sum(1)).ravel() + 1e-12
Dm = sp.diags(dg ** -0.5)
Sn = (Dm @ W @ Dm).tocsr()
lam, U = eigsh(Sn, k=a.rank, which='LA')
o = np.argsort(-lam); lam, U = lam[o], U[:, o]
print(f'lambda: {lam[0]:.4f} .. {lam[-1]:.4f}')

scale = 1.0 / np.sqrt(np.maximum(1.0 - a.alpha * lam, 1e-6))
phi = l2((U * scale[None, :]).astype(np.float32))
out = phi if a.blend <= 0 else l2(np.hstack([phi, a.blend * l2(X)]).astype(np.float32))

def top1(F, qs):
    o_ = np.zeros(len(qs), np.int64)
    for i in range(0, len(qs), 256):
        b = qs[i:i+256]; Sm = (F[b] @ F.T).astype(np.float32)
        Sm[np.arange(len(b)), b] = -2; o_[i:i+len(b)] = Sm.argmax(1)
    return o_
b1, s1 = top1(X, cand), top1(out, cand)
ver = np.isin(cand, q)
print(f'\ntop-1 changed overall      : {(b1!=s1).mean()*100:5.1f}%')
print(f'  among LoFTR-verified     : {(b1!=s1)[ver].mean()*100:5.1f}%  ({ver.sum()} cands)')
print(f'  among NOT verified (prop): {(b1!=s1)[~ver].mean()*100:5.1f}%  <- the propagation effect')
agree = (b1[ver] == np.array([g[list(q).index(c)] for c in cand[ver]]))
print(f'  descriptor already agreed with LoFTR on {agree.mean()*100:.0f}% of verified')

if a.out:
    import pandas as pd
    df = pd.DataFrame(np.round(out, a.precision), columns=[f'feature_{i}' for i in range(out.shape[1])])
    df.insert(0, 'image_name', names); df['ID'] = names
    df.to_csv(a.out, index=False); print(f'wrote {a.out}: {df.shape}')
