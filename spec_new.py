"""Spectral (diffusion) re-ranking on the CURRENT 0.92617 base.

The machinery was built against the 0.86912 space and never submitted. The
manifold at 0.92617 is much cleaner, so diffusion has better neighbours to
propagate along. Costs no GPU, so it runs while the 7B extraction has the card.

The submitted 9728-d CSV IS the whitened base -- reading it back skips a 40-min
rebuild.
"""
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from task1_concat import l2

print("loading the 0.92617 space from its submission ...", flush=True)
df = pd.read_csv("submission_4block_9728.csv")
names = df["image_name"].to_numpy()
X = df[[c for c in df.columns if c.startswith("feature_")]].to_numpy(np.float32)
del df
print("base", X.shape, flush=True)

K, GAMMA, RANK, ALPHA = 50, 3.0, 1024, 0.99
n = len(X); rows, cols, vals = [], [], []
for i in range(0, n, 512):
    S = X[i:i+512] @ X.T
    np.put_along_axis(S, np.arange(i, min(i+512, n))[:, None], -1.0, axis=1)
    idx = np.argpartition(-S, K, axis=1)[:, :K]
    v = np.take_along_axis(S, idx, axis=1)
    rows.append(np.repeat(np.arange(i, i+len(S)), K)); cols.append(idx.ravel()); vals.append(v.ravel())
W = sp.csr_matrix((np.clip(np.concatenate(vals), 0, None)**GAMMA,
                   (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
W = W.maximum(W.T)
d = np.asarray(W.sum(1)).ravel() + 1e-12
Sn = (sp.diags(d**-0.5) @ W @ sp.diags(d**-0.5)).tocsr()
print(f"graph nnz {Sn.nnz:,}", flush=True)

lam, U = eigsh(Sn, k=RANK, which="LA")
o = np.argsort(-lam); lam, U = lam[o], U[:, o]
print(f"lambda {lam[0]:.4f} .. {lam[-1]:.4f}", flush=True)
phi = l2((U * (1.0/np.sqrt(np.maximum(1-ALPHA*lam, 1e-6)))[None, :]).astype(np.float32))

cand = np.load("rot_candidates.npy")
def top1(F):
    o_ = np.zeros(len(cand), np.int64)
    for i in range(0, len(cand), 256):
        b = cand[i:i+256]; S = (F[b] @ F.T).astype(np.float32)
        S[np.arange(len(b)), b] = -2; o_[i:i+len(b)] = S.argmax(1)
    return o_
b1 = top1(X)
for blend in (8.0, 4.0):
    out = l2(np.hstack([phi, blend*l2(X)]).astype(np.float32))
    chg = (b1 != top1(out)).mean()*100
    f = f"submission_spec9728_b{int(blend)}.csv"
    # 9728 + 1024 = 10752 dims -- the width that already scored fine
    o2 = pd.DataFrame(np.round(out, 5), columns=[f"feature_{i}" for i in range(out.shape[1])])
    o2.insert(0, "image_name", names); o2["ID"] = names
    o2.to_csv(f, index=False)
    print(f"  blend {blend}: top-1 changed {chg:.1f}%  dims {out.shape[1]}  -> {f}", flush=True)
