"""Rotation correction applied to the 7B block itself.

The 0.95302 submission has NO rotation handling: 7B alone never had it, and
adding rot-ViT-L flipped zero scored items (284/298 three times). Meanwhile 44%
of queries are rotated by a multiple of 90 degrees and DINOv3 has no invariance
to it, so some of the 14 missing items are plausibly rotated queries.

Same gate as the 0.86912 recipe -- flip a candidate to its 180-degree view only
when that view beats upright by MARGIN -- but on the 7B block.
"""
import numpy as np, pandas as pd
from task1_concat import l2, whiten

MARGIN = 0.15
names = np.load('kout/lost-in-the-museum/feature_names.npy', allow_pickle=True)
cand  = np.load('rot_candidates.npy')
U   = l2(np.load('features_d7b512.npy').astype(np.float64))          # upright
R   = np.load('features_d7b512_rot180.npy').astype(np.float64)
R[cand] = l2(R[cand])
assert np.isfinite(U).all() and np.isfinite(R[cand]).all()

# ONE transform fitted on the upright corpus, applied to both views -- whitening
# each view separately puts them in different spaces and the similarities stop
# being comparable.
mu = U.mean(0, keepdims=True)
_, sv, vt = np.linalg.svd(U - mu, full_matrices=False)
D = 1536
sc = sv[:D] / np.sqrt(len(U) - 1) + 1e-8
proj = lambda z: l2((z - mu) @ vt[:D].T / sc).astype(np.float32)
W, W180 = proj(U), proj(R)
print('projected', W.shape, flush=True)

def top1(Q, ex):
    best = np.zeros(len(Q), np.float32)
    for i in range(0, len(Q), 256):
        S = Q[i:i+256] @ W.T
        for r in range(len(S)):
            S[r, ex[i+r]] = -1
        best[i:i+256] = S.max(1)
    return best

gain = top1(W180[cand], cand) - top1(W[cand], cand)
for m in (0.10, 0.15, 0.25, 0.40):
    print(f'  margin {m}: {(gain > m).sum()} candidates would flip')
flip = cand[gain > MARGIN]
print(f'flipping {len(flip)} at margin {MARGIN} (mean gain {gain[gain>MARGIN].mean():.3f})', flush=True)

Uc = U.copy(); Uc[flip] = R[flip]
f = whiten(l2(Uc), 8192, 1.0)
df = pd.DataFrame(np.round(f, 5), columns=[f'feature_{i}' for i in range(f.shape[1])])
df.insert(0, 'image_name', list(names)); df['ID'] = df['image_name']
assert len(df) == 20000 and not df.isna().any().any()
df.to_csv('submission_7b_rot_8192.csv', index=False)
print('wrote submission_7b_rot_8192.csv', df.shape)
