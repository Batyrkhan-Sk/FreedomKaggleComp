"""Build a spectral submission from cached eigenpairs, and report how much it
actually CHANGES retrieval before a submission slot is spent on it.

The LoFTR lesson: a method can work perfectly and still be worth zero if it
agrees with the descriptor everywhere it is confident. The analogue here is the
top-1 change rate -- near 0% means we are shipping the same ranking in a new
basis; near 100% means we are betting the whole geometry on one submission.
"""
import argparse, numpy as np
from task1_concat import l2

ap = argparse.ArgumentParser()
ap.add_argument('--eigs', default='eigs_k50_g3.npz')
ap.add_argument('--rank', type=int, default=2048)
ap.add_argument('--alpha', type=float, default=0.99)
ap.add_argument('--blend', type=float, default=0.0)
ap.add_argument('--precision', type=int, default=6)
ap.add_argument('--out', default='')
a = ap.parse_args()

d = np.load(a.eigs); lam, U = d['lam'][:a.rank], d['U'][:, :a.rank]
X = np.load('base5120.npy')
names = np.load('kout/lost-in-the-museum/feature_names.npy', allow_pickle=True)
cand = np.load('rot_candidates.npy')

scale = 1.0 / np.sqrt(np.maximum(1.0 - a.alpha * lam, 1e-6))
phi = l2((U * scale[None, :]).astype(np.float32))
out = phi if a.blend <= 0 else l2(np.hstack([phi, a.blend * l2(X)]).astype(np.float32))

def top3(F, qs):
    t1 = np.zeros(len(qs), np.int64); t3 = np.zeros((len(qs), 3), np.int64)
    for i in range(0, len(qs), 256):
        b = qs[i:i+256]
        S = (F[b] @ F.T).astype(np.float32)
        S[np.arange(len(b)), b] = -2
        idx = np.argpartition(-S, 3, axis=1)[:, :3]
        v = np.take_along_axis(S, idx, axis=1)
        srt = np.take_along_axis(idx, np.argsort(-v, axis=1), axis=1)
        t3[i:i+len(b)] = srt; t1[i:i+len(b)] = srt[:, 0]
    return t1, t3

b1, b3 = top3(X, cand)
s1, s3 = top3(out, cand)
chg = (b1 != s1).mean() * 100
ov = np.mean([len(set(x) & set(y)) for x, y in zip(b3, s3)])
print(f"rank={a.rank:5d} alpha={a.alpha:.3f} blend={a.blend:.2f} dim={out.shape[1]:5d}  "
      f"top1 changed {chg:5.1f}%   mean top3 overlap {ov:.2f}/3")

if a.out:
    import pandas as pd
    df = pd.DataFrame(np.round(out, a.precision),
                      columns=[f'feature_{i}' for i in range(out.shape[1])])
    df.insert(0, 'image_name', names); df['ID'] = names
    df.to_csv(a.out, index=False)
    print(f"  wrote {a.out}: {len(df)} rows x {df.shape[1]} cols")
