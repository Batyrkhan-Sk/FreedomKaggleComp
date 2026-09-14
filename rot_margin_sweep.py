"""Where should the rotation-flip gate sit?

Measured on the leaderboard: margin 0.15 (107 flips) gained +0.007, margin 0.08
(362 flips) LOST 0.0034 against it. Widening hurts. Tightening was never tested,
and on this competition tightening a confidence gate has never once lost.

This prints the flip count and the gain distribution at each threshold, so the
next submission is chosen from the shape of the data rather than another guess.
"""
import numpy as np

l2 = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

L0 = l2(np.load('features_l518.npy').astype(np.float64))
L180 = l2(np.load('features_l518_rot180.npy').astype(np.float64))
cand = np.load('rot_candidates.npy')
print(f'{len(cand)} rotation candidates of {len(L0)} images', flush=True)

# One transform fitted on the upright corpus, applied to both views -- whitening
# each view separately puts them in different spaces and the similarities stop
# meaning anything.
mu = L0.mean(0, keepdims=True)
_, sv, vt = np.linalg.svd(L0 - mu, full_matrices=False)
sc = sv[:1536] / np.sqrt(len(L0) - 1) + 1e-8
proj = lambda z: l2((z - mu) @ vt[:1536].T / sc).astype(np.float32)
W, W180 = proj(L0), proj(L180)


def top1(Q, ex):
    best = np.zeros(len(Q), np.float32)
    for i in range(0, len(Q), 256):
        S = Q[i:i+256] @ W.T
        for r in range(len(S)):
            S[r, ex[i+r]] = -1
        best[i:i+256] = S.max(1)
    return best


s0, s1 = top1(W[cand], cand), top1(W180[cand], cand)
gain = s1 - s0
print(f'\ngain percentiles: p50 {np.percentile(gain,50):.3f}  '
      f'p90 {np.percentile(gain,90):.3f}  p99 {np.percentile(gain,99):.3f}  '
      f'max {gain.max():.3f}')
print(f'\n{"margin":>8} {"flips":>7} {"mean gain":>10}  note')
for m in (0.08, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40):
    sel = gain > m
    n = int(sel.sum())
    mg = gain[sel].mean() if n else float('nan')
    note = ''
    if abs(m - 0.15) < 1e-9:
        note = '<- current best, +0.007 on LB'
    elif abs(m - 0.08) < 1e-9:
        note = '<- tested, LOST 0.0034'
    print(f'{m:8.2f} {n:7d} {mg:10.3f}  {note}')
np.save('rot_gain.npy', gain)
print('\nsaved per-candidate gains to rot_gain.npy')
