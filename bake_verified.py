"""Turn LoFTR's geometric verification into a submission.

`kaggle-loftr-verify.ipynb` produces verified.npz -- for each shortlist candidate,
the gallery image with the most homography inliers and how many. This encodes that
into the embedding: a query confidently verified against gallery `g` is pulled
toward `g`, so the host's cosine ranking returns `g`.

Three rules, each learned the expensive way on this competition:

1. **Only bake pairs that DISAGREE with the descriptor top-1.** Baking a query
   toward the image cosine already ranks first cannot change Hit@3 -- the host
   returns it either way. Two no-op submissions were built before this was
   understood. Only disagreements can move the score.

2. **Blend, don't overwrite.** Scoring is Hit@3. A hard overwrite guarantees the
   hit when the match is right and guarantees a miss when it is wrong; a blend
   keeps enough of the query's own signal that a wrong pairing can still leave
   the true match inside the top 3. Asymmetric payoff, so blend.

3. **Start at the TIGHT end of the gate.** Every confidence gate widened on this
   competition has lost -- rotation flips 107->362 (-0.0034), pitcher margin
   2.0->8.0 (-1 question). Items just outside the tightest band are ones the base
   method had a real reason for.

Usage:
    python3 bake_verified.py --inliers 30 --lam 1.0 --out submission_baked30.csv
    python3 bake_verified.py --dry-run          # just print the gate table
"""
import argparse

import numpy as np
import pandas as pd

from task1_concat import l2, whiten


def build_base(margin: float):
    """The 0.86912 space: ViT-g@518 + rotation-corrected ViT-L@518, full width."""
    g = l2(np.load('kout/lost-in-the-museum/features_g.npy').astype(np.float64))
    L0 = l2(np.load('features_l518.npy').astype(np.float64))
    L180 = l2(np.load('features_l518_rot180.npy').astype(np.float64))
    names = np.load('kout/lost-in-the-museum/feature_names.npy', allow_pickle=True)
    cand = np.load('rot_candidates.npy')

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

    Lc = L0.copy()
    gain = top1(W180[cand], cand) - top1(W[cand], cand)
    flip = cand[gain > margin]
    Lc[flip] = L180[flip]
    print(f'flipped {len(flip)} candidates (margin {margin})')
    return np.hstack([l2(g), l2(Lc)]), names


def descriptor_top1(f, qs):
    """What the host would return today, so we can tell agreement from disagreement."""
    out = np.zeros(len(qs), np.int64)
    for i in range(0, len(qs), 256):
        S = f[qs[i:i+256]] @ f.T
        for r, q in enumerate(qs[i:i+256]):
            S[r, q] = -1
        out[i:i+256] = S.argmax(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verified', default='verified.npz')
    ap.add_argument('--dim', type=int, default=5120)
    ap.add_argument('--margin', type=float, default=0.15)
    ap.add_argument('--inliers', type=int, default=30,
                    help='minimum homography inliers to act on. Start TIGHT.')
    ap.add_argument('--lam', type=float, default=1.0,
                    help='blend weight toward the matched gallery vector; '
                         'f[q] <- l2(f[q] + lam * f[g])')
    ap.add_argument('--precision', type=int, default=6)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--out', default='submission_baked.csv')
    args = ap.parse_args()

    v = np.load(args.verified)
    cand, best_g, best_n = v['cand'], v['best_g'], v['best_n']
    print(f'{len(cand)} verified candidates, inliers '
          f'p50 {np.median(best_n):.0f} p90 {np.percentile(best_n,90):.0f} '
          f'max {best_n.max()}')

    x, names = build_base(args.margin)
    f = whiten(x, args.dim, 1.0)
    d1 = descriptor_top1(f, cand)

    print(f'\n{"inliers":>8} {"confident":>10} {"DISAGREE":>10}  <- only disagreements can move the score')
    for t in (10, 15, 20, 30, 50, 75, 100):
        sel = best_n >= t
        print(f'{t:8d} {int(sel.sum()):10d} {int((sel & (best_g != d1)).sum()):10d}')

    take = (best_n >= args.inliers) & (best_g != d1)
    print(f'\nbaking {int(take.sum())} queries at inliers>={args.inliers}, lam={args.lam}')
    if args.dry_run:
        return
    if not take.sum():
        raise SystemExit('nothing to bake -- lower --inliers or check verified.npz')

    q, g = cand[take], best_g[take]
    f = f.copy()
    f[q] = l2(f[q] + args.lam * f[g]).astype(np.float32)

    df = pd.DataFrame(f, columns=[f'feature_{i}' for i in range(f.shape[1])])
    df.insert(0, 'image_name', list(names))
    df['ID'] = df['image_name']
    df.to_csv(args.out, index=False, float_format=f'%.{args.precision}f')
    print(f'wrote {args.out}: {len(df)} rows x {df.shape[1]} cols')


if __name__ == '__main__':
    main()
