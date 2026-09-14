"""Pick each image's true orientation, then submit that -- do not average.

The obvious fix for the 44% of queries rotated by a multiple of 90 degrees is
to average every descriptor over all four orientations, which makes an
upside-down query and its upright gallery scan agree by construction. But it
also blurs the 10,000 clean gallery scans, and those are exactly what has to
stay separable from 9,000 same-artist distractors -- it spends the whitening
gain to buy the rotation gain.

We submit embeddings, not a matching function, but nothing says which
orientation an embedding has to describe. So resolve the orientation here
instead: score each candidate's four views against the un-rotated corpus and
keep whichever one actually lands on a painting. Full sharpness, right way up.

A candidate only leaves 0 degrees if a rotation beats it by MARGIN, so a
mis-flagged gallery scan costs nothing.
"""
import argparse
import numpy as np
import pandas as pd

ANGLES = (90, 180, 270)


def l2(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def fit_whiten(x, dim):
    """PCA + whitening fitted on the upright corpus, reused for every view.

    Every orientation has to land in the same space or the similarities are not
    comparable, so the transform is fitted once and applied, never re-fitted.
    """
    mu = x.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(x - mu, full_matrices=False)
    scale = s[:dim] / np.sqrt(len(x) - 1) + 1e-8
    return lambda z: l2((z - mu) @ vt[:dim].T / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='kout/lost-in-the-museum/features_g.npy')
    ap.add_argument('--names', default='kout/lost-in-the-museum/feature_names.npy')
    ap.add_argument('--cand', default='rot_candidates.npy')
    ap.add_argument('--rot', default='rot{}_g518.npy')
    ap.add_argument('--margin', type=float, default=0.02)
    ap.add_argument('--dim', type=int, default=1536)
    ap.add_argument('--mode', choices=['canonical', 'average'], default='canonical')
    ap.add_argument('--out', default='submission_canonical.csv')
    args = ap.parse_args()

    base = np.load(args.base).astype(np.float64)
    names = np.load(args.names, allow_pickle=True)
    cand = np.load(args.cand)
    print(f'corpus {base.shape}  candidates {len(cand)}')

    views = {0: base[cand]}
    for a in ANGLES:
        views[a] = np.load(args.rot.format(a)).astype(np.float64)
        assert len(views[a]) == len(cand), f'{a}deg has {len(views[a])} rows, expected {len(cand)}'

    if args.mode == 'average':
        # the blunt version, kept so the two can be compared on the leaderboard
        out = base.copy()
        out[cand] = l2(sum(l2(views[a]) for a in views))
    else:
        w = fit_whiten(l2(base), args.dim)
        ref = np.ones(len(base), bool)
        ref[cand] = False                    # match against gallery+distractors only
        R = w(l2(base))[ref].astype(np.float32)
        print(f'reference pool {R.shape[0]} images')

        sims = np.zeros((len(cand), 4), np.float32)
        for j, a in enumerate((0, *ANGLES)):
            Q = w(l2(views[a])).astype(np.float32)
            for i in range(0, len(cand), 512):
                sims[i:i+512, j] = (Q[i:i+512] @ R.T).max(1)

        pick = sims.argmax(1)
        gain = sims.max(1) - sims[:, 0]
        pick[gain < args.margin] = 0         # only move on clear evidence
        counts = np.bincount(pick, minlength=4)
        print('\nchosen orientation among candidates:')
        for j, a in enumerate((0, *ANGLES)):
            print(f'  {a:3d} deg : {counts[j]:5d}  ({counts[j]/len(cand):5.1%})')
        moved = pick > 0
        print(f'\nmoved {moved.sum()} of {len(cand)}; '
              f'mean similarity gain on those {gain[moved].mean():.4f}'
              if moved.any() else '\nnothing moved')

        out = base.copy()
        stacked = np.stack([views[a] for a in (0, *ANGLES)])   # 4 x C x D
        out[cand] = stacked[pick, np.arange(len(cand))]
        np.save('rot_choice.npy', pick)

    x = l2(out)
    mu = x.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(x - mu, full_matrices=False)
    d = args.dim
    f = l2((x - mu) @ vt[:d].T / (s[:d] / np.sqrt(len(x) - 1) + 1e-8)).astype(np.float32)
    print(f'\nPCA {out.shape[1]} -> {d}, explains {(s[:d]**2).sum()/(s**2).sum():.1%}')

    df = pd.DataFrame(f, columns=[f'feature_{i}' for i in range(d)])
    df.insert(0, 'image_name', list(names))
    df['ID'] = df['image_name']
    df.to_csv(args.out, index=False, float_format='%.6f')
    print(f'wrote {args.out}: {len(df)} rows x {df.shape[1]} cols')


if __name__ == '__main__':
    main()
