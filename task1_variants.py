"""Sweep the free knobs on the existing ViT-g descriptors -- no GPU.

0.8285 came from one fixed recipe: L2, PCA to 1536, full whitening. Three
standard retrieval levers were never tried, and all of them are pure CPU
arithmetic on the 20000x3072 matrix already on disk:

  * whitening strength. Dividing by the singular values exactly (alpha=1) also
    amplifies the smallest, noisiest directions. alpha around 0.5-0.7 is the
    usual sweet spot and the notes only ever tested alpha=1 against alpha=0.
  * alpha-query-expansion. Replace each descriptor with a weighted blend of
    itself and its nearest neighbours. This is the standard transductive trick
    for exactly this setting, and it is legal here because we submit vectors:
    whatever we bake in is what the host cosine-matches.
  * output dimension, swept at fixed whitening rather than confounded with it.

Scoring is cosine on whatever we submit, so every one of these is fair game.
"""
import argparse
import numpy as np
import pandas as pd


def l2(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def whiten(x, dim, alpha):
    mu = x.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(x - mu, full_matrices=False)
    sc = (s[:dim] / np.sqrt(len(x) - 1)) ** alpha + 1e-8
    return l2((x - mu) @ vt[:dim].T / sc).astype(np.float32)


def alpha_qe(x, k, alpha):
    """Blend each descriptor with its k nearest neighbours, weighted by sim^alpha.

    Done after whitening, because the neighbours have to be found in the space
    the host will actually match in.
    """
    out = np.empty_like(x)
    for i in range(0, len(x), 512):
        S = x[i:i + 512] @ x.T
        for r in range(len(S)):
            S[r, i + r] = -1                       # never blend with self
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        w = np.take_along_axis(S, idx, 1).clip(min=0) ** alpha
        nb = (x[idx] * w[:, :, None]).sum(1)
        out[i:i + 512] = x[i:i + 512] + nb
    return l2(out).astype(np.float32)


def proxy_score(x, cand):
    """Label-free stand-in: mutual nearest neighbours between candidates and rest.

    A query and its gallery original picking each other out of 20,000 images is
    almost certainly a true pair, so more mutual pairs is better retrieval. It
    is noisy -- it moved only +15 on a change worth far more -- so treat it as a
    filter against obviously bad variants, not as a ranking.
    """
    ref = np.ones(len(x), bool)
    ref[cand] = False
    R, ridx = x[ref], np.flatnonzero(ref)
    bj = np.zeros(len(cand), np.int64)
    for i in range(0, len(cand), 512):
        bj[i:i + 512] = ridx[(x[cand[i:i + 512]] @ R.T).argmax(1)]
    uniq = np.unique(bj)
    back = {}
    for i in range(0, len(uniq), 512):
        S = x[uniq[i:i + 512]] @ x[cand].T
        for r, row in enumerate(S):
            back[uniq[i + r]] = cand[row.argmax()]
    return sum(back.get(bj[k]) == cand[k] for k in range(len(cand)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default='kout/lost-in-the-museum/features_g.npy')
    ap.add_argument('--names', default='kout/lost-in-the-museum/feature_names.npy')
    ap.add_argument('--dim', type=int, default=1536)
    ap.add_argument('--alpha', type=float, default=1.0, help='whitening exponent')
    ap.add_argument('--qe-k', type=int, default=0, help='0 disables query expansion')
    ap.add_argument('--qe-alpha', type=float, default=3.0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--sweep', action='store_true')
    args = ap.parse_args()

    raw = l2(np.load(args.feats).astype(np.float64))
    names = np.load(args.names, allow_pickle=True)
    cand = np.load('rot_candidates.npy')
    print(f'{raw.shape} descriptors, {len(cand)} query candidates')

    if args.sweep:
        print(f'\n{"dim":>5} {"white":>6} {"qe_k":>5}   mutual-NN proxy')
        for dim in (1024, 1536, 3072):
            for al in (0.5, 0.7, 1.0):
                x = whiten(raw, dim, al)
                print(f'{dim:5d} {al:6.1f} {0:5d}   {proxy_score(x, cand)}', flush=True)
        x = whiten(raw, args.dim, 1.0)
        for k in (2, 3, 5):
            print(f'{args.dim:5d} {1.0:6.1f} {k:5d}   '
                  f'{proxy_score(alpha_qe(x, k, args.qe_alpha), cand)}', flush=True)
        return

    x = whiten(raw, args.dim, args.alpha)
    if args.qe_k:
        x = alpha_qe(x, args.qe_k, args.qe_alpha)
    print(f'proxy {proxy_score(x, cand)}')
    if args.out:
        df = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
        df.insert(0, 'image_name', list(names))
        df['ID'] = df['image_name']
        df.to_csv(args.out, index=False, float_format='%.6f')
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
