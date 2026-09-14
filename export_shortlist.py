"""Export each query candidate's top-K gallery guesses for offline verification.

The descriptor already puts the right gallery image in the top few for most
queries -- what it cannot do is tell which of those few is right, because a
Monet bridge looks like another Monet bridge to a global vector. Geometric
verification can, but only if it is given a short list to check.

So this hands LoFTR a shortlist and lets it adjudicate: ~1,000 candidates x 20
guesses is 20k pair checks, cheap next to a 20,000-image backbone pass.
"""
import argparse
import numpy as np


def l2(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--out', default='shortlist.npz')
    args = ap.parse_args()

    # rebuild the 0.86912 space: ViT-g @518 + rotation-corrected ViT-L @518,
    # per-block L2, whitened at full width
    g = l2(np.load('kout/lost-in-the-museum/features_g.npy').astype(np.float64))
    L0 = l2(np.load('features_l518.npy').astype(np.float64))
    L180 = l2(np.load('features_l518_rot180.npy').astype(np.float64))
    cand = np.load('rot_candidates.npy')
    names = np.load('kout/lost-in-the-museum/feature_names.npy', allow_pickle=True)

    mu = L0.mean(0, keepdims=True)
    _, sv, vt = np.linalg.svd(L0 - mu, full_matrices=False)
    proj = lambda z: l2((z - mu) @ vt[:1536].T /
                        (sv[:1536] / np.sqrt(len(L0) - 1) + 1e-8)).astype(np.float32)
    W, W180 = proj(L0), proj(L180)

    def top1(Q, ex):
        b = np.zeros(len(Q), np.float32)
        for i in range(0, len(Q), 256):
            S = Q[i:i+256] @ W.T
            for r in range(len(S)):
                S[r, ex[i+r]] = -1
            b[i:i+256] = S.max(1)
        return b

    s0, s1 = top1(W[cand], cand), top1(W180[cand], cand)
    Lc = L0.copy()
    Lc[cand[(s1 - s0) > 0.15]] = L180[cand[(s1 - s0) > 0.15]]

    x = np.hstack([l2(g), l2(Lc)])
    m = x.mean(0, keepdims=True)
    _, s, v2 = np.linalg.svd(x - m, full_matrices=False)
    X = l2((x - m) @ v2[:5120].T / (s[:5120] / np.sqrt(len(x) - 1) + 1e-8)).astype(np.float32)

    ref = np.ones(len(X), bool)
    ref[cand] = False
    ridx = np.flatnonzero(ref)
    topk = np.zeros((len(cand), args.k), np.int64)
    for i in range(0, len(cand), 256):
        S = X[cand[i:i+256]] @ X[ridx].T
        idx = np.argpartition(-S, args.k, axis=1)[:, :args.k]
        order = np.take_along_axis(S, idx, 1).argsort(1)[:, ::-1]
        topk[i:i+256] = ridx[np.take_along_axis(idx, order, 1)]

    np.savez('shortlist.npz' if args.out is None else args.out,
             cand=cand, topk=topk, names=names)
    print(f'wrote {args.out}: {len(cand)} candidates x {args.k} guesses')
    print(f'  candidate 0 = {names[cand[0]]} -> {[names[j] for j in topk[0][:5]]}')


if __name__ == '__main__':
    main()
