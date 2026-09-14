"""Multi-scale ensemble: concatenate embeddings of the same image at two resolutions.

The earlier squash+pad ensemble lost because one of its two views (squash) was
clearly weaker, and concatenation gave it equal weight. Here both views come
from the identical pipeline and differ only in input resolution, so neither is
a passenger -- and they fail on different queries, which is what an ensemble
needs to pay off.

The same degraded image feeds both scales, so the comparison is honest.
"""

import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from extract import DATA_DIR, IMAGENET_MEAN, IMAGENET_STD
from validate import Downscale, RandomCropAspect, embed, l2


def degrade_pil(strength):
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=12 * strength,
            translate=(0.04 * strength, 0.04 * strength),
            scale=(1 - 0.15 * strength, 1 + 0.05 * strength),
            shear=6 * strength,
            fill=235,
        ),
        transforms.RandomPerspective(distortion_scale=0.22 * strength, p=0.85, fill=235),
        RandomCropAspect(scale=(0.55, 0.95), ratio=(0.7, 1.4)),
        transforms.ColorJitter(
            brightness=0.3 * strength, contrast=0.3 * strength,
            saturation=0.25 * strength, hue=0.03 * strength,
        ),
        transforms.GaussianBlur(kernel_size=7, sigma=(0.4 * strength + 0.2, 2.4 * strength + 0.2)),
        Downscale(0, factor=1.0 + 2.5 * strength),
    ])


def finish(size):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="features_l392")
    ap.add_argument("--a-size", type=int, default=392)
    ap.add_argument("--b", default="features_l518")
    ap.add_argument("--b-size", type=int, default=518)
    ap.add_argument("--weight-b", type=float, default=1.0, help="scale factor on view B before concat")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--strength", type=float, default=1.25)
    ap.add_argument("--per-view", action="store_true",
                    help="whiten each view separately then concat (avoids joint-whitening blowing up\n                          the low-variance directions where two correlated views disagree)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ga = l2(np.load(f"{args.a}.npy").astype(np.float64))
    gb = l2(np.load(f"{args.b}.npy").astype(np.float64)) * args.weight_b
    names = np.load(f"{args.a}_names.npy", allow_pickle=True)
    gallery = np.concatenate([ga, gb], axis=1)
    print(f"Gallery {gallery.shape}  (weight_b={args.weight_b})")

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(names), size=args.n, replace=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False).eval().to(device)

    base, fa, fb = degrade_pil(args.strength), finish(args.a_size), finish(args.b_size)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ta, tb = [], []
    for i in picks:
        img = base(Image.open(DATA_DIR / names[i]).convert("RGB"))
        ta.append(fa(img))
        tb.append(fb(img))

    q = np.concatenate([
        l2(embed(model, ta, device, "cls_gem").astype(np.float64)),
        l2(embed(model, tb, device, "cls_gem").astype(np.float64)) * args.weight_b,
    ], axis=1)

    if args.per_view:
        half = args.dim // 2
        gal_parts, q_parts = [], []
        for g, qv in ((ga, q[:, : ga.shape[1]]), (gb, q[:, ga.shape[1] :])):
            m = g.mean(axis=0, keepdims=True)
            _, sv, vtv = np.linalg.svd(g - m, full_matrices=False)
            sc = sv[:half] / np.sqrt(len(g) - 1)
            gal_parts.append(l2((g - m) @ vtv[:half].T / (sc + 1e-8)))
            q_parts.append(l2((qv - m) @ vtv[:half].T / (sc + 1e-8)))
        gal = l2(np.concatenate(gal_parts, axis=1))
        qq = l2(np.concatenate(q_parts, axis=1))
        order = np.argsort(-(qq @ gal.T), axis=1)
        print(f"per-view whitening: {half}+{half} = {gal.shape[1]} dims")
        for label, k in (("Hit@1", 1), ("Hit@3", 3), ("Hit@10", 10)):
            hit = np.mean([picks[i] in order[i, :k] for i in range(len(picks))])
            print(f"  {label:6s} {hit:.3f}" + ("   <-- competition metric" if k == 3 else ""))
        return

    mu = gallery.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(gallery - mu, full_matrices=False)
    scale = s[: args.dim] / np.sqrt(len(gallery) - 1)
    gal = l2((gallery - mu) @ vt[: args.dim].T / (scale + 1e-8))
    qq = l2((q - mu) @ vt[: args.dim].T / (scale + 1e-8))

    order = np.argsort(-(qq @ gal.T), axis=1)
    for label, k in (("Hit@1", 1), ("Hit@3", 3), ("Hit@10", 10)):
        hit = np.mean([picks[i] in order[i, :k] for i in range(len(picks))])
        print(f"  {label:6s} {hit:.3f}" + ("   <-- competition metric" if k == 3 else ""))


if __name__ == "__main__":
    main()
