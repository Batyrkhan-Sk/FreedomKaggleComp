"""Test whether concatenating squash- and pad-view embeddings beats either alone.

The two views distort an image differently, so they tend to fail on different
queries. Concatenating their (individually L2-normalised) embeddings is the
standard cheap ensemble -- and it needs no new GPU pass, since both feature
files already exist.

Crucially the *same* degraded image must feed both views, so the degradation is
generated once as a PIL image and only then finished two different ways.
"""

import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from extract import IMAGENET_MEAN, IMAGENET_STD, PadSquare
from extract import DATA_DIR
from validate import Downscale, RandomCropAspect, embed, l2


def degrade_pil(strength):
    """Everything up to (but excluding) the aspect handling -- returns a PIL image."""
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


def finish(size, aspect):
    steps = [PadSquare()] if aspect == "pad" else []
    steps += [
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=392)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--strength", type=float, default=1.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    g_squash = l2(np.load("features_l392.npy").astype(np.float64))
    g_pad = l2(np.load("features_l392pad.npy").astype(np.float64))
    names = np.load("features_l392_names.npy", allow_pickle=True)
    gallery = np.concatenate([g_squash, g_pad], axis=1)
    print(f"Ensemble gallery: {gallery.shape}")

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(names), size=args.n, replace=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False).eval().to(device)

    base = degrade_pil(args.strength)
    fin_s, fin_p = finish(args.size, "squash"), finish(args.size, "pad")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ts, tp = [], []
    for i in picks:
        img = base(Image.open(DATA_DIR / names[i]).convert("RGB"))
        ts.append(fin_s(img))
        tp.append(fin_p(img))

    q = np.concatenate([
        l2(embed(model, ts, device, "cls_gem").astype(np.float64)),
        l2(embed(model, tp, device, "cls_gem").astype(np.float64)),
    ], axis=1)

    mu = gallery.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(gallery - mu, full_matrices=False)
    comps = vt[: args.dim]
    scale = s[: args.dim] / np.sqrt(len(gallery) - 1)
    gal = l2((gallery - mu) @ comps.T / (scale + 1e-8))
    qq = l2((q - mu) @ comps.T / (scale + 1e-8))

    order = np.argsort(-(qq @ gal.T), axis=1)
    hit3 = np.mean([picks[i] in order[i, :3] for i in range(len(picks))])
    hit1 = np.mean([picks[i] == order[i, 0] for i in range(len(picks))])
    hit10 = np.mean([picks[i] in order[i, :10] for i in range(len(picks))])
    rank = np.array([int(np.where(order[i] == picks[i])[0][0]) + 1 for i in range(len(picks))])

    print(f"\nsquash+pad ensemble @{args.size} dim={args.dim} strength={args.strength}")
    print(f"  Hit@1  {hit1:.3f}")
    print(f"  Hit@3  {hit3:.3f}   <-- competition metric")
    print(f"  Hit@10 {hit10:.3f}")
    print(f"  median rank {np.median(rank):.0f}   mean rank {rank.mean():.1f}")


if __name__ == "__main__":
    main()
