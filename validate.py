"""Local Hit@3 proxy -- the competition ships no labels, so we make our own.

The query images appear to be synthetically degraded copies of gallery images
(rotation with pale padding, blur, crop, downscale). So: take N images from the
dataset, apply the same kind of degradation ourselves, embed the result, and
check whether the untouched original comes back in the top 3 of all 20,000.

That is structurally identical to what the host scores, so it lets us compare
models and settings without burning leaderboard submissions.
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from extract import DATA_DIR, IMAGENET_MEAN, IMAGENET_STD, PadSquare, gem_pool

Image.MAX_IMAGE_PIXELS = None


class Downscale:
    """Simulate a low-resolution capture: shrink hard, then upsample back.
    The real queries are described as low-resolution, and this destroys
    high-frequency detail in a way blur alone does not."""

    REFERENCE = 224  # degradation is defined in absolute pixels, not model-input pixels

    def __init__(self, size, factor):
        self.size, self.factor = size, factor

    def __call__(self, img):
        # A real query has a fixed capture resolution no matter what size we
        # feed the backbone. Anchoring to REFERENCE keeps 224px and 392px runs
        # comparable -- otherwise larger inputs silently get easier queries.
        # Aspect ratio is preserved: squaring here would pre-empt PadSquare.
        w, h = img.size
        target_long = max(16, self.REFERENCE / self.factor)
        scale = min(1.0, target_long / max(w, h))
        sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
        return img.resize((sw, sh), Image.BILINEAR).resize((w, h), Image.BICUBIC)


class RandomCropAspect:
    """Crop a random sub-region WITHOUT forcing a square result.

    torchvision's RandomResizedCrop always returns a square, which would mask
    the aspect-ratio effect we are trying to measure: a real cropped query has
    its own aspect ratio, different from the gallery original's.
    """

    def __init__(self, scale=(0.55, 0.95), ratio=(0.7, 1.4)):
        self.scale, self.ratio = scale, ratio

    def __call__(self, img):
        w, h = img.size
        area = w * h * np.random.uniform(*self.scale)
        ar = np.exp(np.random.uniform(np.log(self.ratio[0]), np.log(self.ratio[1])))
        cw = min(w, max(8, int(round(np.sqrt(area * ar)))))
        ch = min(h, max(8, int(round(np.sqrt(area / ar)))))
        x = np.random.randint(0, w - cw + 1)
        y = np.random.randint(0, h - ch + 1)
        return img.crop((x, y, x + cw, y + ch))


def degrade(size, strength=1.0, aspect="squash"):
    """Approximate the visitor-photo domain gap."""
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=12 * strength,
            translate=(0.04 * strength, 0.04 * strength),
            scale=(1 - 0.15 * strength, 1 + 0.05 * strength),
            shear=6 * strength,
            fill=235,  # pale padding, matching what we saw in real queries
        ),
        transforms.RandomPerspective(distortion_scale=0.22 * strength, p=0.85, fill=235),
        RandomCropAspect(scale=(0.55, 0.95), ratio=(0.7, 1.4)),
        transforms.ColorJitter(
            brightness=0.3 * strength, contrast=0.3 * strength,
            saturation=0.25 * strength, hue=0.03 * strength,
        ),
        transforms.GaussianBlur(kernel_size=7, sigma=(0.4 * strength + 0.2, 2.4 * strength + 0.2)),
        Downscale(size, factor=1.0 + 2.5 * strength),
        *( [PadSquare()] if aspect == "pad" else [] ),
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def embed(model, tensors, device, pool, batch=16, fp16=False):
    outs = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch):
            b = torch.stack(tensors[i : i + batch]).to(device)
            if fp16:
                b = b.half()
            o = model.forward_features(b)
            cls, patches = o["x_norm_clstoken"], o["x_norm_patchtokens"]
            if pool == "cls":
                v = cls
            elif pool == "gem":
                v = gem_pool(patches)
            else:
                v = torch.cat([F.normalize(cls, dim=1), F.normalize(gem_pool(patches), dim=1)], dim=1)
            outs.append(v.float().cpu())
    return torch.cat(outs).numpy()


def l2(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def expand(x, ref, k, alpha, exclude_self, chunk=2000):
    """Similarity-weighted neighbour blending (alpha-QE).

    Chunked over rows: the full 20,000^2 similarity matrix would be several GB,
    and we only ever need the top-k of each row.
    """
    xf, reff = x.astype(np.float32), ref.astype(np.float32)
    out = np.empty_like(xf)
    for start in range(0, len(xf), chunk):
        stop = min(start + chunk, len(xf))
        sims = xf[start:stop] @ reff.T
        if exclude_self:
            for r in range(stop - start):
                sims[r, start + r] = -np.inf
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        rows = np.arange(stop - start)[:, None]
        w = np.clip(sims[rows, idx], 0, None) ** alpha
        out[start:stop] = xf[start:stop] + (w[:, :, None] * reff[idx]).sum(axis=1)
    return l2(out.astype(np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_l224")
    ap.add_argument("--model", default="dinov2_vitl14")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--pool", default="cls_gem")
    ap.add_argument("--n", type=int, default=200, help="number of synthetic queries")
    ap.add_argument("--dim", type=int, default=0, help="PCA dim to simulate (0 = full)")
    ap.add_argument("--whiten", action="store_true", default=True)
    ap.add_argument("--no-whiten", dest="whiten", action="store_false")
    ap.add_argument("--strength", type=float, default=1.0, help="degradation severity")
    ap.add_argument("--dba-k", type=int, default=0, help="neighbours for expansion (0 = off)")
    ap.add_argument("--dba-alpha", type=float, default=3.0)
    ap.add_argument("--aspect", default="squash", choices=["squash", "pad"])
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gallery = l2(np.load(f"{args.features}.npy").astype(np.float64))
    names = np.load(f"{args.features}_names.npy", allow_pickle=True)
    print(f"Gallery: {gallery.shape}")

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(names), size=args.n, replace=False)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", args.model, verbose=False).eval().to(device)
    if args.fp16:
        model = model.half()  # ViT-g needs half precision to fit in 16GB unified memory

    tf = degrade(args.size, args.strength, args.aspect)
    torch.manual_seed(args.seed)
    tensors = [tf(Image.open(DATA_DIR / names[i]).convert("RGB")) for i in picks]
    q = l2(embed(model, tensors, device, args.pool, args.batch, args.fp16).astype(np.float64))

    if args.dim and args.dim < gallery.shape[1]:
        # Fit the reduction on the gallery only, then apply to both sides --
        # exactly what make_submission.py does to the real submission.
        mu = gallery.mean(axis=0, keepdims=True)
        _, s, vt = np.linalg.svd(gallery - mu, full_matrices=False)
        comps = vt[: args.dim]
        scale = s[: args.dim] / np.sqrt(len(gallery) - 1) if args.whiten else 1.0
        gallery = l2((gallery - mu) @ comps.T / (scale + 1e-8 if args.whiten else 1.0))
        q = l2((q - mu) @ comps.T / (scale + 1e-8 if args.whiten else 1.0))
        print(f"PCA -> {args.dim} (whiten={args.whiten})")

    if args.dba_k:
        # Database-side augmentation / alpha-query-expansion.
        # Each embedding absorbs a similarity-weighted blend of its nearest
        # neighbours, which denoises it toward the local manifold. We control
        # every one of the 20,000 rows we submit, so the host's queries get
        # this treatment too -- pulling a degraded query toward the cluster
        # its true match sits in.
        gallery = expand(gallery, gallery, args.dba_k, args.dba_alpha, exclude_self=True)
        q = expand(q, gallery, args.dba_k, args.dba_alpha, exclude_self=False)
        print(f"DBA k={args.dba_k} alpha={args.dba_alpha}")

    sims = q @ gallery.T
    order = np.argsort(-sims, axis=1)

    hit1 = np.mean([picks[i] == order[i, 0] for i in range(len(picks))])
    hit3 = np.mean([picks[i] in order[i, :3] for i in range(len(picks))])
    hit10 = np.mean([picks[i] in order[i, :10] for i in range(len(picks))])
    rank = np.array([int(np.where(order[i] == picks[i])[0][0]) + 1 for i in range(len(picks))])

    print(f"\n{args.model} @{args.size} pool={args.pool} strength={args.strength} n={args.n}")
    print(f"  Hit@1  {hit1:.3f}")
    print(f"  Hit@3  {hit3:.3f}   <-- competition metric")
    print(f"  Hit@10 {hit10:.3f}")
    print(f"  median rank {np.median(rank):.0f}   mean rank {rank.mean():.1f}")


if __name__ == "__main__":
    main()
