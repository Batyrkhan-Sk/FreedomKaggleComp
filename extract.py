"""Extract DINOv2 embeddings for every image in the Lost in the Museum dataset.

Writes raw (un-reduced) features to a .npy plus a parallel list of image names,
so that dimensionality reduction and submission formatting can be re-run
without paying for a second GPU pass.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None  # a few gallery scans are enormous

DATA_DIR = Path.home() / "Downloads/lost-in-the-museum-f1/archive/kaggle_dataset/kaggle_dataset"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class PadSquare:
    """Letterbox to a square, preserving aspect ratio.

    Squashing a non-square image to a square warps it by a factor that depends
    on its own aspect ratio. A cropped query and its uncropped gallery original
    have *different* aspect ratios, so squashing warps them differently and we
    end up matching partly on distortion artefacts. Padding keeps the geometry
    of both intact.
    """

    def __init__(self, fill=235):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (self.fill,) * 3)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas


class ImageFolder(Dataset):
    def __init__(self, paths, size, aspect="squash"):
        self.paths = paths
        self.size = size
        steps = []
        if aspect == "pad":
            steps.append(PadSquare())
        steps += [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        self.tf = transforms.Compose(steps)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            img = Image.open(path).convert("RGB")
            return self.tf(img), i, True
        except Exception as e:  # noqa: BLE001
            print(f"  ! failed {path.name}: {e}", flush=True)
            # Never drop a row: the submission must carry all 20,000 images.
            return torch.zeros(3, self.size, self.size), i, False


def gem_pool(patch_tokens, p=3.0, eps=1e-6):
    """Generalised-mean pooling over patch tokens -- stronger than plain mean
    for instance retrieval, since it emphasises the most distinctive regions."""
    x = patch_tokens.clamp(min=eps).pow(p)
    return x.mean(dim=1).pow(1.0 / p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dinov2_vitl14", help="dinov2_vits14|vitb14|vitl14|vitg14")
    ap.add_argument("--size", type=int, default=224, help="input resolution (multiple of 14)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only process first N images (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pool", default="cls_gem", choices=["cls", "gem", "cls_gem"])
    ap.add_argument("--aspect", default="squash", choices=["squash", "pad"])
    ap.add_argument("--fp16", action="store_true", help="half-precision weights (halves memory)")
    ap.add_argument("--out", default="features")
    args = ap.parse_args()

    assert args.size % 14 == 0, "DINOv2 uses 14px patches; size must be a multiple of 14"

    paths = sorted(DATA_DIR.glob("*.png"))
    if args.limit:
        paths = paths[: args.limit]
    print(f"Images: {len(paths)}  (from {DATA_DIR})")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}  Model: {args.model}  Size: {args.size}  Pool: {args.pool}  Aspect: {args.aspect}")

    model = torch.hub.load("facebookresearch/dinov2", args.model, verbose=False)
    model.eval().to(device)
    if args.fp16:
        model = model.half()  # ViT-g in fp32 exceeds 16GB unified memory and swap-thrashes

    loader = DataLoader(
        ImageFolder(paths, args.size, args.aspect),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    feats = None
    failed = []
    done = 0
    t0 = time.time()

    with torch.no_grad():
        for batch, idxs, ok in loader:
            batch = batch.to(device, non_blocking=True)
            if args.fp16:
                batch = batch.half()
            out = model.forward_features(batch)
            cls = out["x_norm_clstoken"]
            patches = out["x_norm_patchtokens"]

            if args.pool == "cls":
                vec = cls
            elif args.pool == "gem":
                vec = gem_pool(patches)
            else:  # cls_gem -- concatenating both is a common retrieval win
                vec = torch.cat([F.normalize(cls, dim=1), F.normalize(gem_pool(patches), dim=1)], dim=1)

            vec = vec.float().cpu().numpy()
            if feats is None:
                feats = np.zeros((len(paths), vec.shape[1]), dtype=np.float32)
            feats[idxs.numpy()] = vec
            failed.extend(paths[i].name for i, good in zip(idxs.tolist(), ok.tolist()) if not good)

            done += len(idxs)
            if done % (args.batch * 20) == 0 or done == len(paths):
                rate = done / (time.time() - t0)
                eta = (len(paths) - done) / rate / 60
                print(f"  {done}/{len(paths)}  {rate:.1f} img/s  ETA {eta:.1f} min", flush=True)

    names = np.array([p.name for p in paths])
    np.save(f"{args.out}.npy", feats)
    np.save(f"{args.out}_names.npy", names)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    print(f"Features: {feats.shape}  ->  {args.out}.npy")
    if failed:
        print(f"WARNING: {len(failed)} images failed to load (zero-filled): {failed[:10]}")


if __name__ == "__main__":
    main()
