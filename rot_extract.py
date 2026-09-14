"""Embed the corpus at one 90-degree rotation, with checkpointing.

The 0-degree pass already exists as features_l518.npy, so only 90/180/270 are
needed. Each is written separately and averaged afterwards, which means a
crashed pass costs one rotation rather than the whole job.
"""
import argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None
DATA_DIR = Path.home()/'Downloads/lost-in-the-museum-f1/archive/kaggle_dataset/kaggle_dataset'
MEAN, STD = (0.485,0.456,0.406), (0.229,0.224,0.225)


class RotDataset(Dataset):
    def __init__(self, paths, size, angle):
        self.paths, self.size, self.angle = paths, size, angle
        self.tf = transforms.Compose([
            transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize(MEAN,STD)])

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]).convert('RGB')
            return self.tf(im.rotate(self.angle, expand=True)), i
        except Exception as e:
            print(f'  ! {self.paths[i].name}: {e}', flush=True)
            return torch.zeros(3, self.size, self.size), i


def gem(pt, p=3.0, eps=1e-6): return pt.clamp(min=eps).pow(p).mean(1).pow(1/p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--angle', type=int, required=True)
    ap.add_argument('--size', type=int, default=518)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--ckpt-secs', type=float, default=120)
    args = ap.parse_args()

    out = f'features_l518_rot{args.angle}'
    paths = sorted(DATA_DIR.glob('*.png'))
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'{len(paths)} images  angle {args.angle}  device {dev}', flush=True)

    model = torch.hub.load('facebookresearch/dinov2','dinov2_vitl14',verbose=False).eval().to(dev)

    ck, ckd = f'{out}_ckpt.npy', f'{out}_done.npy'
    feats, done = None, np.zeros(len(paths), dtype=bool)
    if Path(ck).exists() and Path(ckd).exists():
        feats, done = np.load(ck), np.load(ckd)
        print(f'resuming: {done.sum()}/{len(paths)}', flush=True)

    todo = np.flatnonzero(~done)
    loader = DataLoader(Subset(RotDataset(paths, args.size, args.angle), todo.tolist()),
                        batch_size=args.batch, shuffle=False, num_workers=args.workers)

    t0 = last = time.time(); start_n = int(done.sum())
    with torch.no_grad():
        for batch, idxs in loader:
            o = model.forward_features(batch.to(dev))
            v = torch.cat([F.normalize(o['x_norm_clstoken'],dim=1),
                           F.normalize(gem(o['x_norm_patchtokens']),dim=1)],dim=1).float().cpu().numpy()
            if feats is None:
                feats = np.zeros((len(paths), v.shape[1]), dtype=np.float32)
            feats[idxs.numpy()] = v
            done[idxs.numpy()] = True
            if time.time()-last > args.ckpt_secs:
                np.save(ck, feats); np.save(ckd, done); last = time.time()
            n = int(done.sum())
            if n % 400 < args.batch:
                r = (n-start_n)/(time.time()-t0)
                print(f'  {n}/{len(paths)}  {r:.1f} img/s  ETA {(len(paths)-n)/max(r,1e-6)/60:.0f} min', flush=True)

    assert done.all(), f'only {done.sum()}/{len(paths)}'
    np.save(f'{out}.npy', feats)
    for f in (ck, ckd): Path(f).unlink(missing_ok=True)
    print(f'done in {(time.time()-t0)/60:.1f} min -> {out}.npy')


if __name__ == '__main__':
    main()
