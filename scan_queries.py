"""Rank all 20,000 images by how strongly they look like a rotated visitor photo.

The orientation fix only needs the extra 90/180/270 embeddings for *queries*,
not for the 19,000 gallery scans and distractors -- that is a 20x saving, and
the difference between a job that fits the remaining GPU hour and one that
does not. The deskew bounding-box gain is already a proven query detector:
a random sample of 900 flagged 45 images, i.e. 5.0%, against a true query rate
of exactly 1000/20000 = 5.0%.

Writes every image's gain rather than a thresholded list, so the cut can be
chosen afterwards against whatever GPU budget is actually left.
"""
import os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from deskew import _bbox_area, _content_mask, _rotate_mask, ANGLE_LIMIT, COARSE_STEP, FINE_STEP

Image.MAX_IMAGE_PIXELS = None
DATA_DIR = Path.home()/'Downloads/lost-in-the-museum-f1/archive/kaggle_dataset/kaggle_dataset'
OUT = 'query_gain.npz'


def raw_angle(path):
    """estimate_angle without the MIN_GAIN cut -- we want the raw score."""
    try:
        mask = _content_mask(Image.open(path).convert('RGB'))
    except Exception as e:
        print(f'  ! {path.name}: {e}', flush=True)
        return 0.0, 0.0
    if mask.mean() < 0.05 or mask.mean() > 0.995:
        return 0.0, 0.0
    base = _bbox_area(mask)
    area = lambda d: _bbox_area(_rotate_mask(mask, d))
    coarse = np.arange(-ANGLE_LIMIT, ANGLE_LIMIT + COARSE_STEP, COARSE_STEP)
    best = min(coarse, key=area)
    fine = np.arange(best - COARSE_STEP, best + COARSE_STEP + FINE_STEP, FINE_STEP)
    best = min(fine, key=area)
    return float(best), float((base - area(best)) / base)


def main():
    paths = sorted(DATA_DIR.glob('*.png'))
    print(f'{len(paths)} images, {os.cpu_count()} cores', flush=True)
    angles = np.zeros(len(paths)); gains = np.zeros(len(paths))
    t0 = time.time()
    with Pool(7) as pool:
        for i, (a, g) in enumerate(pool.imap(raw_angle, paths, chunksize=16)):
            angles[i], gains[i] = a, g
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f'  {i+1}/{len(paths)}  {rate:.1f} img/s  ETA {(len(paths)-i-1)/rate/60:.0f} min',
                      flush=True)
                np.savez(OUT, names=[p.name for p in paths], angle=angles, gain=gains, done=i + 1)
    np.savez(OUT, names=[p.name for p in paths], angle=angles, gain=gains, done=len(paths))
    print(f'\ndone in {(time.time()-t0)/60:.1f} min', flush=True)
    for t in (0.12, 0.10, 0.08, 0.06, 0.04):
        print(f'  gain > {t:.2f}: {(gains > t).sum():5d} images', flush=True)


if __name__ == '__main__':
    main()
