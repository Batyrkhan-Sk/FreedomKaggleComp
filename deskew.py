"""Undo the rotation applied to query images, so they look like gallery images.

Every real query inspected (16000, 17932, 11728, 05362) is a painting rotated a
few degrees inside a pale frame, with flat padding filling the corners. Two
attempts to teach a model to tolerate that rotation both scored *worse* than
zero-shot, so this takes the opposite approach: detect the tilt and remove it.

Detection works by minimising the area of the content's axis-aligned bounding
box. A rotated rectangle has a larger bounding box than an upright one, so the
angle that shrinks it most is the rotation to undo. No training, no labels, and
the same routine can run over all 20,000 images because it is a no-op on
anything that is already upright.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

WORK_SIZE = 220          # detection resolution; the angle does not need full detail
ANGLE_LIMIT = 16.0
COARSE_STEP = 2.0
FINE_STEP = 0.25
MIN_GAIN = 0.12          # only act on clearly rotated images.
#   A loose threshold would also rotate upright gallery scans, whose bounding
#   box shrinks slightly at any angle through interpolation noise. Those are
#   90% of the corpus, so a false positive there costs more than a missed query.


def _content_mask(img: Image.Image, tol: int = 18) -> np.ndarray:
    """True where the image differs from its padding colour.

    The padding colour is taken from the corners rather than assumed: different
    queries use slightly different pale values.
    """
    small = img.convert("RGB")
    small.thumbnail((WORK_SIZE, WORK_SIZE))
    a = np.asarray(small).astype(np.int16)
    h, w, _ = a.shape
    k = max(3, min(h, w) // 12)
    corners = np.concatenate([
        a[:k, :k].reshape(-1, 3), a[:k, -k:].reshape(-1, 3),
        a[-k:, :k].reshape(-1, 3), a[-k:, -k:].reshape(-1, 3),
    ])
    fill = np.median(corners, axis=0)
    return (np.abs(a - fill).sum(axis=2) > tol)


def _bbox_area(mask: np.ndarray) -> float:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0 or len(cols) == 0:
        return float(mask.size)
    return float((rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1))


def _rotate_mask(mask: np.ndarray, deg: float) -> np.ndarray:
    im = Image.fromarray((mask * 255).astype(np.uint8))
    out = im.rotate(deg, resample=Image.NEAREST, expand=True, fillcolor=0)
    return np.asarray(out) > 127


def estimate_angle(img: Image.Image) -> tuple[float, float]:
    """Rotation to undo, and the fractional bounding-box shrink it achieves."""
    mask = _content_mask(img)
    if mask.mean() < 0.05 or mask.mean() > 0.995:
        return 0.0, 0.0                      # no padding to work with
    base = _bbox_area(mask)

    def area(deg: float) -> float:
        return _bbox_area(_rotate_mask(mask, deg))

    coarse = np.arange(-ANGLE_LIMIT, ANGLE_LIMIT + COARSE_STEP, COARSE_STEP)
    best = min(coarse, key=area)
    fine = np.arange(best - COARSE_STEP, best + COARSE_STEP + FINE_STEP, FINE_STEP)
    best = min(fine, key=area)
    gain = (base - area(best)) / base
    return (float(best), float(gain)) if gain > MIN_GAIN else (0.0, float(gain))


def deskew(img: Image.Image, tol: int = 18) -> tuple[Image.Image, float]:
    """Rotate the painting upright and crop away the padding.

    Returns the corrected image and the angle removed (0.0 if left untouched).
    """
    deg, gain = estimate_angle(img)
    if deg == 0.0:
        return img, 0.0

    corners = np.asarray(img.convert("RGB").resize((32, 32)))
    fill = tuple(int(v) for v in np.median(
        np.concatenate([corners[:6, :6].reshape(-1, 3), corners[:6, -6:].reshape(-1, 3),
                        corners[-6:, :6].reshape(-1, 3), corners[-6:, -6:].reshape(-1, 3)]),
        axis=0))

    rotated = img.convert("RGB").rotate(deg, resample=Image.BICUBIC, expand=True,
                                        fillcolor=fill)
    mask = _content_mask(rotated, tol)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0 or len(cols) == 0:
        return rotated, deg
    sy = rotated.height / mask.shape[0]
    sx = rotated.width / mask.shape[1]
    box = (int(cols[0] * sx), int(rows[0] * sy),
           int((cols[-1] + 1) * sx), int((rows[-1] + 1) * sy))
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        return rotated, deg
    return rotated.crop(box), deg
