"""Degradation matched to the *observed* query images.

The first version of this pipeline was reverse-engineered from a single query
and was far too harsh: it cropped away up to 45% of the painting and downscaled
to roughly 64px. Fine-tuning against it cost ~0.09 on the leaderboard, because
teaching invariance to distortions that never occur throws away exactly the fine
detail needed to tell two similar paintings apart.

Inspecting real queries found by their padding signature (17932, 11728, 05362,
16000) shows something much milder:

* the **whole painting stays visible** -- it is rotated inside a frame, not cropped
* rotation is roughly 5-10 degrees, with pale padding filling the corners
* detail is largely retained; some are sharp, some mildly soft
* colour is sometimes washed out or near-greyscale

So rotation-with-padding is the dominant transform and everything else is light.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_FILL = 240


class RotatePad:
    """Rotate with expand=True so the full painting survives inside a pale frame.

    `expand=True` is the detail that reproduces the look of the real queries:
    without it the rotation crops the corners off instead of padding them.
    """

    def __init__(self, max_deg: float = 10.0, fill: int = PAD_FILL):
        self.max_deg, self.fill = max_deg, fill

    def __call__(self, img: Image.Image) -> Image.Image:
        # Sample the *magnitude* away from zero: a uniform(-10, 10) draw spends
        # most of its mass on 1-2 degree tilts that leave almost no padding,
        # whereas every real query inspected showed a clearly visible rotation.
        mag = float(np.random.uniform(0.3 * self.max_deg, self.max_deg))
        deg = mag if np.random.rand() < 0.5 else -mag
        return img.rotate(deg, resample=Image.BICUBIC, expand=True,
                          fillcolor=(self.fill,) * 3)


class MildDownscale:
    """Lose some resolution, but nothing like the original 64px collapse."""

    def __init__(self, lo: float = 0.45, hi: float = 1.0):
        self.lo, self.hi = lo, hi

    def __call__(self, img: Image.Image) -> Image.Image:
        f = float(np.random.uniform(self.lo, self.hi))
        if f > 0.97:
            return img
        w, h = img.size
        small = (max(32, int(w * f)), max(32, int(h * f)))
        return img.resize(small, Image.BILINEAR).resize((w, h), Image.BICUBIC)


def query_view(size: int, strength: float = 1.0) -> transforms.Compose:
    return transforms.Compose([
        RotatePad(max_deg=10.0 * strength),
        transforms.RandomPerspective(distortion_scale=0.10 * strength, p=0.35,
                                     fill=PAD_FILL),
        # Resize the whole rotated frame rather than cropping into it. Cropping
        # here would remove the pale corners, which are the single most visible
        # signature of a real query -- the model must learn to look past them,
        # not be shielded from them.
        transforms.Resize((size, size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ColorJitter(brightness=0.25 * strength, contrast=0.25 * strength,
                               saturation=0.35 * strength, hue=0.02 * strength),
        transforms.RandomGrayscale(p=0.10),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.3, 1.2 * strength))], p=0.5),
        MildDownscale(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def gallery_view(size: int, jitter: bool = True) -> transforms.Compose:
    steps = [transforms.Resize((size, size),
                               interpolation=transforms.InterpolationMode.BICUBIC)]
    if jitter:
        steps.append(transforms.ColorJitter(brightness=0.06, contrast=0.06))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)
