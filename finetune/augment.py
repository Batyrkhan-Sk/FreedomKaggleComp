"""Query-like degradation, shared by training and evaluation.

The competition's query images are synthetically degraded copies of gallery
images: rotated with pale padding filling the corners, perspective-warped,
cropped, blurred and downscaled. Reproducing that transform is what lets us
manufacture labelled pairs from an unlabelled dataset.

Two properties matter and are easy to get wrong:

* Degradation is defined in **absolute pixels**, not model-input pixels. A real
  photograph has a fixed capture resolution regardless of what size we feed the
  network, so anchoring the downscale to a fixed reference keeps 392px and
  518px runs comparable.
* The crop must **not** force a square. A real cropped query has its own aspect
  ratio, different from its gallery original's; squaring it here would hide the
  very mismatch the model needs to learn to tolerate.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Pale grey matches the padding observed in real query images (e.g. 16000.png).
PAD_FILL = 235

# Absolute pixel reference for the downscale, independent of model input size.
DOWNSCALE_REFERENCE = 224


class Downscale:
    """Shrink hard, then upsample back, destroying high-frequency detail.

    Blur alone does not model a low-resolution capture: it smooths without
    removing the information a larger sensor would have recorded.
    """

    def __init__(self, factor: float, reference: int = DOWNSCALE_REFERENCE):
        self.factor = factor
        self.reference = reference

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        target_long = max(16.0, self.reference / self.factor)
        scale = min(1.0, target_long / max(w, h))
        small = (max(8, int(w * scale)), max(8, int(h * scale)))
        return img.resize(small, Image.BILINEAR).resize((w, h), Image.BICUBIC)


class RandomCropAspect:
    """Crop a random sub-region, preserving a random (non-square) aspect ratio."""

    def __init__(self, scale=(0.55, 0.95), ratio=(0.7, 1.4)):
        self.scale, self.ratio = scale, ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        area = w * h * np.random.uniform(*self.scale)
        ar = float(np.exp(np.random.uniform(np.log(self.ratio[0]), np.log(self.ratio[1]))))
        cw = min(w, max(8, int(round(np.sqrt(area * ar)))))
        ch = min(h, max(8, int(round(np.sqrt(area / ar)))))
        x = np.random.randint(0, w - cw + 1)
        y = np.random.randint(0, h - ch + 1)
        return img.crop((x, y, x + cw, y + ch))


def query_view(size: int, strength: float = 1.0) -> transforms.Compose:
    """The degraded view -- stands in for a visitor's photograph."""
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=12 * strength,
            translate=(0.04 * strength, 0.04 * strength),
            scale=(1 - 0.15 * strength, 1 + 0.05 * strength),
            shear=6 * strength,
            fill=PAD_FILL,
        ),
        transforms.RandomPerspective(distortion_scale=0.22 * strength, p=0.85, fill=PAD_FILL),
        RandomCropAspect(),
        transforms.ColorJitter(
            brightness=0.3 * strength, contrast=0.3 * strength,
            saturation=0.25 * strength, hue=0.03 * strength,
        ),
        transforms.GaussianBlur(kernel_size=7, sigma=(0.4 * strength + 0.2, 2.4 * strength + 0.2)),
        Downscale(factor=1.0 + 2.5 * strength),
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def gallery_view(size: int, jitter: bool = True) -> transforms.Compose:
    """The clean view -- stands in for the studio-captured gallery image.

    A little jitter is kept even on the anchor: the gallery side is not
    pixel-identical to what the model saw in training either, and a completely
    static anchor makes the contrastive task degenerately easy.
    """
    steps = [transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC)]
    if jitter:
        steps.append(transforms.ColorJitter(brightness=0.08, contrast=0.08))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)
