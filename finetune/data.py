"""Paired dataset: one clean anchor and one degraded positive per image.

Every image in the competition set is usable as a training anchor. We are never
told which of the 20,000 are gallery images, queries or distractors, and it does
not matter: the objective is "a degraded copy of X should retrieve X", which is
well-defined for any X.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from augment import gallery_view, query_view

Image.MAX_IMAGE_PIXELS = None  # a few gallery scans are enormous


class PairDataset(Dataset):
    """Yields (anchor, positive, index) with the positive heavily degraded."""

    def __init__(self, paths: list[Path], size: int, strength: float = 1.0):
        self.paths = paths
        self.size = size
        self.anchor_tf = gallery_view(size)
        self.query_tf = query_view(size, strength)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        path = self.paths[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:  # noqa: BLE001 -- a corrupt file must not kill training
            print(f"  ! unreadable {path.name}: {e}", flush=True)
            blank = torch.zeros(3, self.size, self.size)
            return blank, blank, i
        return self.anchor_tf(img), self.query_tf(img), i


class InferenceDataset(Dataset):
    """Clean, deterministic view for embedding the full corpus at submission time."""

    def __init__(self, paths: list[Path], size: int):
        self.paths = paths
        self.size = size
        self.tf = gallery_view(size, jitter=False)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"  ! unreadable {self.paths[i].name}: {e}", flush=True)
            # Never drop a row -- the submission must carry all 20,000 images.
            return torch.zeros(3, self.size, self.size), i
        return self.tf(img), i
