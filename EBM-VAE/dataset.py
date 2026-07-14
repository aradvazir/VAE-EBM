"""
Anime face dataset loader.

Kaggle dataset: https://www.kaggle.com/datasets/splcher/animefacedataset
After downloading, unzip so the structure is:

    data/
      animefacedataset/
        images/
          000001.jpg
          000002.jpg
          ...

Usage:
    train_loader, val_loader = get_dataloaders(cfg)
"""
import random
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class AnimeFaceDataset(Dataset):
    """Flat image dataset for anime face images.

    Normalises to [-1, 1] (Tanh output range) via mean/std = 0.5.
    """

    VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        root_dir: str,
        image_size: int = 32,
        subset_size: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        # Collect all image paths
        self.paths = [
            p for p in self.root_dir.rglob("*")
            if p.suffix.lower() in self.VALID_EXTS
        ]
        if not self.paths:
            raise FileNotFoundError(
                f"No images found under '{root_dir}'. "
                "Make sure you've unzipped the Kaggle dataset there."
            )

        if subset_size is not None and subset_size < len(self.paths):
            rng = random.Random(seed)
            self.paths = rng.sample(self.paths, subset_size)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5]),
            # [-1,1]
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))


def get_dataloaders(
    cfg,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from config."""
    dataset = AnimeFaceDataset(
        root_dir=cfg.data_dir,
        image_size=cfg.image_size,
        subset_size=cfg.subset_size,
    )

    n_total = len(dataset)
    n_val   = max(1, int(n_total * cfg.val_split))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    print(f"Dataset  : {cfg.data_dir}")
    print(f"  Total  : {n_total:,} images")
    print(f"  Train  : {n_train:,} | Val: {n_val:,}")
    print(f"  Size   : {cfg.image_size}×{cfg.image_size}")

    shared_kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        **shared_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        **shared_kwargs,
    )

    return train_loader, val_loader
