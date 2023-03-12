from src.data.DataModule import SegmentationData
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch


def train_split(
    train_size,
    dataset,
    batch_size,
    to_binary,
    num_workers=0,
    seed=None,
    interpolate_image=None,
    interpolate_mask=None,
):
    """
    Move to data filder
    """
    random_state = seed if seed else np.random.choice(0, 500, 1)
    dataset = SegmentationData(
        dataset=dataset,
        img_height=64,
        img_width=64,
        to_binary=to_binary,
        interpolate_image=interpolate_image,
        interpolate_mask=interpolate_mask,
    )
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=0.33, random_state=random_state
    )
    if (len(train_idx) * train_size) < 1.0:
        train_size = 1
    train_idx, _ = train_test_split(train_idx, train_size=train_size, random_state=random_state)
    print(f"Training with {len(train_idx)} images!")
    print(f"Validating with {len(val_idx)} images!")
    train_loader = Subset(dataset, train_idx)
    val_loader = Subset(dataset, val_idx)
    train_loader = DataLoader(
        train_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(random_state),
    )
    val_loader = DataLoader(
        val_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(random_state),
    )
    return (train_loader, val_loader)
