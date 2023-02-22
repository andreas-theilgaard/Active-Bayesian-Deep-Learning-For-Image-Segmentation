from src.data.DataModule import SegmentationData
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset


def train_split(train_size, dataset, batch_size, to_binary, num_workers=0):
    """
    Move to data filder
    """
    dataset = SegmentationData(dataset=dataset, img_height=64, img_width=64, to_binary=to_binary)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.33)
    train_idx, _ = train_test_split(train_idx, train_size=train_size)
    print(f"Training with {len(train_idx)} images!")
    print(f"Validating with {len(val_idx)} images!")
    train_loader = Subset(dataset, train_idx)
    val_loader = Subset(dataset, val_idx)
    train_loader = DataLoader(
        train_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_loader, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return (train_loader, val_loader)
