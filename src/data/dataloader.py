from src.data.DataModule import SegmentationData
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch

# np.random.seed(7)
# torch.manual_seed(7)


def train_split(train_size, dataset, batch_size, to_binary, num_workers=0, seed=None):
    """
    Move to data folder
    """
    random_state = seed if seed else np.random.choice(0, 500, 1)
    dataset = SegmentationData(dataset=dataset, img_height=64, img_width=64, to_binary=to_binary)
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=0.33, random_state=random_state
    )
    if (len(train_idx) * train_size) < 1.0:
        train_size = 1
    train_idx, unlabeled_pool_idx = train_test_split(
        train_idx, train_size=train_size, random_state=random_state
    )
    print(f"Training with {len(train_idx)} images!")
    print(f"Validating with {len(val_idx)} images!")
    train_loader = Subset(dataset, train_idx)
    val_loader = Subset(dataset, val_idx)
    unlabeled_loader = Subset(dataset, unlabeled_pool_idx)
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
    unlabeled_loader = DataLoader(
        unlabeled_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(random_state),
    )
    return (train_loader, val_loader, unlabeled_loader, train_idx, val_idx, unlabeled_pool_idx)


def data_from_index(
    dataset, batch_size, to_binary, train_idx, val_idx, unlabeled_pool_idx, num_workers=0, seed=None
):
    random_state = seed if seed else np.random.choice(0, 500, 1)
    dataset = SegmentationData(dataset=dataset, img_height=64, img_width=64, to_binary=to_binary)

    train_loader = Subset(dataset, train_idx)
    val_loader = Subset(dataset, val_idx)
    unlabeled_loader = Subset(dataset, unlabeled_pool_idx)
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
    unlabeled_loader = DataLoader(
        unlabeled_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(random_state),
    )
    return (train_loader, val_loader, unlabeled_loader, train_idx, val_idx, unlabeled_pool_idx)


def process(input, first_run):
    import pandas as pd
    import os

    file_exists = os.path.isfile(f"results/active_learning/test.json")
    if first_run:
        if not file_exists:
            df = pd.DataFrame(
                {"Train Round": input["epoch"], "Mu": input["mu"], "idx": input["idx"]}
            )
            df.to_json("results/active_learning/test.json")
        else:
            df = pd.read_json("results/active_learning/test.json")
            df_new = pd.DataFrame(
                {"Train Round": input["epoch"], "Mu": input["mu"], "idx": input["idx"]}
            )
            df_out = pd.concat([df, df_new], ignore_index=True)
            df_out.to_json("results/active_learning/test.json")
    else:
        df = pd.read_json(f"results/active_learning/test.json")
        df_new = pd.DataFrame(
            {"Train Round": input["epoch"], "Mu": input["mu"], "idx": input["idx"]}
        )
        df_out = pd.concat([df, df_new], ignore_index=True)
        df_out.to_json("results/active_learning/test.json")


def next_label(train_round, train_idx, val_idx, unlabeled_pool_idx):
    import pandas as pd

    assert len(np.unique(np.array(unlabeled_pool_idx))) == len(unlabeled_pool_idx)
    df = pd.read_json(f"results/active_learning/test.json")
    df_subset = df[df["Train Round"] == train_round]
    next_label = df_subset[df_subset["Mu"] == df_subset["Mu"].min()]["idx"].values
    print(next_label)
    if len(next_label) != 1:
        next_label = next_label[0]
    elif len(next_label) == 0:
        return -1
    else:
        next_label = next_label[0]

    train_idx.append(next_label)  # torch.cat([train_idx,torch.tensor([next_label['idx']])])
    unlabeled_pool_idx.remove(
        next_label
    )  # = unlabeled_pool_idx[unlabeled_pool_idx!=next_label['idx']]
    print("After addition")
    print(train_idx)
    return (train_idx, val_idx, unlabeled_pool_idx)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    first_run = True
    dataset = "warwick"
    batch_size = 4
    to_binary = True
    seed = 7
    import os

    file_exists = os.path.isfile(f"results/active_learning/test.json")
    if file_exists:
        os.remove("results/active_learning/test.json")

    (
        train_loader,
        val_loader,
        unlabeled_loader,
        train_idx,
        val_idx,
        unlabeled_pool_idx,
    ) = train_split(
        0.01, dataset=dataset, batch_size=batch_size, to_binary=True, num_workers=0, seed=seed
    )
    for epoch in range(10):
        if not first_run:
            (
                train_loader,
                val_loader,
                unlabeled_loader,
                train_idx,
                val_idx,
                unlabeled_pool_idx,
            ) = data_from_index(
                dataset,
                batch_size,
                to_binary,
                train_idx,
                val_idx,
                unlabeled_pool_idx,
                num_workers=0,
                seed=seed,
            )
        # Train Model Here
        for batch in train_loader:
            img, mask, idx = batch

        # Use train model to acquire next label
        for batch in unlabeled_loader:
            img, mask, idx = batch
            mu = torch.mean(img.view(img.shape[0], -1), dim=1, keepdim=True)
            process(
                {
                    "epoch": [epoch for x in range(img.shape[0])],
                    "mu": mu.detach().cpu().numpy().flatten(),
                    "idx": idx.detach().cpu().numpy().flatten(),
                },
                first_run,
            )
        # Move next label to train and remove from pool
        train_idx, val_idx, unlabeled_pool_idx = next_label(
            epoch, train_idx, val_idx, unlabeled_pool_idx
        )

        for batch in val_loader:
            img, mask, idx = batch
            # do validation

        first_run = False
    import pandas as pd

    print(pd.read_json(f"results/active_learning/test.json"))
