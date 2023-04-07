from src.data.dataloader import train_split, data_from_index
import torch
import numpy as np
from src.config import Config
import os
import pytest
import numpy as np


def Expected(dataset, seed):
    if dataset == "warwick":
        if seed == 261:
            return np.array([63, 115, 333])
        elif seed == 21:
            return np.array([114, 126, 138])


class Expected:
    def __init__(self, dataset=None):
        self.dataset = dataset

    def get_train_idx(self, dataset, seed):
        if dataset == "warwick":
            if seed == 261:
                return np.array([104, 278, 470])
            elif seed == 21:
                return np.array([36, 385, 279])
        elif dataset == "membrane":
            if seed == 261:
                return np.array([40])
            elif seed == 21:
                return np.array([189])
        elif dataset == "PhC-C2DH-U373":
            if seed == 261:
                return np.array([68])
            elif seed == 21:
                return np.array([50])
        elif dataset == "DIC_C2DH_Hela":
            if seed == 261:
                return np.array([122])
            elif seed == 21:
                return np.array([98])

    def get_val_idx(self, dataset):
        if dataset == "warwick":
            return np.array([121, 330, 381, 183, 53, 461, 463, 316, 14, 122])  # Only First 10
        elif dataset == "membrane":
            return np.array([48, 215, 162, 38, 170, 100, 78, 118, 80, 93])
        elif dataset == "PhC-C2DH-U373":
            return np.array([16, 81, 152, 53, 55, 77, 127, 97, 104, 113])
        elif dataset == "DIC_C2DH_Hela":
            return np.array([81, 85, 133, 90, 0, 109, 130, 131, 16, 66])

    def get_unlabel_idx(self, dataset, seed):
        if dataset == "warwick":
            if seed == 261:
                return np.array([311, 83, 433, 60, 344, 468, 403, 21, 135, 315])
            elif seed == 21:
                return np.array([266, 447, 184, 428, 61, 282, 474, 155, 265, 149])
        elif dataset == "membrane":
            if seed == 261:
                return np.array([235, 173, 161, 157, 167, 172, 117, 225, 212, 194])
            elif seed == 21:
                return np.array([204, 59, 89, 199, 201, 227, 211, 140, 76, 210])
        elif dataset == "PhC-C2DH-U373":
            if seed == 261:
                return np.array([6, 21, 124, 36, 128, 116, 79, 109, 30, 61])
            elif seed == 21:
                return np.array([36, 46, 33, 95, 125, 20, 6, 61, 64, 26])
        elif dataset == "DIC_C2DH_Hela":
            if seed == 261:
                return np.array([22, 101, 68, 74, 71, 125, 32, 49, 45, 95])
            elif seed == 21:
                return np.array([91, 120, 119, 2, 13, 126, 86, 110, 96, 47])


@pytest.mark.skipif(not os.path.exists("data/raw/membrane"), reason="Data files not found")
def test_data_idx():
    """
    Test if different idx for data split are returned.
    The val_idx should be consistent no matter which seed is chosen only train_idx and unlabeled_pool_idx should vary
    """
    all_datasets = list(Config.n_classes.keys())
    Expectation = Expected()
    seeds = [261, 21]
    for dataset in all_datasets:
        for seed in seeds:
            _, _, _, train_idx, val_idx, unlabeled_pool_idx = train_split(
                train_size=0.01,
                dataset=dataset,
                batch_size=4,
                to_binary=True,
                num_workers=0,
                seed=seed,
            )
            assert np.sum(np.array(train_idx) == Expectation.get_train_idx(dataset, seed)) == len(
                train_idx
            )
            assert np.sum(np.array(val_idx[:10]) == Expectation.get_val_idx(dataset)) == 10
            assert (
                np.sum(
                    np.array(unlabeled_pool_idx[:10]) == Expectation.get_unlabel_idx(dataset, seed)
                )
                == 10
            )

            # Test Overlaps
            assert len(np.intersect1d(train_idx, val_idx)) == 0
            assert len(np.intersect1d(train_idx, unlabeled_pool_idx)) == 0
            assert len(np.intersect1d(val_idx, unlabeled_pool_idx)) == 0


@pytest.mark.skipif(not os.path.exists("data/raw/membrane"), reason="Data files not found")
def test_datashapes():
    all_datasets = list(Config.n_classes.keys())
    for dataset in all_datasets:
        train_loader, _, _, _, _, _ = train_split(
            train_size=0.61, dataset=dataset, batch_size=4, to_binary=True, num_workers=0, seed=261
        )
        batch = next(iter(train_loader))
        img, mask, _ = batch

        assert (
            np.sum(np.array(list(img.shape)) == np.array([4, 64, 64])) == 3
            if dataset != "warwick"
            else np.sum(np.array(list(img.shape)) == np.array([4, 3, 64, 64])) == 4
        )
        assert np.sum(np.array(list(mask.shape)) == np.array([4, 64, 64])) == 3


@pytest.mark.skipif(not os.path.exists("data/raw/membrane"), reason="Data files not found")
def test_from_idx():
    all_datasets = list(Config.n_classes.keys())
    Expectation = Expected()
    seeds = [261, 21]
    for dataset in all_datasets:
        for seed in seeds:
            _, _, _, train_idx, val_idx, unlabeled_pool_idx = train_split(
                train_size=0.01,
                dataset=dataset,
                batch_size=4,
                to_binary=True,
                num_workers=0,
                seed=seed,
            )
            for _ in range(2):
                _, _, _, train_idx_N, val_idx_N, unlabeled_pool_idx_N = data_from_index(
                    dataset,
                    batch_size=4,
                    to_binary=True,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    unlabeled_pool_idx=unlabeled_pool_idx,
                    num_workers=0,
                    seed=seed,
                )
                assert np.sum(
                    np.array(train_idx) == Expectation.get_train_idx(dataset, seed)
                ) == len(train_idx)
                assert np.sum(np.array(val_idx[:10]) == Expectation.get_val_idx(dataset)) == 10
                assert (
                    np.sum(
                        np.array(unlabeled_pool_idx[:10])
                        == Expectation.get_unlabel_idx(dataset, seed)
                    )
                    == 10
                )

                assert np.sum(
                    np.array(train_idx_N) == Expectation.get_train_idx(dataset, seed)
                ) == len(train_idx)
                assert np.sum(np.array(val_idx_N[:10]) == Expectation.get_val_idx(dataset)) == 10
                assert (
                    np.sum(
                        np.array(unlabeled_pool_idx_N[:10])
                        == Expectation.get_unlabel_idx(dataset, seed)
                    )
                    == 10
                )


@pytest.mark.skipif(not os.path.exists("data/raw/membrane"), reason="Data files not found")
def test_all_train():
    all_datasets = list(Config.n_classes.keys())
    seeds = [261, 21]
    for dataset in all_datasets:
        for seed in seeds:
            _, _, _, train_idx, val_idx, unlabeled_pool_idx = train_split(
                train_size="100%",
                dataset=dataset,
                batch_size=4,
                to_binary=True,
                num_workers=0,
                seed=seed,
            )
            if seed == 261:
                train_idx_old = train_idx
            assert len(np.setdiff1d(train_idx, train_idx_old)) == 0


@pytest.mark.skipif(not os.path.exists("data/raw/membrane"), reason="Data files not found")
def test_how_many_classes():
    all_datasets = list(Config.datasets)
    for dataset in all_datasets:
        train_loader, val_loader, _, _, _, _ = train_split(
            train_size="100%",
            dataset=dataset,
            batch_size=4,
            to_binary=False if dataset != "membrane" else True,
            num_workers=0,
            seed=261,
        )
        uniq_classes = set()
        for batch in train_loader:
            _, masks, _ = batch
            for x in torch.unique(masks):
                uniq_classes.add(x.item())
        for batch in val_loader:
            _, masks, _ = batch
            for x in torch.unique(masks):
                uniq_classes.add(x.item())

        print(f"\nUnique Classes {dataset}: {len(uniq_classes)}\n")
        assert len(uniq_classes) == Config.n_classes[dataset]


def test_active_start():
    all_datasets = list(Config.datasets)
    for dataset in all_datasets:
        for seed in [261, 17]:
            for i in range(2):
                _, _, _, train_idx, val_idx, unlabel_idx = train_split(
                    train_size="2-Samples",
                    dataset=dataset,
                    batch_size=4,
                    to_binary=False if dataset != "membrane" else True,
                    num_workers=0,
                    seed=seed,
                )
                if i == 0:
                    tmp_train = train_idx
                    tmp_val = val_idx
                    tmp_unlabel = unlabel_idx
                if i == 1:
                    assert np.sum(np.array(tmp_train) == np.array(train_idx)) == len(train_idx)
                    assert np.sum(np.array(tmp_val) == np.array(val_idx)) == len(val_idx)
                    assert np.sum(np.array(tmp_unlabel) == np.array(unlabel_idx)) == len(
                        unlabel_idx
                    )
                ##
                assert (np.size(train_idx) + np.size(val_idx) + np.size(unlabel_idx)) == len(
                    [x for x in os.listdir(f"data/raw/{dataset}/image") if x != ".DS_Store"]
                )


# if __name__=='__main__':
#     #test_active_start()
#     test_data_idx()
