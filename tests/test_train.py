import pytest
import os
from src.models.train_model import train
from src.config import find_best_device

datasets = ["warwick", "PhC-C2DH-U373", "DIC_C2DH_Hela", "membrane"]


def Expected(dataset):
    if dataset == "warwick":
        return {
            "train_loss": 0.628966212272644,
            "train_dice": 0.6731500029563904,
            "val_loss": 0.730646550655365,
            "val_dice": 0.5577081441879272,
            "val_dice_all": 0.5767558217048645,
            "val_NLL_all": 0.7315257787704468,
        }
    elif dataset == "membrane":
        return {
            "train_loss": 0.6939770579338074,
            "train_dice": 0.7257911562919617,
            "val_loss": 0.4101446270942688,
            "val_dice": 0.8927456140518188,
            "val_dice_all": 0.8928332924842834,
            "val_NLL_all": 0.410144567489624,
        }
    elif dataset == "PhC-C2DH-U373":
        return {
            "train_loss": 0.1838405430316925,
            "train_dice": 0.5304087996482849,
            "val_loss": 0.1175345852971077,
            "val_dice": 0.6170467138290405,
            "val_dice_all": 0.6180570125579834,
            "val_NLL_all": 0.1175345927476883,
        }
    elif dataset == "DIC_C2DH_Hela":
        return {
            "train_loss": 0.595565915107727,
            "train_dice": 0.6300970911979675,
            "val_loss": 0.4885545074939728,
            "val_dice": 0.7735194563865662,
            "val_dice_all": 0.7763318419456482,
            "val_NLL_all": 0.48891156911849976,
        }


@pytest.mark.skipif(not os.path.exists("data/raw/membrane"), reason="Data files not found")
def test_model_train():
    for dataset in datasets:
        res = train(
            dataset=dataset,
            train_size=0.61,
            epochs=1,
            turn_off_wandb=True,
            device="cpu",
        )
        expectation = Expected(dataset)
        assert sum([res[x][0][0] == expectation[x] for x in expectation.keys()]) == 6
