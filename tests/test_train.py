import pytest
import os
from src.models.new_train_llop import train


datasets = ["warwick", "PhC-C2DH-U373", "DIC_C2DH_Hela", "membrane"]


@pytest.mark.skipif(not os.path.exists("data/raw/membrane"), reason="Data files not found")
def Expected(dataset):
    if dataset == "warwick":
        return {
            "train_loss": 0.652055025100708,
            "train_dice": 0.6800824999809265,
            "val_loss": 0.5792975425720215,
            "val_dice": 0.7148911356925964,
            "val_dice_all": 0.7196431159973145,
            "val_NLL_all": 0.5763683319091797,
        }
    elif dataset == "membrane":
        return {
            "train_loss": 0.6655133962631226,
            "train_dice": 0.7325355410575867,
            "val_loss": 0.4362846910953522,
            "val_dice": 0.8816827535629272,
            "val_dice_all": 0.8818323612213135,
            "val_NLL_all": 0.4362432360649109,
        }
    elif dataset == "PhC-C2DH-U373":
        return {
            "train_loss": 0.19119656085968018,
            "train_dice": 0.5030667781829834,
            "val_loss": 0.12329605221748352,
            "val_dice": 0.6123033761978149,
            "val_dice_all": 0.6123578548431396,
            "val_NLL_all": 0.12329605221748352,
        }
    elif dataset == "DIC_C2DH_Hela":
        return {
            "train_loss": 0.5812353491783142,
            "train_dice": 0.6570649147033691,
            "val_loss": 0.6112041473388672,
            "val_dice": 0.7202469706535339,
            "val_dice_all": 0.7273185849189758,
            "val_NLL_all": 0.5911003947257996,
        }


def test_model_train():
    for dataset in datasets:
        res = train(dataset=dataset, train_size=0.61, epochs=1, turn_off_wandb=True)
        expectation = Expected(dataset)
        assert sum([res[x][0][0] == expectation[x] for x in expectation.keys()]) == 6
