from src.models.train_active import run_active
from src.config import find_best_device

seeds = [21]  # , 4, 7, 9, 12]  # , 45, 17, 5, 8, 10]
model_methods = ["MCD"]
AcquisitionFunctions = ["Random", "ShanonEntropy", "BALD"]
dataset = "PhC-C2DH-U373"

methods = {
    "BatchNorm": {"enable_dropout": False, "enable_pool_dropout": False},
    "MCD": {"enable_dropout": False, "enable_pool_dropout": True},
}


def active_run(seeds, model_method, AcquisitionFunctions, dataset):
    for seed in seeds:
        for AcquisitionFunction in AcquisitionFunctions:
            for model_method in model_methods:
                if AcquisitionFunction == "Random":
                    model_params = methods["BatchNorm"]
                    model_method = "BatchNorm"
                if model_method == "MCD":
                    model_params = methods["MCD"]
                print(f"-----------------------------------\n")
                print(f"Running Active Train Loop For {dataset}\n")
                print(
                    f""" AcquisitionFunction={AcquisitionFunction} \n Bayesian Inference = {None if model_method=="BatchNorm" else model_method} \n seed={seed}"""
                )
                print(f"-----------------------------------\n")
                run_active(
                    batch_size=4,
                    learning_rate=0.001,
                    epochs=1,
                    momentum=0.9,
                    beta0=0.9,
                    train_size=0.95,
                    dataset=dataset,
                    device=find_best_device(),
                    validation_size=0.33,
                    binary=True,
                    enable_dropout=model_params["enable_dropout"],
                    dropout_prob=0.5,
                    enable_pool_dropout=model_params["enable_pool_dropout"],
                    pool_dropout_prob=0.5,
                    bilinear_method=False,
                    model_method=model_method,
                    seed=seed,
                    torch_seeds=[17],
                    first_train=True,
                    AcquisitionFunction=AcquisitionFunction,  # [Random,ShanonEntropy,BALD]
                    train_loader=None,
                    val_loader=None,
                    unlabeled_loader=None,
                    train_idx=None,
                    val_idx=None,
                    unlabeled_pool_idx=None,
                )


if __name__ == "__main__":
    active_run(seeds, model_methods, AcquisitionFunctions, dataset)
