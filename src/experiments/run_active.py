import torch
from src.models.train_active_loop import run_active
import argparse
from src.config import find_best_device

# seeds: [21, 4, 7, 9, 12]
parser = argparse.ArgumentParser(description="Training arguments")
parser.add_argument("--device", default=find_best_device())  # find_best_device
parser.add_argument("--epochs", default=100)
parser.add_argument("--seed", default=21)
parser.add_argument("--dataset", default="PhC-C2DH-U373")
parser.add_argument("--start_size", default="2-Samples")
parser.add_argument("--binary", default=True)
parser.add_argument("--n_items_to_label", default=2)
parser.add_argument("--EarlyStop", default=True)
parser.add_argument("--TurnOffWandB", default=True)

args = parser.parse_args()
args.epochs = int(args.epochs)
args.seed = int(args.seed)
args.n_items_to_label = int(args.n_items_to_label)

torch_seeds = [17, 8, 42, 19, 5]

model_params = {
    "in_ch": 1,
    "n_classes": 1,
    "bilinear_method": False,
    "momentum": 0.9,
    "enable_dropout": False,
    "dropout_prob": 0.5,
    "enable_pool_dropout": False,
    "pool_dropout_prob": 0.5,
}

methods = {
    "BatchNorm": {"enable_pool_dropout": False},
    "MCD": {"enable_pool_dropout": True},
    "DeepEnsemble": {"enable_pool_dropout": False},
    "Laplace": {"enable_pool_dropout": False},
}

AcquisitionFunctions = ["Random", "ShanonEntropy", "BALD", "JensenDivergence"]

# Methods that will be applied for this run
apply_methods = ["BatchNorm", "MCD"]


def active_trainer():
    for Acq_func in AcquisitionFunctions:
        for apply_method in apply_methods:
            if (apply_method == "BatchNorm" and Acq_func != "Random") or (
                Acq_func == "Random" and apply_method != "BatchNorm"
            ):
                continue
            if apply_method == "DeepEnsemble":
                applied_torch_seeds = torch_seeds
            else:
                applied_torch_seeds = [torch_seeds[0]]

            print(f"-----------------------------------\n")
            print(f"Running Active Train Loop For {args.dataset}\n")
            print(
                f""" AcquisitionFunction={Acq_func} \n Bayesian Inference = {None if apply_method=="BatchNorm" else apply_method} \n seed={args.seed} \n torch_seed={','.join(map(str,applied_torch_seeds))}"""
            )
            print(f"-----------------------------------\n")

            applied_model_params = model_params.copy()
            applied_model_params["enable_pool_dropout"] = methods[apply_method][
                "enable_pool_dropout"
            ]

            run_active(
                model_params=applied_model_params,
                dataset=args.dataset,
                epochs=args.epochs,
                start_size=args.start_size,
                model_method=apply_method,
                AcquisitionFunction=Acq_func,
                torch_seeds=applied_torch_seeds,
                seed=args.seed,
                binary=args.binary,
                device=args.device,
                n_items_to_label=args.n_items_to_label,
                Earlystopping_=args.EarlyStop,
                turn_off_wandb=args.TurnOffWandB,
            )


if __name__ == "__main__":
    active_trainer()
