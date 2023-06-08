import argparse
from src.data.data_utils import get_data
import os
import wandb
import subprocess
from src.config import find_best_device
import torch
from src.experiments.experiment_utils import arrayify_results
from src.models.train_model import train

# hyper_params in dictinary based on dataset type
dataset_size = [0.01, 0.32, 0.63]
seeds = [21, 4, 7, 9, 12]  # , 45, 17, 5, 8, 10]
methods = {
    "batchnorm": {"enable_dropout": False, "enable_pool_dropout": False},
    "conv__layer_dropout": {"enable_dropout": True, "enable_pool_dropout": False},
    "pool__layer_dropout": {"enable_dropout": False, "enable_pool_dropout": True},
    "pool__and_conv_layer_dropout": {"enable_dropout": True, "enable_pool_dropout": True},
}

parser = argparse.ArgumentParser(description="Training arguments")
parser.add_argument("--save_path", default="compare_results/warwick")  #
parser.add_argument("--dataset", default="warwick")  #
parser.add_argument("--device", default=find_best_device())  # find_best_device
parser.add_argument("--binary", default=True)
parser.add_argument("--number_iters", default=len(seeds))
parser.add_argument("--epochs", default=100)


args = parser.parse_args()
print(f"Using dataset: {args.dataset}")
print(f"The results '{args.save_path}' will be saved at 'results/{args.save_path}'")
args.number_iters = int(args.number_iters)
args.epochs = int(args.epochs)
args.save_path = f"results/{args.save_path}"


def compare():
    for method in methods:
        for train_size in dataset_size:
            for iteration in range(args.number_iters):
                # Get learning rate and momentum
                experiment_name = f"{args.dataset}_{method}_{train_size}"
                job_type = f"run {iteration+1}"
                seed = seeds[iteration]
                stored_metrics = train(
                    dataset=args.dataset,
                    train_size=train_size,
                    epochs=args.epochs,
                    device=args.device,
                    binary=args.binary,
                    iter=iteration,
                    experiment_name=experiment_name,
                    job_type=job_type,
                    enable_dropout=methods[method]["enable_dropout"],
                    enable_pool_dropout=methods[method]["enable_pool_dropout"],
                    model_method=method,
                    seed=seed,
                    turn_off_wandb=True,
                )
                res = arrayify_results(stored_metrics, args.save_path)


if __name__ == "__main__":
    get_data()
    compare()
