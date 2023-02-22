import torch
import argparse
from src.models.train_model import train
from src.experiments.experiment_utils import arrayify_results
from src.config import find_best_device
import numpy as np

# hyper_params in dictinary based on dataset type
dataset_size = [0.01, 0.32, 0.63, 0.99]  # TO DO: do some more runs with other values

seeds = [182, 322, 291, 292, 122, 53, 261, 427, 174, 128]

hyper_params = {
    "membrane": {
        0.01: {"lr": 0.0001, "momentum": 0.9},
        0.32: {"lr": 0.001, "momentum": 0.6},
        0.63: {"lr": 0.001, "momentum": 0.1},
        0.99: {"lr": 0.001, "momentum": 0.1},
    },
    "warwick": {
        0.01: {"lr": 0.001, "momentum": 0.9},
        0.32: {"lr": 0.0001, "momentum": 0.6},
        0.63: {"lr": 0.0001, "momentum": 0.1},
        0.99: {"lr": 0.0001, "momentum": 0.1},
    },
}

methods = {
    "batchnorm": {"enable_dropout": False, "enable_pool_dropout": False},
    "conv__layer_dropout": {"enable_dropout": True, "enable_pool_dropout": False},
    "pool__layer_dropout": {"enable_dropout": False, "enable_pool_dropout": True},
    "pool__and_conv_layer_dropout": {"enable_dropout": True, "enable_pool_dropout": True},
}


def compare(dataset_size):
    # Clean up here remove redundancy
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--save_path", default="new_membrane_dropout_test")
    parser.add_argument("--dataset", default="membrane")
    parser.add_argument("--number_iters", default=2)
    parser.add_argument("--device", default="cpu")  # find_best_device
    parser.add_argument("--bilinear_method", default=True)  # bilinear_method

    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--epochs", default=5)
    parser.add_argument("--validation_size", default=0.33)
    parser.add_argument("--binary", default=True)
    parser.add_argument("--dropout_prob", default=0.5)
    parser.add_argument("--dropout_pool_prob", default=0.5)

    args = parser.parse_args()
    args.number_iters = int(args.number_iters)
    args.epochs = int(args.epochs)
    args.save_path = f"compare_results/{args.save_path}"

    for method in methods:
        for train_size in dataset_size:
            for iteration in range(args.number_iters):
                # Get learning rate and momentum
                lr = hyper_params[args.dataset][train_size]["lr"]
                momentum = hyper_params[args.dataset][train_size]["momentum"]
                experiment_name = f"{args.dataset}_{method}_{train_size}"
                job_type = f"run {iteration+1}"
                seed = seeds[iteration]
                stored_metrics = train(
                    dataset=args.dataset,
                    train_size=train_size,
                    epochs=args.epochs,
                    lr=lr,
                    momentum=momentum,
                    device=args.device,
                    batch_size=args.batch_size,
                    validation_size=args.validation_size,
                    binary=args.binary,
                    iter=iteration,
                    experiment_name=experiment_name,
                    job_type=job_type,
                    enable_dropout=methods[method]["enable_dropout"],
                    dropout_prob=args.dropout_prob,
                    enable_pool_dropout=methods[method]["enable_pool_dropout"],
                    pool_dropout_prob=args.dropout_pool_prob,
                    bilinear_method=args.bilinear_method,
                    model_method=method,
                    seed=seed,
                )
                res = arrayify_results(stored_metrics, args.save_path)


if __name__ == "__main__":
    compare(dataset_size=dataset_size)
