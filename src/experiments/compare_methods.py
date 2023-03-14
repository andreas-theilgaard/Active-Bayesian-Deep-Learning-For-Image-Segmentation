import argparse
from src.data.data_utils import get_data
import os
import wandb
import subprocess
from src.config import find_best_device
import torch
from src.data.data_utils import upload_file


dataset = os.environ["dataset"]
save_path = os.environ["save_path"]
# dataset_size_used = os.environ["ds"]
print(f"Using dataset: {dataset}")
print(f"The results '{save_path}' will be saved at 'results/{save_path}'")
# print(f"Using dataset size of {dataset_size_used}")
# dataset_size = [dataset_size_used]

############## Setting Up Weights & Biases ###############
bash_cmd = f"wandb login 82a3b5a7b8ff626de2d5ae45becdac5fa040d0f7"
process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
if not error:
    print("Logged succesfully into wandb")

from src.models.train_model import train
from src.experiments.experiment_utils import arrayify_results

# hyper_params in dictinary based on dataset type
dataset_size = [0.01, 0.32, 0.63, 0.99]
seeds = [21, 4, 7, 9, 12, 45, 17, 5, 8, 10]
methods = {
    "batchnorm": {"enable_dropout": False, "enable_pool_dropout": False},
    "conv__layer_dropout": {"enable_dropout": True, "enable_pool_dropout": False},
    "pool__layer_dropout": {"enable_dropout": False, "enable_pool_dropout": True},
    "pool__and_conv_layer_dropout": {"enable_dropout": True, "enable_pool_dropout": True},
}


def compare(dataset_size):
    # Clean up here remove redundancy
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--save_path", default=save_path)  #
    parser.add_argument("--dataset", default=dataset)  #
    parser.add_argument("--number_iters", default=10)
    parser.add_argument("--device", default=find_best_device())  # find_best_device
    parser.add_argument("--bilinear_method", default=False)  # bilinear_method

    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--validation_size", default=0.33)
    parser.add_argument("--binary", default=True)
    parser.add_argument("--dropout_prob", default=0.5)
    parser.add_argument("--dropout_pool_prob", default=0.5)

    args = parser.parse_args()
    args.number_iters = int(args.number_iters)
    args.epochs = int(args.epochs)
    args.save_path = f"results/{args.save_path}"

    for method in methods:
        for train_size in dataset_size:
            for iteration in range(args.number_iters):
                # Get learning rate and momentum
                lr = 0.001
                momentum = 0.9
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
                    turn_off_wandb=False,
                )
                res = arrayify_results(stored_metrics, args.save_path)
            upload_file(file_path=save_path, file_name=f"{save_path}_{method}_{train_size}")


if __name__ == "__main__":
    get_data()
    compare(dataset_size=dataset_size)
    upload_file(file_path=save_path, file_name=f"{save_path.split('/')[1]}")
