import argparse

# from src.data.get_data import get_data
import os
import wandb
import subprocess
from src.config import find_best_device

print(os.listdir())
print(os.listdir("data"))

# dataset = os.environ['dataset']
# print(f"Using dataset: {dataset}")
# Download data
# get_data()
##############
save_path = os.environ["save_path"]
dataset = os.environ["dataset"]

bash_cmd = f"wandb server start"
process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

bash_cmd = f"wandb login 82a3b5a7b8ff626de2d5ae45becdac5fa040d0f7"
process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()


def init_env():
    if not os.path.isdir("compare_results/"):
        os.mkdir("compare_results")
    bash_cmd = f"wandb login 82a3b5a7b8ff626de2d5ae45becdac5fa040d0f7"
    process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if not error:
        print("Logged succesfully in to wandb")


from src.models.train_model import train
from src.experiments.experiment_utils import arrayify_results


# hyper_params in dictinary based on dataset type
dataset_size = [0.01]  # , 0.32, 0.63, 0.99]  # TO DO: do some more runs with other values
# 5 and 6 maybe bad choice??
seeds = [182]  # , 322, 291, 292,261]# 122, 53, 261, 427, 174, 128]

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
    "PhC-C2DH-U373": {
        0.01: {"lr": 0.001, "momentum": 0.9},
        0.32: {"lr": 0.0001, "momentum": 0.6},
        0.63: {"lr": 0.0001, "momentum": 0.1},
        0.99: {"lr": 0.0001, "momentum": 0.1},
    },
    "DIC_C2DH_Hela": {
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
    parser.add_argument("--save_path", default=save_path)
    parser.add_argument("--dataset", default=dataset)
    parser.add_argument("--number_iters", default=1)
    parser.add_argument("--device", default=find_best_device())  # find_best_device
    parser.add_argument("--bilinear_method", default=True)  # bilinear_method

    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--epochs", default=1)
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


import subprocess


def upload_result(save_path):
    bash_cmd = f"/root/google-cloud-sdk/bin/gsutil cp results/{save_path}.json gs://compare_methods_results"
    process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


if __name__ == "__main__":
    init_env()
    compare(dataset_size=dataset_size)
    upload_result(save_path=save_path)
