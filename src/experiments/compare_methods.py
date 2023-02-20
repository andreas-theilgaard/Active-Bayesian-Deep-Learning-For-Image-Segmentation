import torch
import argparse
from src.models.train_model import train
from src.experiments.experiment_utils import arrayify_results

# hyper_params in dictinary based on dataset type
dataset_size = [0.01]  # [0.01,0.32,0.63,0.99] # TO DO: do some more runs with other values


def compare(dataset_size):
    # Clean up here remove redundancy
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--save_path", default="test")
    parser.add_argument("--dataset", default="membrane")
    parser.add_argument("--number_iters", default=10)

    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--epochs", default=50)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--validation_size", default=0.33)
    parser.add_argument("--binary", default=True)
    args = parser.parse_args()
    args.number_iters = int(args.number_iters)
    args.lr = float(args.lr)
    args.epochs = int(args.epochs)
    args.save_path = f"compare_results/{args.save_path}"
    print(args)

    for train_size in dataset_size:
        for iteration in range(args.number_iters):
            print(train_size, iteration)
            experiment_name = f"{args.dataset}_{train_size}"
            job_type = f"run {iteration+1}"
            stored_metrics = train(
                dataset=args.dataset,
                train_size=train_size,
                epochs=args.epochs,
                lr=args.lr,
                momentum=args.momentum,
                device=args.device,
                batch_size=args.batch_size,
                validation_size=args.validation_size,
                binary=args.binary,
                iter=iteration,
                experiment_name=experiment_name,
                job_type=job_type,
            )
            res = arrayify_results(stored_metrics, args.save_path)


if __name__ == "__main__":
    compare(dataset_size=dataset_size)


# import pandas as pd
# res = pd.read_json('compare_results/test.json')
