import torch
import argparse
from src.config import find_best_device
from src.experiments.experiment_utils import arrayify_results
from src.models.train_model import train

[
    (0.630103349685669, 0.628966212272644),
    (0.6722779273986816, 0.6731500029563904),
    (0.7230668663978577, 0.730646550655365),
    (0.5525455474853516, 0.5577081441879272),
    (0.5717694163322449, 0.5767558217048645),
    (0.7238587141036987, 0.7315257787704468),
]

parser = argparse.ArgumentParser(description="Training arguments")
parser.add_argument("--device", default=find_best_device())  # find_best_device
parser.add_argument("--binary", default=True)
parser.add_argument("--epochs", default=100)
parser.add_argument("--seed", default=21)

args = parser.parse_args()
args.epochs = int(args.epochs)

torch_seeds = [17, 8, 42, 19, 5]
datasets = ["membrane", "warwick", "PhC-C2DH-U373", "DIC_C2DH_Hela"]


def run_MAP():
    for torch_seed in torch_seeds:
        for dataset in datasets:
            save_path = f"results/MAP/{dataset}_{torch_seed}"
            stored_metrics = train(
                dataset=dataset,
                train_size="100%",
                epochs=args.epochs,
                device=args.device,
                binary=args.binary,
                enable_dropout=False,
                enable_pool_dropout=False,
                model_method="batchnorm",
                seed=args.seed,
                torch_seed=torch_seed,
                turn_off_wandb=True,
                save_model=True,
            )
            res = arrayify_results(stored_metrics, save_path)


if __name__ == "__main__":
    run_MAP()
