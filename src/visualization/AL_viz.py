import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

type_ = "max"
# type_ = 'last'

datasets = ["PhC-C2DH-U373", "membrane", "warwick", "DIC_C2DH_Hela"]
for dataset in datasets:
    print(len([x for x in os.listdir(f"data/raw/{dataset}/image") if x != ".DS_Store"]))

csfont = {"fontname": "sans-serif"}
fig, axes = plt.subplots(1, 4)
fig.set_size_inches(15, 7)
title_mapper = {
    "DIC_C2DH_Hela": "DIC-Hela",
    "PhC-C2DH-U373": "PhC-U373",
    "membrane": "Membrane",
    "warwick": "GlaS",
}

colors = ["black", "mediumpurple", "deeppink", "darkgreen"]

metric = "val_dice"  # ,'val_ECE'


def get_MAP_perform(dataset):
    df = pd.read_json(f"HPC_results/MAP/{dataset}_5_PoolDropout.json")
    return np.array(df["val_dice"][0]).max()


for i, dataset in enumerate(datasets):
    df_BALD = pd.read_json(f"HPC_results/active_learning/train_val_{dataset}_MCD_BALD_21_17.json")
    df_Shanon = pd.read_json(
        f"HPC_results/active_learning/train_val_{dataset}_MCD_ShanonEntropy_21_17.json"
    )
    df_random = pd.read_json(
        f"HPC_results/active_learning/train_val_{dataset}_MCD_Random_21_17.json"
    )
    df_Jensen = pd.read_json(
        f"HPC_results/active_learning/train_val_{dataset}_MCD_JensenDivergence_21_17.json"
    )
    df_random_queries = pd.read_json(f"HPC_results/active_learning/{dataset}_Random_21_17_MCD.json")
    df_BALD_queries = pd.read_json(f"HPC_results/active_learning/{dataset}_BALD_21_17_MCD.json")
    df_Shanon_queries = pd.read_json(
        f"HPC_results/active_learning/{dataset}_ShanonEntropy_21_17_MCD.json"
    )
    df_jensen_queries = pd.read_json(
        f"HPC_results/active_learning/{dataset}_JensenDivergence_21_17_MCD.json"
    )

    # Random
    df_random = pd.merge(
        df_random[["Query ID", metric]],
        df_random_queries,
        how="inner",
        left_on="Query ID",
        right_on="Query_id",
    )
    random_val_dice = df_random[metric].apply(
        lambda x: np.array(x).max() if type_ == "max" else x[-1]
    )
    random_xvals = df_random["Train_size"] / (df_random["Train_size"] + df_random["Unlabeled_size"])

    # BALD
    df_BALD = pd.merge(
        df_BALD[["Query ID", metric]],
        df_BALD_queries,
        how="inner",
        left_on="Query ID",
        right_on="Query_id",
    )
    BALD_val_dice = df_BALD[metric].apply(lambda x: np.array(x).max() if type_ == "max" else x[-1])
    BALD_xvals = df_BALD["Train_size"] / (df_BALD["Train_size"] + df_BALD["Unlabeled_size"])

    # Shanon
    df_Shanon = pd.merge(
        df_Shanon[["Query ID", metric]],
        df_Shanon_queries,
        how="inner",
        left_on="Query ID",
        right_on="Query_id",
    )
    Shanon_val_dice = df_Shanon[metric].apply(
        lambda x: np.array(x).max() if type_ == "max" else x[-1]
    )
    Shanon_xvals = df_Shanon["Train_size"] / (df_Shanon["Train_size"] + df_Shanon["Unlabeled_size"])

    # Jensen
    df_Jensen = pd.merge(
        df_Jensen[["Query ID", metric]],
        df_jensen_queries,
        how="inner",
        left_on="Query ID",
        right_on="Query_id",
    )
    Jensen_val_dice = df_Jensen[metric].apply(
        lambda x: np.array(x).max() if type_ == "max" else x[-1]
    )
    Jensen_xvals = df_Jensen["Train_size"] / (df_Jensen["Train_size"] + df_Jensen["Unlabeled_size"])

    ##
    axes[i].set_title(title_mapper[dataset], fontsize=10, weight="bold", **csfont)
    if i == 0:
        axes[i].set_ylabel(f"{metric}", fontsize=14, **csfont)
    axes[i].plot(random_xvals, random_val_dice, color=colors[0], label="Random")
    axes[i].plot(BALD_xvals, BALD_val_dice, color=colors[1], label="BALD")
    axes[i].plot(Shanon_xvals, Shanon_val_dice, color=colors[2], label="ShanonEntropy")
    axes[i].plot(Jensen_xvals, Jensen_val_dice, color=colors[3], label="Jensen")
    axes[i].legend()
    axes[i].set_xlabel("% of labeled images", fontsize=8, **csfont)
    if metric == "val_dice":
        MAP_Perform = get_MAP_perform(dataset)
    axes[i].plot(
        Jensen_xvals,
        [MAP_Perform for x in range(len(Jensen_xvals))],
        color="black",
        linestyle="dotted",
    )

plt.savefig(f"AL_tmp_results_{metric}.png", dpi=1200)
# plt.show()
