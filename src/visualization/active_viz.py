import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_random = pd.read_json(
    "results/active_learning/train_val_PhC-C2DH-U373_BatchNorm_Random_21_17.json"
)
df_BALD = pd.read_json("results/active_learning/train_val_PhC-C2DH-U373_MCD_BALD_21_17.json")
df_Shanon = pd.read_json(
    "results/active_learning/train_val_PhC-C2DH-U373_MCD_ShanonEntropy_21_17.json"
)

#
df_random_queries = pd.read_json("results/active_learning/PhC-C2DH-U373_Random_21_17.json")
df_BALD_queries = pd.read_json("results/active_learning/PhC-C2DH-U373_BALD_21_17.json")
df_Shanon_queries = pd.read_json("results/active_learning/PhC-C2DH-U373_ShanonEntropy_21_17.json")

# Random
df_random_queries.loc[len(df_random_queries)] = [
    0,
    None,
    df_random_queries["Train_size"].min() - 2,
    df_random_queries["Unlabeled_size"].max() + 2,
]
df_random = pd.merge(
    df_random[["Query ID", "val_dice"]],
    df_random_queries,
    how="inner",
    left_on="Query ID",
    right_on="Query_id",
)
random_val_dice = df_random["val_dice"].apply(lambda x: x[-1])
random_xvals = df_random["Train_size"] / (df_random["Train_size"] + df_random["Unlabeled_size"])

# BALD
df_BALD_queries.loc[len(df_BALD_queries)] = [
    0,
    None,
    df_BALD_queries["Train_size"].min() - 2,
    df_BALD_queries["Unlabeled_size"].max() + 2,
]
df_BALD = pd.merge(
    df_BALD[["Query ID", "val_dice"]],
    df_BALD_queries,
    how="inner",
    left_on="Query ID",
    right_on="Query_id",
)
BALD_val_dice = df_BALD["val_dice"].apply(lambda x: x[-1])
BALD_xvals = df_BALD["Train_size"] / (df_BALD["Train_size"] + df_BALD["Unlabeled_size"])

# Shanon
df_Shanon_queries.loc[len(df_Shanon_queries)] = [
    0,
    None,
    df_Shanon_queries["Train_size"].min() - 2,
    df_Shanon_queries["Unlabeled_size"].max() + 2,
]
df_Shanon = pd.merge(
    df_Shanon[["Query ID", "val_dice"]],
    df_Shanon_queries,
    how="inner",
    left_on="Query ID",
    right_on="Query_id",
)
Shanon_val_dice = df_Shanon["val_dice"].apply(lambda x: x[-1])
Shanon_xvals = df_Shanon["Train_size"] / (df_Shanon["Train_size"] + df_Shanon["Unlabeled_size"])


plt.plot(random_xvals, random_val_dice, color="blue", label="Random")
plt.plot(BALD_xvals, BALD_val_dice, color="red", label="BALD")
plt.plot(Shanon_xvals, Shanon_val_dice, color="green", label="ShanonEntropy")
plt.legend()
plt.show()
