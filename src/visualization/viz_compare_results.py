import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csfont = {"fontname": "Times New Roman"}


def show_saidu_results():
    batchnorm_res = (
        "/Users/andreastheilgaard/Desktop/U_Net_experiments/results/membrane_0_0.5_200.npy"
    )
    dropout_res = (
        "/Users/andreastheilgaard/Desktop/U_Net_experiments/results/membrane_1_0.5_200.npy"
    )
    dropout_poolres = (
        "/Users/andreastheilgaard/Desktop/U_Net_experiments/results/membrane_2_0.5_200.npy"
    )
    dropout_pool_and_lay_res = (
        "/Users/andreastheilgaard/Desktop/U_Net_experiments/results/membrane_3_0.5_200.npy"
    )
    # 0.01,0.32,0.63
    idx = {0: 0.01, 3: 0.32, 6: 0.63}  # [0,3,6]
    saidu_results = {
        "batchnorm": batchnorm_res,
        "dropout": dropout_res,
        "pool_dropout": dropout_poolres,
        "pool_and_layer_dropout": dropout_pool_and_lay_res,
    }
    fig, axes = plt.subplots(3, 4)
    fig.set_size_inches(11, 6)
    for i, id in enumerate(idx):
        for j, method in enumerate(saidu_results):
            big_array = np.load(saidu_results[method], allow_pickle=True)
            res = big_array[0][4][id]
            for x in range(1, big_array.shape[0]):
                res = np.vstack((res, big_array[x][4][id]))

            mu = np.mean(res, axis=0)
            sigma = np.std(res, axis=0)
            x_data = [x for x in range(len(mu))]
            axes[i, j].plot(mu, color="navy")
            axes[i, j].fill_between(
                x_data, (mu - (sigma * 2)), (mu + (sigma * 2)), color="mediumpurple", alpha=0.5
            )
            axes[i, j].set_title(f"DS size {idx[id]}% and {method}", fontsize=10, **csfont)
            if j == 0:
                axes[i, j].set_ylabel(f"Dice coef", fontsize=14, **csfont)
            axes[i, j].set_xlabel("Epochs", fontsize=14, **csfont)
    plt.show()


show_saidu_results()


membrane_001 = "compare_results/membrane_size=0.01.json"
membrane_032_063 = "compare_results/membrane_size=0.32_0.63.json"
# df.columns
fig, axes = plt.subplots(3, 4)
fig.set_size_inches(16, 11)
df = pd.concat([pd.read_json(membrane_001), pd.read_json(membrane_032_063)])

df.columns
sizes = [0.01, 0.32, 0.63]
methods = df["method"].unique()
for i, size in enumerate(sizes):
    for j, method in enumerate(methods):
        res = df[(df["method"] == method) & (df["train_size"] == size)]["val_dice"]
        for x in range(res.shape[0]):
            if x == 0:
                arr = np.array(res.iloc[x])
            else:
                arr = np.vstack((arr, np.array(res.iloc[x])))
        mu = np.mean(arr, axis=0)
        sigma = np.std(arr, axis=0)
        if isinstance(mu, int):
            x_data = [1]
        else:
            x_data = [x for x in range(len(mu))]
        # epochs =
        axes[i, j].plot(mu, color="navy")
        axes[i, j].fill_between(
            x_data, (mu - (sigma * 2)), (mu + (sigma * 2)), color="mediumpurple", alpha=0.5
        )
        axes[i, j].set_title(
            f"DS size of {size}% and {method}",
            fontsize=10,
            **csfont,
        )
        if j == 0:
            axes[i, j].set_ylabel(f"Dice Coef", fontsize=14, **csfont)
        if i == 2:
            axes[i, j].set_xlabel("Epochs", fontsize=14, **csfont)


plt.show()


# res = pd.read_json("compare_results/new_membrane_dropout_test.json")
# res.columns
# var_ = "val_pixel_accuracy"
# train_size = 0.01
# method = "batchnorm"
# res.columns
# res["method"].unique()

# csfont = {"fontname": "Times New Roman"}


# def plot_comare_results(df, var_, train_size, method, dataset=None):
#     res = df[(df["method"] == method) & (df["train_size"] == train_size)][var_]
#     for i in range(res.shape[0]):
#         if i == 0:
#             arr = np.array(res.iloc[i])
#         else:
#             arr = np.vstack((arr, np.array(res.iloc[i])))
#     mu = np.mean(arr, axis=0)
#     sigma = np.std(arr, axis=0)
#     x_data = [x for x in range(len(mu))]
#     # epochs =
#     plt.plot(mu, color="navy")
#     plt.fill_between(
#         x_data, (mu - (sigma * 2)), (mu + (sigma * 2)), color="mediumpurple", alpha=0.5
#     )
#     plt.title(
#         f"{var_} results using a dataset size of {int(train_size*100)}% and {method}",
#         fontsize=24,
#         **csfont,
#     )
#     plt.ylabel(f"{var_}", fontsize=14, **csfont)
#     plt.xlabel("Epochs", fontsize=14, **csfont)
#     plt.show()


# # plot_comare_results(res,var_,train_size,method)
