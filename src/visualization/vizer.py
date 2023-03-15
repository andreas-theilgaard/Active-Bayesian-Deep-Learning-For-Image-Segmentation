import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csfont = {"fontname": "Times New Roman"}

# membrane_001 = "results/compare_results/membrane_size=0.01.json"
# membrane_032_063 = "results/compare_results/membrane_size=0.32_0.63.json"
# df.columns
# df = pd.concat([pd.read_json(membrane_001), pd.read_json(membrane_032_063)])
# df = pd.read_json('results/compare_results/DIC_C2DH_Hela.json')


def show_compare_results(dataset, cols, sizes, label=None, save_path=None, methods=None):
    assert len(cols) <= 3, "Only three columns can be plotted"
    try:
        df = pd.read_json(f"results/compare_results/{dataset}.json")
    except:
        print(f"{dataset} not found. Sure the dataset has been specified correctly?")
        return NameError
    methods = df["method"].unique() if not methods else methods
    method_mapper = {
        "batchnorm": "BatchNorm",
        "conv__layer_dropout": "Dropout",
        "pool__layer_dropout": "PoolDropout",
        "pool__and_conv_layer_dropout": "Conv and Pool Dropout",
    }
    colors = [("navy", "mediumpurple"), ("orangered", "salmon"), ("yellowgreen", "darkgreen")]
    title_mapper = {"DIC_C2DH_Hela": "DIC-Hela", "PhC-C2DH-U373": "PhC-U373"}

    fig, axes = plt.subplots(len(sizes), len(methods))
    fig.set_size_inches(15, 12)
    fig.suptitle(
        f"Comparison of dropout methods for {title_mapper[dataset]}",
        weight="bold",
        fontsize=20,
        **csfont,
    )
    for col_i, col in enumerate(cols):
        for i, size in enumerate(sizes):
            for j, method in enumerate(methods):
                res = df[(df["method"] == method) & (df["train_size"] == size)][col]
                for x in range(res.shape[0]):
                    if x == 0:
                        arr = np.array(res.iloc[x])
                    else:
                        arr = np.vstack((arr, np.array(res.iloc[x])))
                mu = np.mean(arr, axis=0)
                sigma = np.std(arr, axis=0)
                if isinstance(mu, np.float64):
                    x_data = [1]
                else:
                    x_data = [x for x in range(len(mu))]

                axes[i, j].plot(mu, color=colors[col_i][0])
                axes[i, j].fill_between(
                    x_data,
                    (mu - (sigma * 2)),
                    (mu + (sigma * 2)),
                    color=colors[col_i][1],
                    alpha=0.5,
                )
                axes[i, j].set_title(
                    f"{method_mapper[method]}: DS size of {round(100*size)}%",
                    fontsize=10,
                    weight="bold",
                    **csfont,
                )
                if j == 0:
                    axes[i, j].set_ylabel(f"{label if label else col}", fontsize=14, **csfont)
                if i == 2:
                    axes[i, j].set_xlabel("Epochs", fontsize=14, **csfont)
    if save_path:
        plt.savefig(save_path, dpi=1000)
    else:
        plt.show()


if __name__ == "__main__":
    # show_compare_results('DIC_C2DH_Hela',['val_dice'],[0.01, 0.32, 0.63],'Dice Score',save_path='results/assets/test.png')
    show_compare_results("PhC-C2DH-U373", ["val_dice"], [0.01, 0.32, 0.63], "Dice Score")


# df.columns
# sizes = [0.01, 0.32, 0.63]
# methods = df["method"].unique()
# for i, size in enumerate(sizes):
#     for j, method in enumerate(methods):
#         res = df[(df["method"] == method) & (df["train_size"] == size)]["val_dice"]
#         for x in range(res.shape[0]):
#             if x == 0:
#                 arr = np.array(res.iloc[x])
#             else:
#                 arr = np.vstack((arr, np.array(res.iloc[x])))
#         mu = np.mean(arr, axis=0)
#         sigma = np.std(arr, axis=0)
#         if isinstance(mu, int):
#             x_data = [1]
#         else:
#             x_data = [x for x in range(len(mu))]
#         # epochs =
#         axes[i, j].plot(mu, color="navy")
#         axes[i, j].fill_between(
#             x_data, (mu - (sigma * 2)), (mu + (sigma * 2)), color="mediumpurple", alpha=0.5
#         )
#         axes[i, j].set_title(
#             f"DS size of {size}% and {method}",
#             fontsize=10,
#             **csfont,
#         )
#         if j == 0:
#             axes[i, j].set_ylabel(f"Dice Coef", fontsize=14, **csfont)
#         if i == 2:
#             axes[i, j].set_xlabel("Epochs", fontsize=14, **csfont)


# plt.show()
