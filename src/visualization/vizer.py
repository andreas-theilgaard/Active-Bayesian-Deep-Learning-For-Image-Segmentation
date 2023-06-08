import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csfont = {"fontname": "sans-serif"}


def show_compare_results(dataset, cols, sizes, label=None, save_path=None, methods=None):
    assert len(cols) <= 3, "Only three columns can be plotted"
    try:
        df = pd.read_json(f"results/compare_results/{dataset}.json")
        print(df.columns)
    except:
        print(f"{dataset} not found. Sure the dataset has been specified correctly?")
        return NameError
    methods = df["method"].unique() if not methods else methods
    method_mapper = {
        "batchnorm": "BatchNorm",
        "conv__layer_dropout": "Conv Dropout",
        "pool__layer_dropout": "MaxPool Dropout",
        "pool__and_conv_layer_dropout": "Conv + MaxPool Dropout",
    }
    colors = [("navy", "mediumpurple"), ("orangered", "salmon"), ("yellowgreen", "darkgreen")]
    title_mapper = {
        "DIC_C2DH_Hela": "DIC-Hela",
        "PhC-C2DH-U373": "PhC-U373",
        "membrane": "Membrane",
        "warwick": "GlaS",
    }

    fig, axes = plt.subplots(len(sizes), len(methods))
    fig.set_size_inches(20, 12)
    fig.suptitle(
        f"Comparison of dropout configurations for {title_mapper[dataset]}",
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
                    f"{method_mapper[method]}: TS {round(100*size)}%",
                    fontsize=10,
                    weight="bold",
                    **csfont,
                )
                if j == 0:
                    axes[i, j].set_ylabel(f"{label if label else col}", fontsize=14, **csfont)
                if i == 2:
                    axes[i, j].set_xlabel("Epochs", fontsize=14, **csfont)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    show_compare_results(
        "DIC_C2DH_Hela",
        ["val_dice"],
        [0.01, 0.32, 0.63],
        "Dice",
        save_path="Thesis/assets/DIC_C2DH_Hela_lower.png",
    )
    show_compare_results(
        "warwick",
        ["val_dice"],
        [0.01, 0.32, 0.63],
        "Dice",
        save_path="Thesis/assets/Warwick_lower.png",
    )
    show_compare_results(
        "PhC-C2DH-U373",
        ["val_dice"],
        [0.01, 0.32, 0.63],
        "Dice",
        save_path="Thesis/assets/PhC-C2DH-U373_lower.png",
    )
    show_compare_results(
        "membrane",
        ["val_dice"],
        [0.01, 0.32, 0.63],
        "Dice",
        save_path="Thesis/assets/membrane_lower.png",
    )
