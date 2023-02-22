import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

res = pd.read_json("compare_results/test.json")
res.columns
var_ = "val_pixel_accuracy"
train_size = 0.01
method = "batchnorm"

csfont = {"fontname": "Times New Roman"}


def plot_comare_results(df, var_, train_size, method, dataset=None):
    res = df[(df["method"] == method) & (df["train_size"] == train_size)][var_]
    for i in range(res.shape[0]):
        if i == 0:
            arr = np.array(res.iloc[i])
        else:
            arr = np.vstack((arr, np.array(res.iloc[i])))
    mu = np.mean(arr, axis=0)
    sigma = np.std(arr, axis=0)
    x_data = [x for x in range(len(mu))]
    # epochs =
    plt.plot(mu, color="navy")
    plt.fill_between(
        x_data, (mu - (sigma * 2)), (mu + (sigma * 2)), color="mediumpurple", alpha=0.5
    )
    plt.title(
        f"{var_} results using a dataset size of {int(train_size*100)}% and {method}",
        fontsize=24,
        **csfont,
    )
    plt.ylabel(f"{var_}", fontsize=14, **csfont)
    plt.xlabel("Epochs", fontsize=14, **csfont)
    plt.show()


# plot_comare_results(res,var_,train_size,method)
