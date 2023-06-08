from src.data.dataloader import data_from_index
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

csfont = {"fontname": "sans-serif"}


def mask_list(train_loader, masks):
    if masks:
        for batch in train_loader:
            _, masks, _ = batch
        return masks
    else:
        for batch in train_loader:
            images, _, _ = batch
        return images


def plot_selected_images(dataset, methods: list, acq_funcs: list, n: int = 5, seed=21, masks_=True):
    """
    dataset: The dataset of interest
    methods: The methods of interest, possible [Random,RandomDE,MCD,DeepEnsemble,Laplace]
    acq_funcs: The acquisition function of interest [Random,ShanonEntropy,BALD,JensenDivergence]
    n: The number of iterations to visualize
    """
    assert seed in [4, 7, 21]
    assert len(
        [x for x in methods if x in ["Random", "RandomDE", "MCD", "DeepEnsemble", "Laplace"]]
    ) == len(
        methods
    ), "All Method Must Be In The Following Array [Random,RandomDE,MCD,DeepEnsemble,Laplace]"
    assert len(
        [x for x in acq_funcs if x in ["Random", "ShanonEntropy", "BALD", "JensenDivergence"]]
    ) == len(
        acq_funcs
    ), "All Acquisition Function Must Be In The Following Array [Random,ShanonEntropy,BALD,JensenDivergence]"
    translator = {
        "JensenDivergence": "JSD",
        "DeepEnsemble": "DE",
        "MCD": "MCD",
        "Random": "Random",
        "RandomDE": "RandomDE",
        "ShanonEntropy": "Entropy",
        "BALD": "BALD",
        "Laplace": "Laplace",
    }
    cols = []

    for method in methods:
        for acq_func in acq_funcs:
            if method in ["RandomDE", "Random"]:
                if acq_func == "Random":
                    cols.append((method, acq_func))
            else:
                if acq_func != "Random":
                    cols.append((method, acq_func))

    bigfig = plt.figure(layout="constrained", figsize=(15, 5))
    bigfig.subplots_adjust(top=0.8, wspace=0.00, hspace=0.00)
    subfigs = bigfig.subfigures(n, len(cols), wspace=0.00)

    for i, col in enumerate(cols):
        # Get Relevant DataFrame And Pick Masks Here
        for j in range(n):
            if col[1] == "Random":
                tmp = pd.read_json(
                    f"HPC_results/active_learning/AL_results/{dataset}_Random_{seed}_17_BatchNorm.json"
                )
            elif col[0] == "DeepEnsemble":
                tmp = pd.read_json(
                    f"HPC_results/active_learning/AL_results/{dataset}_{col[1]}_{seed}_17,8,42,19,5_DeepEnsemble.json"
                )
            else:
                tmp = pd.read_json(
                    f"HPC_results/active_learning/AL_results/{dataset}_{col[1]}_{seed}_17_{col[0]}.json"
                )
            indices = tmp.loc[j + 1, "labels_added"]
            train_loader, _, _, train_idx, _, _ = data_from_index(
                dataset,
                batch_size=4,
                to_binary=True,
                train_idx=indices,
                val_idx=[],
                unlabeled_pool_idx=[],
                num_workers=0,
                seed=21,
            )
            masks = mask_list(train_loader, masks_)
            # print(masks.shape)
            cur_axes = subfigs[j, i].subplots(1, 2, sharey=True)
            if i == 0:
                cur_axes[0].set_ylabel(
                    f"Iter {j+1}", size=14, rotation=0, labelpad=25, weight="bold", **csfont
                )  # add a y-label to the first subplot
            for k, ax in enumerate(cur_axes):
                if not masks_:
                    ax.imshow(masks[k], cmap="gray")
                else:
                    ax.imshow(masks[k])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")
            if j == 0:
                subfigs[j, i].suptitle(
                    f"{translator[col[0]]}-{translator[col[1]]}", weight="bold", **csfont, size=14
                )

    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.00,hspace=0.00)
    plt.show()


if __name__ == "__main__":
    plot_selected_images(
        "membrane",
        ["Random", "DeepEnsemble"],
        ["Random", "ShanonEntropy"],
        n=5,
        seed=4,
        masks_=False,
    )
