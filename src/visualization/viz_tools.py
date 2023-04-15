###################################################################################################################
#                                          Functions: viz_tools
#   * plot_prediction_batch() ->plot of img,mask and prediciton side by size
#
#   * viz_data() -> Shows images and masks from the datasets
#
#   *
#
###################################################################################################################
from src.visualization.viz_utils import styles
from src.ActiveLearning.AL_utils import unbinarize, binarize
from src.ActiveLearning.AcquisitionFunctions import ActiveLearningAcquisitions
import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import random
from src.visualization.viz_utils import styles
from src.config import Config
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams.update({"font.size": 14})


def plot_prediction_batch(
    images, masks, predictions, save_, dataset_type="membrane", dataset=None, save_path=None
):
    assert images.shape[0] == masks.shape[0]
    assert masks.shape[0] == predictions.shape[0]
    predictions = predictions.squeeze(1)
    plt.close("all")  # close plot windows, so they don't take up space
    is_one_dim = True if masks.shape[0] == 1 else False
    fig, axes = plt.subplots(masks.shape[0], 3)
    fig.set_size_inches(6, 7)

    for i in range(masks.shape[0]):
        img = images[i]
        mask = masks[i]
        pred = predictions[i]
        # img
        img = img.permute(1, 2, 0)
        if is_one_dim:
            if dataset != "warwick":
                axes[0].imshow(img.detach().cpu().numpy(), cmap="gray")
                axes[0].axis("off")
            else:
                axes[0].imshow(img.detach().cpu().numpy())
                axes[0].axis("off")
        else:
            if dataset != "warwick":
                axes[i, 0].imshow(img.detach().cpu().numpy(), cmap="gray")
                axes[i, 0].axis("off")
            else:
                axes[i, 0].imshow(img.detach().cpu().numpy())
                axes[i, 0].axis("off")

        # mask
        if is_one_dim:
            axes[1].imshow(mask, cmap="viridis")
            axes[1].axis("off")
        else:
            axes[i, 1].imshow(mask, cmap="viridis")
            axes[i, 1].axis("off")

        # pred mask
        pred_mask = ((torch.sigmoid(pred) > 0.5).float()).detach().cpu().numpy()
        if is_one_dim:
            axes[2].imshow(pred_mask, cmap="viridis")
            axes[2].axis("off")
        else:
            axes[i, 2].imshow(pred_mask, cmap="viridis")
            axes[i, 2].axis("off")
    if save_:
        print(f"{save_path}_predictions.png")
        fig.savefig(f"{save_path}.png", dpi=1000)
    return fig


def viz_batch(
    images,
    masks,
    predictions,
    cols,
    from_logits,
    reduction,
    save_,
    dataset_type="membrane",
    dataset=None,
    save_path=None,
):
    """
    images: [batch_size,n_classes,img_width,img_height]
    masks: [batch_size,img_width,img_height]
    predictions: [batch_size,n_classes,img_width,img_height]
    cols: List of ordering of cols, possible inputs are [img,mask,pred,err,var,entropy,mut_info,jsd]
    from_logits: bool Specifying whether predictions is logits or not
    reduction: Bool if true assuming predictions is of size [batch_size,n_predictions,n_classes,img_width,img_height] else as above
    save_: Bool whether plot should be saved or not
    dataset_type: Binary or Not ??
    dataset: the dataset that is used
    save_path: The path specifying where the plot should be saved if save_ is true
    """
    assert images.shape[0] == masks.shape[0]
    assert masks.shape[0] == predictions.shape[0]
    assert isinstance(cols, list)
    expect = ["img", "mask", "pred", "err", "var", "entropy", "mut_info", "jsd"]
    translator = {
        "mask": "Masks",
        "pred": "Predictions",
        "err": "Error",
        "var": "PV",
        "entropy": "Entropy",
        "mut_info": "MI",
        "jsd": "JSD",
    }
    cols_checker = [x for x in cols if x not in expect]
    assert (
        len(cols_checker) == 0
    ), f"Expected any columns in {expect}, but got {cols_checker} instead"

    cols_vars = {x: None for x in cols}

    if "img" in cols:
        cols_vars["img"] = images.permute(0, 2, 3, 1).detach().cpu().numpy()
    if "mask" in cols:
        cols_vars["mask"] = masks.detach().cpu().numpy()

    if from_logits:
        predictions = torch.sigmoid(predictions)

    if reduction:
        assert len(predictions.shape) == 5
        if predictions.shape[-1] == 1:
            predictions = binarize(predictions)
        Acq_funcs = ActiveLearningAcquisitions()
        Entropy, BALD, JSD = Acq_funcs.Get_All_Pixel_Wise(predictions)

        unbinarized_prediction = unbinarize(predictions)  # Get P(Y=1)
        var = unbinarized_prediction.var(1)  # [batch_size,img_height,img_width,n_classes=1]
        mean_prediction = torch.mean(
            unbinarized_prediction, dim=1
        )  # [batch_size,img_height,img_width,n_classes=1]
        var = var.squeeze(-1)  # [batch_size,img_height,img_width]
        mean_prediction = mean_prediction.squeeze(-1)  # [batch_size,img_height,img_width]
        mean_prediction = (mean_prediction > 0.5).float()
    else:
        mean_prediction = predictions.squeeze(1)
        mean_prediction = (mean_prediction > 0.5).float()

    error = (~masks.eq(mean_prediction)).float()

    if "pred" in cols:
        cols_vars["pred"] = mean_prediction.detach().cpu().numpy()
    if "err" in cols:
        cols_vars["err"] = error.detach().cpu().numpy()
    if "var" in cols:
        cols_vars["var"] = var.detach().cpu().numpy()
    if "entropy" in cols:
        cols_vars["entropy"] = Entropy.detach().cpu().numpy()
    if "mut_info" in cols:
        cols_vars["mut_info"] = BALD.detach().cpu().numpy()
    if "jsd" in cols:
        cols_vars["jsd"] = JSD.detach().cpu().numpy()

    # Plot Setup
    plt.close("all")  # close plot windows, so they don't take up space
    is_one_dim = True if masks.shape[0] == 1 else False
    fig, axes = plt.subplots(masks.shape[0], len(cols))
    fig.set_size_inches(6, 7)

    for i in range(masks.shape[0]):
        for j in range(len(cols)):

            if cols[j] == "img":
                if is_one_dim:
                    if dataset != "warwick":
                        axes[j].imshow(cols_vars[cols[j]][i], cmap="gray")
                        axes[j].axis("off")
                    else:
                        axes[j].imshow(cols_vars[cols[j]][i])
                        axes[j].axis("off")
                    if i == 0:
                        axes[j].set_title("Images")
                else:
                    if dataset != "warwick":
                        axes[i, j].imshow(cols_vars[cols[j]][i], cmap="gray")
                        axes[i, j].axis("off")
                    else:
                        axes[i, j].imshow(cols_vars[cols[j]][i])
                        axes[i, j].axis("off")
                    if i == 0:
                        axes[i, j].set_title("Images")
            else:
                if is_one_dim:
                    axes[j].imshow(cols_vars[cols[j]][i], cmap="viridis")
                    axes[j].axis("off")
                    if i == 0:
                        axes[j].set_title(translator[cols[j]])

                else:
                    axes[i, j].imshow(cols_vars[cols[j]][i], cmap="viridis")
                    axes[i, j].axis("off")
                    if i == 0:
                        axes[i, j].set_title(translator[cols[j]])
    if save_:
        fig.savefig(f"{save_path}_predictions.png", dpi=1200)
    return fig


def viz_data(
    sample=True,
    binarize=False,
    save_=False,
    show=False,
    raw_path="data/raw",
    indices={"warwick": 276, "DIC_C2DH_Hela": 38, "PhC-C2DH-U373": 100, "membrane": 152},
):
    datasets = Config.datasets
    n_classes = Config.n_classes
    fig, axes = plt.subplots(2, 4)
    fig.set_size_inches(11, 6)

    for i, dataset in enumerate(datasets):
        label = dataset.split("/")[-1]
        for j, folder in enumerate(["image", "label"]):
            files = glob.glob(os.path.join(f"{raw_path}/{dataset}/{folder}", "*"))
            idx = (
                indices[label] if not sample else (random.randint(0, len(files)) if j == 0 else idx)
            )
            picked_file = files[idx]

            img = Image.open(picked_file).convert("RGB") if j == 0 else Image.open(picked_file)

            img = img.resize((512, 512))

            if j == 1 and binarize:
                img = np.array(img)
                if dataset != "membrane":
                    img[img > 0] = 1
                else:
                    img = img / 255
                    img = img.astype(np.int64)

            axes[j, i].set_yticklabels([])
            axes[j, i].set_xticklabels([])
            if j == 0:
                axes[j, i].set_title(label, fontsize=18, weight="bold")
            if i == 0:
                axes[j, i].set_ylabel("Image" if j == 0 else "Label")
            # if j == 1:
            #    axes[j, i].set_xlabel(f"{n_classes[label] if not binarize else 2} classes")
            axes[j, i].imshow(img)
    if save_:
        plt.savefig("Thesis/assets/initial_viz_data_multiclass.png", dpi=styles.dpi_level)
    if show:
        plt.show()


if __name__ == "__main__":
    viz_data(
        sample=False,
        binarize=False,
        save_=True,
        show=False,
        raw_path="data/raw",
        indices={"warwick": 276, "DIC_C2DH_Hela": 38, "PhC-C2DH-U373": 100, "membrane": 152},
    )
