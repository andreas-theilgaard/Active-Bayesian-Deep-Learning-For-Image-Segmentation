import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import torch.nn.functional as F


def load_ColorMapper():
    with open("data/color_mapping/ColorMapper.json", "r") as f:
        ColorMapper = json.load(f)
    return ColorMapper


def viz_mask_helper(mask, dataset_type):
    ColorMapper = load_ColorMapper()

    seg_arr = mask.argmax(dim=2)
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]
    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(2):
        seg_img[seg_arr == c] = tuple(ColorMapper["membrane"][str(c)])
    return seg_img.astype(np.uint8)


def mask_to_class_channel(mask, n_classes):
    mask = F.one_hot(mask, n_classes)
    mask = mask.cpu()
    return mask


def plot_prediction_batch(
    images, masks, predictions, save_, dataset_type="membrane", dataset=None, save_path=None
):
    assert images.shape[0] == masks.shape[0]
    assert masks.shape[0] == predictions.shape[0]
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
        mask = mask_to_class_channel(mask, 2)
        mask = viz_mask_helper(mask, dataset_type)

        if is_one_dim:
            axes[1].imshow(mask, cmap="viridis")
            axes[1].axis("off")
        else:
            axes[i, 1].imshow(mask, cmap="viridis")
            axes[i, 1].axis("off")

        # pred mask
        pred_mask = mask_to_class_channel(
            (((torch.sigmoid(pred).cpu()) > 0.5).float()).type(torch.int64), 2
        )  # mask_to_class_channel(pred.argmax(dim=0),2)
        pred_mask = viz_mask_helper(pred_mask.squeeze(0), dataset_type)

        if is_one_dim:
            axes[2].imshow(pred_mask, cmap="viridis")
            axes[2].axis("off")
        else:
            axes[i, 2].imshow(pred_mask, cmap="viridis")
            axes[i, 2].axis("off")
    if save_:
        print(f"{save_path}.png")
        fig.savefig(f"{save_path}.png", dpi=1000)
    return fig
