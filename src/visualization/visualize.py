import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import random
from src.visualization.viz_utils import styles, viz_mask_helper

# from src.data.dataloader import unwrap_batch
import torch
import numpy as np
from src.config import Config
import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
from src.config import Config
import random

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams.update({"font.size": 14})


def viz_data(
    sample=True,
    save_=False,
    raw_path="data/raw",
    indices={"warwick": 276, "DIC_C2DH_Hela": 38, "PhC-C2DH-U373": 100, "membrane": 152},
):
    datasets = glob.glob(os.path.join(raw_path, "*"))
    n_classes = Config.n_classes
    fig, axes = plt.subplots(2, 4)
    fig.set_size_inches(11, 6)
    for i, dataset in enumerate(datasets):
        label = dataset.split("/")[-1]
        for j, folder in enumerate(["image", "label"]):
            files = glob.glob(os.path.join(f"{dataset}/dataset/{folder}", "*"))
            print(len(files))
            idx = (
                indices[label] if not sample else (random.randint(0, len(files)) if j == 0 else idx)
            )
            picked_file = files[idx]

            img = Image.open(picked_file).convert("RGB") if j == 0 else Image.open(picked_file)
            img = img.resize((512, 512))
            axes[j, i].set_yticklabels([])
            axes[j, i].set_xticklabels([])
            if j == 0:
                axes[j, i].set_title(label, fontsize=18, weight="bold")
            if i == 0:
                axes[j, i].set_ylabel("Image" if j == 0 else "Label")
            if j == 1:
                axes[j, i].set_xlabel(f"{n_classes[label]} classes")
            axes[j, i].imshow(img)
    if save_:
        plt.savefig("Thesis/assets/initial_viz_data.png", dpi=styles.dpi_level)
    plt.show()


# viz_data(sample=True,save_=False)


def mask_to_class_channel(mask, n_classes):
    mask = mask.type(torch.uint8)
    # masks = tf.keras.utils.to_categorical(mask,self.n_classes) #require tensorflow
    masks = np.eye(n_classes, dtype="uint8")[mask]
    # masks = np.eye(256,dtype='uint8')[mask]
    masks = torch.from_numpy(masks)
    masks = masks.squeeze(1) if len(masks.shape) > 4 else masks
    masks = masks.permute(0, 3, 1, 2).type(torch.float32)
    return masks


def viz_batch(batch, dataset_type):
    # https://datascience.stackexchange.com/questions/40637/how-to-visualize-image-segmentation-results
    images, masks = batch
    # masks = masks.to(dtype=torch.int64)
    masks = mask_to_class_channel(masks, Config.n_classes[dataset_type])
    fig, axes = plt.subplots(2, images.shape[0])
    fig.set_size_inches(11, 6)

    for i in range(images.shape[0]):
        for j in range(2):
            axes[j, i].set_yticklabels([])
            axes[j, i].set_xticklabels([])
            # if j ==0:
            #    axes[j,i].set_title(label,fontsize=18,weight="bold")
            # if i ==0:
            #    axes[j,i].set_ylabel('Image' if j==0 else 'Label')
            # if j ==1:
            #    axes[j,i].set_xlabel(f"{n_classes[label]} classes")
            if j == 0:
                axes[j, i].imshow(images[i].permute(1, 2, 0))
            elif j == 1:
                processed_mask = viz_mask_helper(masks[i], dataset_type)
                axes[j, i].imshow(processed_mask, cmap="viridis")
    plt.show()
