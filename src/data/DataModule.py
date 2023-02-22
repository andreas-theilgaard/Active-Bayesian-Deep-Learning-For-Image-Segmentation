from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import torch


class SegmentationData(Dataset):
    def __init__(
        self,
        dataset: str,
        img_height: int,
        img_width: int,
        to_binary=False,
        extra_path=None,
        augment=None,
    ):

        self.img_height = img_height
        self.img_width = img_width
        self.dataset_type = dataset
        self.extra_path = extra_path
        self.augment = augment
        self.to_binary = to_binary

        # if split dataset: caravana, extra_path: image
        self.path = (
            self.dataset_type if not self.extra_path else f"{self.dataset_type}/{self.extra_path}"
        )
        self.data_path = f"data/raw/{self.path}"
        self.images = [x for x in os.listdir(f"{self.data_path}/image") if x != ".DS_Store"]
        # print(f"Images in total: {len(self.images)}")
        # self.images = glob.glob(os.path.join(f"{self.data_path}/image",'*'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(f"{self.data_path}/image", self.images[idx])
        mask_path = os.path.join(
            f"{self.data_path}/label", self.images[idx].replace(".jpg", "_mask.gif")
        )
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = image.resize((self.img_height, self.img_width), resample=Image.BICUBIC)
        mask = mask.resize((self.img_height, self.img_width), resample=Image.NEAREST)

        image = np.array(image)
        image = image / 255.0
        mask = np.array(mask, dtype=np.int64)

        if self.to_binary and self.dataset_type != "membrane":  # membrane already binary
            mask[mask > 0] = 1
        # if self.dataset_type == 'membrane':
        #    mask = mask#mask/255

        image = torch.tensor(image)
        mask = torch.tensor(mask)

        if image.shape[0] in [self.img_height, self.img_height]:
            image = image.permute(2, 0, 1)

        return (image, mask)