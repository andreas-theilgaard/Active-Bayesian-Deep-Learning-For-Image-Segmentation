import torch

test = torch.rand((4, 1024, 4, 4))


res = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)(test)
res.shape

ged = torch.cat([res, torch.rand(4, 512, 8, 8)])

ged.shape


ged = torch.cat([torch.rand(4, 512, 8, 8), res], dim=1)

ged.shape


torch.nn.Conv2d(1024, 512, kernel_size=3, padding=1)(ged).shape

##
import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

img_path = "data/raw/PhC-C2DH-U373/image/1.png"
image = Image.open(img_path).convert("RGB")  # convert("L")
image = np.asarray(image)
image = torch.tensor(image)
image = image.permute(2, 0, 1)
# image = image/255

image.shape
data = image.view(image.size(0), -1).float()
data.mean(1)
data.std(1)

(data - 84.1169) / (8.7073)


mean += data.mean(2).sum(0)
std += data.std(2).sum(0)
nb_samples += batch_samples


transform = torch.nn.Sequential(transforms.Normalize((0.2, 0.2, 0.2), (0.229, 0.224, 0.225)))
transform(image.float())
image.float()

image / 255

image

image.shape

###
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 3, 24, 24)

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)


dataset = MyDataset()
loader = DataLoader(dataset, batch_size=10, num_workers=0, shuffle=False)


ged = next(iter(loader))

ged.shape

ged.size(0)
hey = ged.view(10, ged.size(1), -1)

hey.mean(2).sum(0)


mean = 0.0
std = 0.0
nb_samples = 0.0
for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

###
masks = os.listdir("data/raw/PhC-C2DH-U373/label")
for mask in masks:
    mask_ = Image.open("data/raw/PhC-C2DH-U373/label" + "/" + mask)
    mask_ = mask_.resize((512, 512), resample=Image.Resampling.BILINEAR)
    mask_ = np.array(mask_)
    mask_[mask_ > 0] = 1
    if list(np.unique(mask_)) != [0, 1]:
        print(mask)

mask = Image.open("data/raw/PhC-C2DH-U373/label/0.png")
mask = np.array(mask)
np.unique(mask)
