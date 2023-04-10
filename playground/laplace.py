import torch
from backpack import extend, backpack, extensions
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import datasets
from tqdm import tqdm
import random

seed = 7777

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(99999)  # Set Torch Seed
torch.cuda.manual_seed_all(99999)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# class ToyModel(nn.Module):
#     def __init__(self,n=2,out=2,hidden=20):
#         super().__init__()
#         self.hidden = hidden
#         self.n=n
#         self.out=out
#         self.features = nn.Sequential(
#             nn.Conv2d(3,3,1),
#             )
#         #self.last_layer = nn.Linear(3*64*64,2,bias=False)
#         self.last_layer = nn.Conv2d(3,2,kernel_size=1,bias=False)
#     def forward(self,x):
#         features = self.features(x)
#         #features = torch.flatten(features,1)
#         return self.last_layer(features)


from src.models.model import UNET

model = UNET(in_ch=3, out_ch=2)
# from src.data.dataloader import train_split
# data_loader,_,_,_,_,_ = train_split(0.5,dataset='PhC-C2DH-U373',batch_size=4,to_binary=True,seed=261)

# model = ToyModel()
X_train = torch.rand((40, 3, 64, 64))
y_train = torch.rand((40, 64, 64)).round().long()

opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

for it in tqdm(range(10)):
    y = model(X_train)
    l = F.cross_entropy(y, y_train)
    l.backward()
    opt.step()
    opt.zero_grad()
print(y.shape)
model.eval()
Weights_last = list(model.parameters())[-2]
extend(model.out)
loss_func = extend(nn.CrossEntropyLoss(reduction="sum"))
loss = loss_func(model(X_train), y_train)
with backpack(extensions.KFAC()):
    loss.backward()
A, B = Weights_last.kfac

if __name__ == "__main__":
    print(A[0][0])
    print(B[3][3])
