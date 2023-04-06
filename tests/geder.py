import torch
import torch.nn as nn
from backpack import backpack, extend
from backpack.extensions import KFAC
import torch.nn.functional as F

import numpy as np

tensor = np.zeros((4, 64, 64, 64))
matrix = np.ones((64, 64))

reshaped_tensor = np.reshape(tensor, (4, -1))
result = np.matmul(reshaped_tensor, matrix)
result = np.reshape(result, (4, 64, 64, 64))

print(result.shape)  # (4, 64, 64, 64)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


model = MyModel()
loss_func = torch.nn.BCEWithLogitsLoss()

extend(model)
loss_func = extend(torch.nn.BCEWithLogitsLoss())

input = torch.rand(10, 10)
target = torch.rand(10).round()

import torch.nn.functional as F
from backpack import extend

output = model(input)

loss = loss_func(output.squeeze(1), target)

with backpack(KFAC()):
    loss.backward()
