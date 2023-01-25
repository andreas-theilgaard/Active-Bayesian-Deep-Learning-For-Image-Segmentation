# quick test for packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def find_best_device():

    if torch.cuda.is_available():
        device = torch.device("gpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


a = torch.tensor([3])
