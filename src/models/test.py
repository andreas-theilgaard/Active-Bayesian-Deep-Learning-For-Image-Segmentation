import torch
import torchmetrics


if __name__ == "__main__":
    A = torch.rand((4, 3, 64, 64))
    B = torch.rand((4, 3, 64, 64))
    C = A * B
    print(C)
