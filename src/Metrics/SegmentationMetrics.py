import torch
import numpy as np
import torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex, BinaryF1Score


class SegmentationMetrics:
    def __init__(self, device="cpu", multiclass=False, numeric_stability=1e-8, torch_metrics=False):
        self.multiclass = multiclass
        self.numeric_stability = numeric_stability
        self.torch_metrics = torch_metrics
        self.device = device

        self.TP = None
        self.FN = None
        self.FP = None
        self.TN = None

        if torch_metrics:
            # TorchMetrics
            self.dice = BinaryF1Score().to(self.device)
            self.pixel_acc = BinaryAccuracy().to(self.device)
            self.IOU_score = BinaryJaccardIndex().to(self.device)

    def ConfusionMatrix(self, y_hat, y_true):
        assert torch.is_tensor(y_hat) == True
        assert torch.is_tensor(y_true) == True
        if y_hat.min().item() < 0.0:
            y_hat = (torch.sigmoid(y_hat) >= 0.50).float()

        TP = (y_hat.flatten() * y_true.flatten()).sum()
        FN = y_true[y_hat == 0].sum()
        FP = y_hat[y_true == 0].sum()
        TN = y_hat.numel() - TP - FN - FP

        ##
        self.TP = TP
        self.FN = FN
        self.FP = FP
        self.TN = TN

    def Dice(self):
        return ((2 * self.TP) / (2 * self.TP + self.FN + self.FP + self.numeric_stability)).item()

    def IOU(self):
        return (self.TP / (self.TP + self.FN + self.FP)).item()

    def PixelAccuracy(self):
        return ((self.TP + self.TN) / (self.TP + self.FN + self.FP + self.TN)).item()

    def Soft_Dice(self, pred, mask):
        assert torch.is_tensor(pred) == True
        assert torch.is_tensor(mask) == True
        mask = mask.unsqueeze(1)
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask, dim=(1, 2, 3))
        union = torch.sum(mask, dim=(1, 2, 3)) + torch.sum(pred, dim=(1, 2, 3))
        dice = (2.0 * intersection + self.numeric_stability) / (union + self.numeric_stability)
        return torch.mean(dice).item()

    def Calculate_Segmentation_Metrics(self, y_hat, y_true):
        if self.torch_metrics:
            Dice = self.dice(y_hat, y_true.type(torch.float32)).item()
            Acc = self.pixel_acc(y_hat, y_true.type(torch.float32)).item()
            IOU = self.IOU_score(y_hat, y_true.type(torch.float32)).item()
        else:
            self.ConfusionMatrix(y_hat, y_true)
            Dice, IOU, Acc = self.Dice(), self.IOU(), self.PixelAccuracy()
        Soft_Dice = self.Soft_Dice(y_hat.unsqueeze(1), y_true)
        return (Dice, IOU, Acc, Soft_Dice)
