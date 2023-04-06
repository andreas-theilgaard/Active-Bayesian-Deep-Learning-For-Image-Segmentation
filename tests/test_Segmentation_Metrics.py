import torch
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex, BinaryF1Score
import pytest
import os
from src.Metrics.SegmentationMetrics import SegmentationMetrics


class SegmentationMetricsOld:
    def __init__(self, multiclass=False):
        self.multiclass = multiclass

    def Dice_Coef(self, yhat, y_true):
        assert torch.is_tensor(yhat) == True
        assert torch.is_tensor(y_true) == True

        if yhat.min().item() < 0.0:
            yhat = (torch.sigmoid(yhat) >= 0.50).float()

        intersection = (yhat.flatten() * y_true.flatten()).sum()
        union = yhat.flatten().sum() + y_true.flatten().sum()
        numeric_stability = 1e-8
        return (2 * intersection) / (union + numeric_stability)

    def Dice_Coef_Confusion(self, yhat, y_true):
        assert torch.is_tensor(yhat) == True
        assert torch.is_tensor(y_true) == True

        if yhat.min().item() < 0.0:
            yhat = (torch.sigmoid(yhat) >= 0.50).float()

        TP = (yhat.flatten() * y_true.flatten()).sum()
        FN = y_true[yhat == 0].sum()
        FP = yhat[y_true == 0].sum()
        TN = yhat.numel() - TP - FN - FP
        numeric_stability = 1e-8
        return (2 * TP) / (2 * TP + FN + FP + numeric_stability)

    def IOU_(self, yhat, y_true):
        assert torch.is_tensor(yhat) == True
        assert torch.is_tensor(y_true) == True

        if yhat.min().item() < 0.0:
            yhat = (torch.sigmoid(yhat) >= 0.50).float()

        TP = (yhat.flatten() * y_true.flatten()).sum()
        FN = y_true[yhat == 0].sum()
        FP = yhat[y_true == 0].sum()
        TN = yhat.numel() - TP - FN - FP
        # numeric_stability = 1e-8
        IOU = TP / (TP + FN + FP)
        return IOU

    def torch_their_dice(self, pred, mask):
        assert torch.is_tensor(pred) == True
        assert torch.is_tensor(mask) == True
        mask = mask.unsqueeze(1)
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * mask, dim=(1, 2, 3))
        union = torch.sum(mask, dim=(1, 2, 3)) + torch.sum(pred, dim=(1, 2, 3))
        smooth = 1e-12
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return torch.mean(dice)

    def pixel_acc(self, yhat, y_true):
        assert torch.is_tensor(yhat) == True
        assert torch.is_tensor(y_true) == True
        if yhat.min().item() < 0.0:
            yhat = (torch.sigmoid(yhat) >= 0.50).float()

        TP = (yhat.flatten() * y_true.flatten()).sum()
        FN = y_true[yhat == 0].sum()
        FP = yhat[y_true == 0].sum()
        TN = yhat.numel() - TP - FN - FP

        pixel_acc = (TP + TN) / (TP + FN + FP + TN)
        return pixel_acc


@pytest.mark.skipif(
    not os.path.exists("test_assets/predictionsNormal.pth"), reason="Data files not found"
)
def test_segmentation_metrics():
    masks = torch.load("test_assets/masksNormalQ.pth")
    predictions = torch.load("test_assets/predictionsNormalQ.pth")

    Seg_Metrics_Old = SegmentationMetricsOld()
    Dice_Old = Seg_Metrics_Old.Dice_Coef(predictions.squeeze(1), masks).item()
    Dice_Confuse_Old = Seg_Metrics_Old.Dice_Coef_Confusion(predictions.squeeze(1), masks).item()

    IOU_Old = Seg_Metrics_Old.IOU_(predictions.squeeze(1), masks).item()
    Soft_Dice_Old = Seg_Metrics_Old.torch_their_dice(predictions, masks).item()
    pixel_acc_Old = Seg_Metrics_Old.pixel_acc(predictions.squeeze(1), masks).item()

    assert Dice_Old == Dice_Confuse_Old

    dice = BinaryF1Score()
    pixel_acc = BinaryAccuracy(validate_args=True)
    IOU_score = BinaryJaccardIndex()
    Dice_Torch = dice(predictions.squeeze(1), masks.type(torch.float32)).item()
    pixel_acc_Torch = pixel_acc(predictions.squeeze(1), masks.type(torch.float32)).item()
    IOU_Torch = IOU_score(predictions.squeeze(1), masks.type(torch.float32)).item()
    print(Dice_Torch)
    assert IOU_Old == IOU_Torch
    assert pixel_acc_Old == pixel_acc_Torch
    assert Dice_Torch == Dice_Old

    Seg_Metrics = SegmentationMetrics(torch_metrics=False)
    Dice, IOU, Acc, Soft_Dice = Seg_Metrics.Calculate_Segmentation_Metrics(
        predictions.squeeze(1), masks
    )

    assert Dice == Dice_Old
    assert IOU == IOU_Torch
    assert Acc == pixel_acc_Old
    assert Soft_Dice == Soft_Dice_Old

    Seg_Metrics_Torch = SegmentationMetrics(torch_metrics=True)
    Dice_T, IOU_T, Acc_T, Soft_Dice_T = Seg_Metrics_Torch.Calculate_Segmentation_Metrics(
        predictions.squeeze(1), masks
    )

    assert Dice == Dice_T
    assert IOU == IOU_T
    assert Acc == Acc_T
    assert Soft_Dice == Soft_Dice_T


if __name__ == "__main__":
    test_segmentation_metrics()
