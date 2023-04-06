import torch
from torchmetrics import JaccardIndex, Dice
from torchmetrics.classification import (
    MulticlassAccuracy,
    BinaryAccuracy,
    BinaryJaccardIndex,
    BinaryF1Score,
)
from src.models.model_utils import SegmentationMetrics, Calibration_Scoring_Metrics
from torchmetrics.classification import BinaryCalibrationError


class CollectMetrics:
    def __init__(self, device, out_ch):

        self.device = device
        self.out_ch = out_ch
        # Define metrics that will be used
        self.dice = (
            BinaryF1Score().to(device)
            if out_ch == 1
            else Dice(average="micro", ignore_index=0).to(device)
        )
        self.pixel_acc = (
            MulticlassAccuracy(num_classes=out_ch, validate_args=True).to(device)
            if out_ch > 1
            else BinaryAccuracy(validate_args=True).to(device)
        )
        self.IOU_score = BinaryJaccardIndex().to(device)
        self.metrics = SegmentationMetrics()
        self.calibration_metrics = Calibration_Scoring_Metrics(
            nbins=15, multiclass=False, device=device
        )
        self.torch_ECE = BinaryCalibrationError(n_bins=15, norm="l1").to(device)
        self.torch_MCE = BinaryCalibrationError(n_bins=15, norm="max").to(device)

        # Globals
        self.dice_global = []
        self.pixel_acc_global = []
        self.IOU_score_global = []
        self.loss_global = []
        self.NLL_global = []
        self.ECE_global = []
        self.MCE_global = []
        self.brier_global = []
        self.soft_dice_global = []

        # Locals
        self.dice_vec_local = []
        self.loss_local = []
        self.pixel_acc_local = []
        self.iou_local = []
        self.dice_own_local = []
        self.dice_own_confuse_local = []
        self.soft_dice_local = []
        self.IOU_own_local = []
        self.NLL_local = []
        self.ECE_local = []
        self.MCE_local = []
        self.brier_local = []

    def GetMetrics(self, predictions, masks, loss):
        # Dice Score
        self.dice_vec_local.append(
            self.dice(predictions.squeeze(1), masks.type(torch.float32)).item()
        )
        self.dice_own_local.append(
            self.metrics.Dice_Coef(predictions.squeeze(1), masks.type(torch.float32)).item()
        )
        self.dice_own_confuse_local.append(
            self.metrics.Dice_Coef_Confusion(
                predictions.squeeze(1), masks.type(torch.float32)
            ).item()
        )
        # Soft dice
        self.soft_dice_local.append(self.metrics.torch_their_dice(predictions, masks).item())
        # IOU
        self.IOU_own_local.append(
            self.metrics.IOU_(predictions.squeeze(1), masks.type(torch.float32)).item()
        )
        self.iou_local.append(self.IOU_score(predictions.squeeze(1), masks).item())
        # Pixel accuracy
        self.pixel_acc_local.append(self.pixel_acc(predictions.squeeze(1), masks).item())
        # Loss
        self.loss_local.append(loss.item())
        # Calib Metrics
        calib_metrics = self.calibration_metrics.metrics(predictions, masks)
        self.NLL_local += calib_metrics["NLL"]
        try:
            self.ECE_local.append(
                self.torch_ECE(predictions.squeeze(1), masks).item()
            )  # calib_metrics['ECE']
            self.MCE_local.append(
                self.torch_MCE(predictions.squeeze(1), masks).item()
            )  # calib_metrics['MCE']
        except:
            self.ECE_local.append(0)
            self.MCE_local.append(0)
        self.brier_local += calib_metrics["brier_score"]

        return (self.loss_local[-1], self.dice_vec_local[-1])

    def AppendToGlobal(self):
        self.dice_global.append(torch.tensor(self.dice_vec_local).mean().item())
        self.pixel_acc_global.append(torch.tensor(self.pixel_acc_local).mean().item())
        self.IOU_score_global.append(torch.tensor(self.iou_local).mean().item())
        self.loss_global.append(torch.tensor(self.loss_local).mean().item())
        self.NLL_global.append(torch.tensor(self.NLL_local).mean().item())
        self.ECE_global.append(torch.tensor(self.ECE_local).mean().item())
        self.MCE_global.append(torch.tensor(self.MCE_local).mean().item())
        self.brier_global.append(torch.tensor(self.brier_local).mean().item())
        self.brier_local.append(torch.tensor(self.soft_dice_local).mean())
        # Now reset local arrays
        self.dice_vec_local = []
        self.loss_local = []
        self.pixel_acc_local = []
        self.iou_local = []
        self.dice_own_local = []
        self.dice_own_confuse_local = []
        self.soft_dice_local = []
        self.IOU_own_local = []
        self.NLL_local = []
        self.ECE_local = []
        self.MCE_local = []
        self.brier_local = []
