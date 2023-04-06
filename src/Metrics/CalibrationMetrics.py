import torch
from numpy import ndarray as array
from torch.nn.functional import one_hot, softmax
import numpy as np
import netcal
import matplotlib.pyplot as plt
from src.visualization.RealiabilityDiagramUtils import ReliabilityDiagram
from netcal.metrics import ECE, MCE
from torchmetrics.classification import BinaryCalibrationError, MulticlassCalibrationError
import os


def logit(predictions):
    """
    input: predictions, probabilities
    Output: predictions, logits
    """
    return np.log(predictions / (1 - predictions))


class Calibration_Scoring_Metrics:
    def __init__(
        self,
        nbins,
        torchmetrics,
        multiclass,
        device,
        is_prob=False,
        numeric_stability=1e-8,
        reduction="mean",
    ):
        self.nbins = nbins
        self.torchmetrics = torchmetrics
        self.is_prob = is_prob
        self.multiclass = multiclass
        self.device = device
        self.reduction = reduction
        self.numeric_stability = numeric_stability

        self.torch_ECE = BinaryCalibrationError(n_bins=self.nbins, norm="l1")
        self.torch_MCE = BinaryCalibrationError(n_bins=self.nbins, norm="max")

        self.ECE_Metric = ECE(bins=self.nbins)
        self.MCE_Metric = MCE(bins=self.nbins)

    def NLL(self, y_hat, y_true):
        if y_hat.min() < 0 or y_hat.max() > 1:
            # Assuming Logits has been passed, so normalize
            y_hat = torch.sigmoid(y_hat)
        # Reduce Dimensions
        y_hat = y_hat.reshape(y_hat.shape[0], -1)
        y_true = y_true.reshape(y_true.shape[0], -1)

        assert y_hat.shape == y_true.shape
        loss_per_sample = -(
            y_true * torch.log(y_hat + self.numeric_stability)
            + (1 - y_true) * torch.log((1 - y_hat) + self.numeric_stability)
        )
        loss_per_sample_in_batch = torch.mean(loss_per_sample, dim=-1)
        if self.reduction == "mean":
            return torch.mean(loss_per_sample_in_batch).item()
        return loss_per_sample_in_batch

    def brier_score(self, y_hat, y_true, sample_wise=True):
        """
        y_hat: predictions
        """
        if isinstance(y_hat, array) and isinstance(y_true, array):
            y_hat, y_true = torch.tensor(y_hat).to(self.device), torch.tensor(y_true).to(
                self.device
            )
        if self.multiclass:
            n_classes = y_hat.shape[1]
            p = softmax(y_hat, dim=1)
            y_true = one_hot(y_true.to(dtype=torch.int64), n_classes).permute(0, 3, 1, 2)
            brier = torch.mean(
                torch.sum(((p - y_true) ** 2), dim=1).view(y_hat.shape[0], -1), dim=1, keepdim=True
            )
            if not sample_wise:
                brier = torch.mean(brier)
            return brier.detach().cpu().numpy().ravel()
        else:
            p = torch.sigmoid(y_hat)
            brier = torch.mean(((p - y_true) ** 2).reshape(y_hat.shape[0], -1), dim=1, keepdim=True)
            if self.reduction == "mean":
                brier = torch.mean(brier).item()
                return brier
            return brier.detach().cpu().numpy().ravel()

    def CalibrationErrors(self, y_hat, y_true):
        if self.torchmetrics:
            ECE = self.torch_ECE(y_hat, y_true).item()
            MCE = self.torch_MCE(y_hat, y_true).item()
        else:
            ECE = self.ECE_Metric.measure(
                torch.sigmoid(y_hat).detach().cpu().numpy().ravel(),
                y_true.detach().cpu().numpy().ravel(),
            )
            MCE = self.MCE_Metric.measure(
                torch.sigmoid(y_hat).detach().cpu().numpy().ravel(),
                y_true.detach().cpu().numpy().ravel(),
            )
        return (ECE, MCE)

    def PlotRealiabilityDiagram(
        self, y_hat, y_true, title=None, save_path=None, dataset=None, show=False
    ):
        ECE, _ = self.CalibrationErrors(y_hat, y_true)
        diagram = ReliabilityDiagram(bins=10)
        diagram.plot(
            torch.sigmoid(y_hat.squeeze(1)).detach().cpu().numpy().ravel(),
            y_true.detach().cpu().numpy().ravel(),
            ECE,
            title_suffix=title,
        )
        if save_path:
            plt.savefig(f"{save_path}.png", dpi=1200)
        if show:
            plt.show()

    def Calculate_Calibration_Metrics(self, y_hat, y_true):
        """
        y_hat: logits predictions, [batch_size,img_width,img_height]
        y_true: masks, [batch_size,img_width,img_height]
        """
        ECE, MCE = self.CalibrationErrors(y_hat, y_true)
        NLL = self.NLL(y_hat, y_true)
        Brier = self.brier_score(y_hat, y_true)
        return (NLL, Brier, ECE, MCE)
