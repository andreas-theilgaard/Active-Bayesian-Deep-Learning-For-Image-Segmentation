import torch
import pytest
from src.Metrics.CalibrationMetrics import Calibration_Scoring_Metrics
from src.config import find_best_device
import os


@pytest.mark.skipif(
    not os.path.exists("tests/test_assets/predictionsNormal.pth"), reason="Data files not found"
)
def test_calibration_metrics():
    images = torch.load("tests/test_assets/imagesNormal.pth")
    masks = torch.load("tests/test_assets/masksNormal.pth")
    predictions = torch.load("tests/test_assets/predictionsNormal.pth")

    # Test On CPU
    CalibMetrics = Calibration_Scoring_Metrics(
        nbins=10,
        multiclass=False,
        device="cpu",
        is_prob=False,
        reduction="mean",
        torchmetrics=False,
    )
    CalibMetricsTorch = Calibration_Scoring_Metrics(
        nbins=10,
        multiclass=False,
        device="cpu",
        is_prob=False,
        reduction="mean",
        torchmetrics=False,
    )

    NLL, Brier, ECE, MCE = CalibMetrics.Calculate_Calibration_Metrics(predictions.squeeze(1), masks)
    NLL_P, Brier_P, ECE_P, MCE_P = CalibMetrics.Calculate_Calibration_Metrics(
        torch.sigmoid(predictions.squeeze(1)), masks, from_logits=False
    )

    NLL_T, Brier_T, ECE_T, MCE_T = CalibMetricsTorch.Calculate_Calibration_Metrics(
        predictions.squeeze(1), masks
    )

    assert torch.equal(
        torch.tensor([NLL, Brier, ECE, MCE]),
        torch.tensor(
            [0.34668129682540894, 0.10216283053159714, 0.0510956967915845, 0.12541159082632425]
        ),
    )
    assert torch.equal(
        torch.tensor([NLL, Brier, ECE, MCE]), torch.tensor([NLL_T, Brier_T, ECE_T, MCE_T])
    )

    assert torch.equal(
        torch.tensor([NLL, Brier, ECE, MCE]), torch.tensor([NLL_P, Brier_P, ECE_P, MCE_P])
    )

    # Test On None CPU Device
    device = find_best_device()
    if device.type != "cpu":
        CalibMetrics = Calibration_Scoring_Metrics(
            nbins=10,
            multiclass=False,
            device=device,
            is_prob=False,
            reduction="mean",
            torchmetrics=False,
        )
        NLL, Brier, ECE, MCE = CalibMetrics.Calculate_Calibration_Metrics(
            predictions.squeeze(1), masks
        )
        assert torch.equal(
            torch.tensor([NLL, Brier, ECE, MCE]),
            torch.tensor(
                [0.34668129682540894, 0.10216283053159714, 0.0510956967915845, 0.12541159082632425]
            ),
        )


if __name__ == "__main__":
    test_calibration_metrics()
