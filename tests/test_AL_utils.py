import torch
from src.ActiveLearning.AL_utils import binarize, unbinarize
from src.config import find_best_device


def test_binarize():
    Type_ = "UncertainPredictions"
    if Type_ == "UncertainPredictions":
        pseudo_predictions = torch.sigmoid(torch.rand((4, 50, 64, 64, 1)))  # P(Y=1|x,theta)
        binarized_predictions = binarize(
            pseudo_predictions
        )  # [P(Y=0|x,theta),P(Y=1|x,theta)], [First layer, Second Layer]

        assert torch.equal(
            binarized_predictions.sum(-1), torch.ones((binarized_predictions.shape[:-1]))
        )
        assert list(binarized_predictions.shape) == [4, 50, 64, 64, 2]
        assert torch.equal(binarized_predictions[:, :, :, :, 1].unsqueeze(-1), pseudo_predictions)
        assert torch.equal(
            pseudo_predictions, unbinarize(binarized_predictions)
        )  # Unbinarize Returns P(Y=1|x,theta) again
        Type_ = "NormalPredictions"

    if Type_ == "NormalPredictions":
        # Test Normal Predictions
        pseudo_predictions = torch.sigmoid(torch.rand((4, 1, 64, 64)))  # P(Y=1|x,theta)
        binarized_predictions = binarize(
            pseudo_predictions
        )  # [P(Y=0|x,theta),P(Y=1|x,theta)], [First layer, Second Layer]

        assert torch.equal(binarized_predictions.sum(1), torch.ones((4, 64, 64)))
        assert list(binarized_predictions.shape) == [4, 2, 64, 64]
        assert torch.equal(binarized_predictions[:, 1, :, :].unsqueeze(1), pseudo_predictions)
        assert torch.equal(
            pseudo_predictions, unbinarize(binarized_predictions)
        )  # Unbinarize Returns P(Y=1|x,theta) again
        Type_ = "Device"

    if Type_ == "Device":
        device = find_best_device()
        if device.type != "cpu":
            pseudo_predictions = torch.sigmoid(torch.rand((4, 1, 64, 64)))  # P(Y=1|x,theta)
            pseudo_predictions = pseudo_predictions.to(device)
            binarized_predictions = binarize(
                pseudo_predictions
            )  # [P(Y=0|x,theta),P(Y=1|x,theta)], [First layer, Second Layer]

            assert (str(binarized_predictions.device) == device) or (
                str(binarized_predictions.device)
                == f"{device}:{binarized_predictions.get_device()}"
            )
            assert torch.equal(binarized_predictions.sum(1), torch.ones((4, 64, 64)).to(device))
            assert list(binarized_predictions.shape) == [4, 2, 64, 64]
            assert torch.equal(binarized_predictions[:, 1, :, :].unsqueeze(1), pseudo_predictions)
            assert torch.equal(
                pseudo_predictions, unbinarize(binarized_predictions)
            )  # Unbinarize Returns P(Y=1|x,theta) again
