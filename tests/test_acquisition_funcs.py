import torch
import numpy as np
from scipy.special import xlogy
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import pytest
import os

# from src.models.active_learning_utils import ActiveLearningAcquisitions
from src.ActiveLearning.AcquisitionFunctions import ActiveLearningAcquisitions

AL_Acquisitions = ActiveLearningAcquisitions()


def get_predictions(type_=1):
    if type_ == 1:
        predictions = np.array(
            [
                [
                    [0.8, 0.1, 0.1],
                    [0.3, 0.7, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.2, 0.2, 0.6],
                    [0.2, 0.7, 0.1],
                ],
                [
                    [0.0, 1.0, 0.0],
                    [0.4, 0.6, 0.0],
                    [0.2, 0.7, 0.1],
                    [0.3, 0.1, 0.6],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.0, 0.6],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.0, 0.9],
                    [0.0, 0.1, 0.9],
                ],
            ]
        )
        predictions = np.moveaxis(predictions, [0, 1, 2], [1, 0, 2])
    elif type_ == 2:
        predictions = torch.load("tests/test_assets/predictions.pth")
    return predictions


def test_accquisition_Entropy():
    # Entropy 2D Case
    predictions = get_predictions(type_=1)
    consensus = predictions.mean(1)
    Entropy = -xlogy(consensus, consensus).sum(-1)
    npEntropy = entropy(consensus, axis=-1)
    expected = np.array([0.88948441, 1.05213917, 1.02965301, 0.80181855, 0.80331498])
    assert np.sum(np.isclose(Entropy, npEntropy)) == len(npEntropy)
    assert np.sum(np.isclose(Entropy, expected)) == len(expected)

    # Entropy MultiDim Case
    if os.path.isfile("tests/test_assets/predictions.pth"):
        predictions = get_predictions(type_=2)
        Entropy = AL_Acquisitions.ApplyAcquisition(predictions, "ShanonEntropy")
        expected = torch.tensor(
            [
                692.8059,
                710.8376,
                753.0989,
                673.0214,
                721.8343,
                774.0728,
                699.3755,
                574.3660,
                747.3176,
                604.4395,
                829.6035,
            ]
        )
        assert torch.isclose(Entropy, expected).sum() == len(expected)
        Entropy_pixel, _, _ = AL_Acquisitions.Get_All_Pixel_Wise(predictions)
        assert abs(torch.sum((Entropy_pixel.sum(dim=(1, 2)) - Entropy)).item()) < 0.0001


def test_accquisition_BALD():
    # BALD 2D Case
    predictions = get_predictions(type_=1)
    consensus = predictions.mean(1)
    H1 = -xlogy(consensus, consensus).sum(-1)
    H2 = np.mean((-xlogy(predictions, predictions).sum(-1)), -1)
    BALD = H1 - H2
    expected = np.array([0.40920094, 0.39984329, 0.41916249, 0.07738547, 0.42768113])
    assert np.sum(np.isclose(BALD, expected)) == len(expected)

    # BALD MultiDim Case
    if os.path.isfile("tests/test_assets/predictions.pth"):
        predictions = get_predictions(type_=2)
        BALD = AL_Acquisitions.ApplyAcquisition(predictions, "BALD")
        expected = torch.tensor(
            [
                68.81616,
                78.83368,
                70.045715,
                53.030212,
                51.62805,
                81.35907,
                49.35785,
                42.786865,
                81.59186,
                50.5531,
                75.182495,
            ]
        )
        assert torch.isclose(BALD, expected).sum() == len(expected)
        _, BALD_pixel, _ = AL_Acquisitions.Get_All_Pixel_Wise(predictions)
        assert abs(torch.sum((BALD_pixel.sum(dim=(1, 2)) - BALD)).item()) < 0.0001


def test_accquisition_JSD():
    # JSD 2D Case
    predictions = get_predictions(type_=1)
    consenus_prob = predictions.mean(1)
    consenus_prob = np.repeat(consenus_prob, repeats=predictions.shape[1], axis=0).reshape(
        predictions.shape
    )
    M = 0.5 * (predictions + consenus_prob)
    KL_PM = entropy(predictions, M, axis=-1)
    KL_QM = entropy(consenus_prob, M, axis=-1)
    JSD_values = (0.5 * (KL_PM + KL_QM)).mean(-1)
    expected = np.array([0.1204363, 0.12203822, 0.11649324, 0.02226583, 0.12294496])
    JensenDistance = (jensenshannon(predictions, consenus_prob, axis=-1) ** 2).mean(axis=1)
    assert np.sum(np.isclose(JSD_values, JensenDistance)) == len(JensenDistance)
    assert np.sum(np.isclose(JSD_values, expected)) == len(expected)

    # JSD MultiDim Case
    if os.path.isfile("tests/test_assets/predictions.pth"):
        predictions = get_predictions(type_=2)
        JSD_values = AL_Acquisitions.ApplyAcquisition(predictions, "JensenDivergence")
        expected = torch.tensor(
            [
                17.43856,
                19.954088,
                17.860067,
                13.4221325,
                13.06631,
                20.83108,
                12.481562,
                10.732777,
                20.840754,
                12.81344,
                19.020733,
            ]
        )
        assert torch.isclose(JSD_values, expected).sum() == len(expected)
        _, _, jsd_pixel = AL_Acquisitions.Get_All_Pixel_Wise(predictions)
        assert abs(torch.sum((jsd_pixel.sum(dim=(1, 2)) - JSD_values)).item()) < 0.0001
