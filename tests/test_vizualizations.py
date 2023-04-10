import torch
import matplotlib.pyplot as plt
from src.visualization.viz_tools import viz_batch, viz_data
import os
import pytest


@pytest.mark.skipif(
    not os.path.exists("tests/test_assets/predictionsNormal.pth"), reason="Data files not found"
)
def test_viz_data():
    viz_data()
    assert 1 == 1


@pytest.mark.skipif(
    not os.path.exists("tests/test_assets/predictionsNormal.pth"), reason="Data files not found"
)
def test_viz_batch():
    predictions = torch.load("tests/test_assets/predictionsNormal.pth")
    images = torch.load("tests/test_assets/imagesNormal.pth")
    masks = torch.load("tests/test_assets/masksNormal.pth")
    fig = viz_batch(
        images[0:4],
        masks[0:4],
        predictions[0:4],
        cols=["img", "mask", "pred", "err"],
        from_logits=True,
        reduction=False,
        save_=False,
        dataset="warwick",
    )
    assert 1 == 1
    predictions = torch.load("tests/test_assets/predictions.pth")
    images = torch.load("tests/test_assets/images.pth")
    masks = torch.load("tests/test_assets/masks.pth")
    fig = viz_batch(
        images[0:3],
        masks[0:3],
        predictions[0:3],
        cols=["img", "mask", "pred", "err"],
        from_logits=False,
        reduction=True,
        save_=False,
        dataset="PhC-C2DH-U373",
    )
    assert 1 == 1
