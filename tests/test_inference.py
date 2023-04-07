import torch
from src.models.inference import inference
from src.data.dataloader import train_split
from src.ActiveLearning.AL_utils import binarize, unbinarize
import pandas as pd
from src.Metrics.CollectMetrics import CollectMetrics
from src.visualization.viz_tools import viz_batch
import matplotlib.pyplot as plt
import time
from src.ActiveLearning.AcquisitionFunctions import ActiveLearningAcquisitions
import pytest


@pytest.mark.skipif(1 == 1, reason="Too Computationally Expensive - Run Local !!!")
def test_inference():
    """
    Check if metric is consistent within function and after loading from checkpoint
    """
    models = ["models/MAP/warwick_0.61_261_17.pth"]

    MetricsCalulator = CollectMetrics(validation=True, device="cpu", out_ch=1)
    df = pd.read_json("results/test_frame.json")
    model_params = {
        "in_ch": 3,
        "n_classes": 1,
        "bilinear_method": False,
        "momentum": 0.9,
        "enable_dropout": False,
        "dropout_prob": 0.5,
        "enable_pool_dropout": True,
        "pool_dropout_prob": 0.5,
    }
    _, val_loader, _, _, _, _ = train_split(
        0.61, dataset="warwick", batch_size=4, to_binary=True, num_workers=0, seed=261
    )
    images, masks, predictions = inference(
        models=models,
        model_params=model_params,
        data_loader=val_loader,
        method="Normal",
        seed=261,
        torch_seeds=[17],
        dataset="warwick",
        device="cpu",
    )
    # Get Metrics
    NLL, Brier, ECE, MCE, Dice, IOU, Acc, Soft_Dice = MetricsCalulator.Metrics(
        predictions.squeeze(1), masks
    )
    assert round(df["val_dice_all"][0][-1], 5) == round(Dice, 5)
    assert round(df["val_pixel_accuracy_all"][0][-1], 5) == round(Acc, 5)
    assert round(df["val_MCE_all"][0][-1], 5) == round(MCE, 5)
    assert round(df["val_NLL_all"][0][-1], 5) == round(NLL, 5)
    assert round(df["val_soft_dice_all"][0][-1], 5) == round(Soft_Dice, 5)

    # Example On Plotting
    MetricsCalulator.plots(predictions, masks, show=True, dataset="warwick")
    fig = viz_batch(
        images[0:4],
        masks[0:4],
        predictions[0:4],
        cols=["img", "mask", "pred", "err"],
        from_logits=True,
        reduction=False,
        save_=False,
        dataset="warwick",
        save_path=None,
    )
    plt.show()


@pytest.mark.skipif(1 == 1, reason="Too Computationally Expensive - Run Local !!!")
def test_inference_MCD():
    start = time.time()
    models = ["models/MAP/warwick_0.61_261_17.pth"]
    MetricsCalulator = CollectMetrics(validation=True, device="cpu", out_ch=1)
    df = pd.read_json("results/test_frame.json")
    model_params = {
        "in_ch": 3,
        "n_classes": 1,
        "bilinear_method": False,
        "momentum": 0.9,
        "enable_dropout": False,
        "dropout_prob": 0.5,
        "enable_pool_dropout": True,
        "pool_dropout_prob": 0.5,
    }
    _, val_loader, _, _, val_idx, _ = train_split(
        0.61, dataset="warwick", batch_size=4, to_binary=True, num_workers=0, seed=261
    )
    images, masks, predictions, prediction_idx = inference(
        models=models,
        model_params=model_params,
        data_loader=val_loader,
        method="MCD",
        seed=261,
        torch_seeds=[17],
        dataset="warwick",
        device="cpu",
    )
    assert torch.equa(val_idx, prediction_idx)
    mean_predictions = torch.mean(torch.sigmoid(predictions), dim=1)
    mean_predictions = mean_predictions.permute(0, 3, 1, 2)

    MetricsCalulator.plots(mean_predictions, masks, show=True, dataset="warwick", from_logits=False)
    fig = viz_batch(
        images[0:4],
        masks[0:4],
        predictions[0:4],
        cols=["img", "mask", "pred", "err"],
        from_logits=True,
        reduction=True,
        save_=False,
        dataset="warwick",
        save_path=None,
    )
    plt.show()

    Acq_fucns = ActiveLearningAcquisitions()
    BALD = Acq_fucns.ApplyAcquisition(binarize(torch.sigmoid(predictions)), method="BALD")
    JSD = Acq_fucns.ApplyAcquisition(
        binarize(torch.sigmoid(predictions)), method="JensenDivergence"
    )
    Entropy = Acq_fucns.ApplyAcquisition(
        binarize(torch.sigmoid(predictions)), method="ShanonEntropy"
    )

    top_values, top_indicies = torch.topk(BALD, k=4)
    fig = viz_batch(
        images[top_indicies],
        masks[top_indicies],
        predictions[top_indicies],
        cols=["var", "entropy", "mut_info", "jsd"],
        from_logits=True,
        reduction=True,
        save_=False,
        dataset="warwick",
        save_path=None,
    )
    plt.show()

    top_values, top_indicies = torch.topk(JSD, k=4)
    fig = viz_batch(
        images[top_indicies],
        masks[top_indicies],
        predictions[top_indicies],
        cols=["var", "entropy", "mut_info", "jsd"],
        from_logits=True,
        reduction=True,
        save_=False,
        dataset="warwick",
        save_path=None,
    )
    plt.show()

    top_values, top_indicies = torch.topk(Entropy, k=4)
    fig = viz_batch(
        images[top_indicies],
        masks[top_indicies],
        predictions[top_indicies],
        cols=["var", "entropy", "mut_info", "jsd"],
        from_logits=True,
        reduction=True,
        save_=False,
        dataset="warwick",
        save_path=None,
    )
    plt.show()

    end = time.time()
    print(f"Execution Time: {end-start}")
    # apply sihmoid
    # pass to metric
    #
    # Get Metrics
    # NLL, Brier, ECE, MCE, Dice, IOU, Acc, Soft_Dice=MetricsCalulator.Metrics(predictions.squeeze(1),masks)
    # assert df['val_dice_all'][0][-1] == Dice


# ## Plots test
# plots

# # Realibility Diagram test
# unbinarize(binarize(torch.sigmoid(predictions)))

# # Acquision Function Function
# binarize(predictions)

# if __name__ == "__main__":
#     #test_inference()
#     test_inference_MCD()
