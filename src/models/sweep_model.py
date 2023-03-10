# Import libraries
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import JaccardIndex, Dice
from torchmetrics.classification import (
    MulticlassAccuracy,
    BinaryAccuracy,
    BinaryJaccardIndex,
    BinaryF1Score,
)
import argparse
import os
from src.models.model_utils import SegmentationMetrics
import hydra
from src.data.initialize import get_data

# from sklearn.model_selection import train_test_split
# import numpy as np

# Import modules
# from src.models.unet_model import UNET
# from src.models.model import UNET, init_weights
from src.models.new_model import UNET, init_weights, store_init
from src.visualization.plot import plot_prediction_batch
from src.config import Config
from src.data.dataloader import train_split
import pandas as pd

# Initialize wandb
import wandb


def torch_their_dice(pred, mask):
    mask = mask.unsqueeze(1)
    pred = torch.sigmoid(pred)
    intersection = torch.sum(pred * mask, dim=(1, 2, 3))
    union = torch.sum(mask, dim=(1, 2, 3)) + torch.sum(pred, dim=(1, 2, 3))
    smooth = 1e-12
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return torch.mean(dice)


# @hydra.main(config_path="../configs",config_name="base.yaml",version_base="1.1")
def train(
    dataset="PhC-C2DH-U373",
    train_size=0.99,
    device="cpu",
    validation_size=0.33,
    binary=True,
    iter=10,
    experiment_name=None,
    job_type=None,
    enable_dropout=False,
    dropout_prob=0.5,
    enable_pool_dropout=False,
    pool_dropout_prob=0.5,
    bilinear_method=False,
    model_method=None,
    seed=261,
):

    run = wandb.init()
    config = wandb.config
    # print(config.parameters.batch_size)
    batch_size = 4
    lr = 0.001
    epochs = 50
    # momentum=config.parameters.momentum
    momentum = config["momentum"]
    beta0 = config["beta0"]
    interpolate_image = config["interpolate_image"]
    interpolate_mask = config["interpolate_mask"]
    bias = config["bias"]
    init_ = config["init_"]
    store_init.init_ = init_

    # print(momentum)
    # beta0 = config.parameters.beta0

    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--model_method", default=model_method)
    parser.add_argument("--iter", default=iter)
    parser.add_argument("--lr", default=lr)  #
    parser.add_argument("--device", default=device)  #
    parser.add_argument("--batch_size", default=batch_size)  #
    parser.add_argument("--epochs", default=epochs)  #
    parser.add_argument("--momentum", default=momentum)  #
    parser.add_argument("--train_size", default=train_size)  #
    parser.add_argument("--validation_size", default=validation_size)  #
    parser.add_argument("--dataset", default=dataset)  #
    parser.add_argument("--binary", default=binary)  #
    args = parser.parse_args()
    print(args)
    print(f"bilinear_method:", bilinear_method)
    torch.manual_seed(
        17
    )  # seed used for weight initialization, this should be stochastic when using deep ensemble

    n_classes = (
        1 if args.binary else Config.n_classes[args.dataset]
    )  # out channels to use for U-Net
    DEVICE = args.device  # Device to use
    save_ = False  # boolean, set to True when last epoch reached for saving image predictions

    # Load training and validation data
    train_loader, val_loader = train_split(
        train_size=args.train_size,
        dataset=args.dataset,
        batch_size=args.batch_size,
        to_binary=args.binary,
        num_workers=0,
        seed=seed,
        interpolate_image=interpolate_image,
        interpolate_mask=interpolate_mask,
    )

    # Get model and intilize weights
    # model = UNET(
    #     in_ch=1,
    #     out_ch=n_classes,
    #     bilinear_method=bilinear_method,
    #     momentum=args.momentum,
    #     enable_dropout=enable_dropout,
    #     dropout_prob=dropout_prob,
    #     enable_pool_dropout=enable_pool_dropout,
    #     pool_dropout_prob=pool_dropout_prob,
    # )

    # model = UNET(in_ch=1,out_ch=1)

    model = UNET(in_ch=1, out_ch=n_classes, momentum=0.9, bias=bias)
    model.apply(init_weights)
    model.to(args.device)

    # Intilize criterion and optimizer
    loss_fn = (
        nn.BCEWithLogitsLoss().to(DEVICE) if model.out_ch == 1 else nn.CrossEntropyLoss().to(DEVICE)
    )
    optimizer = optim.Adam(
        model.parameters(), lr=lr, betas=(beta0, 0.999)
    )  # eps=1e-07,weight_decay=1e-7
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.1, threshold=0.01
    )
    # goal: maximize Dice score € change to track val loss

    # Define metricsthat will be used
    dice = (
        BinaryF1Score().to(args.device)
        if model.out_ch == 1
        else Dice(average="micro", ignore_index=0).to(DEVICE)
    )
    pixel_acc = (
        MulticlassAccuracy(num_classes=model.out_ch, validate_args=True).to(args.device)
        if model.out_ch > 1
        else BinaryAccuracy(validate_args=True).to(args.device)
    )
    IOU_score = BinaryJaccardIndex().to(args.device)
    metrics = SegmentationMetrics()

    # define arrays for global storage
    train_dice_global = []
    train_pixel_acc_global = []
    train_IOU_score_global = []
    train_loss_global = []

    val_dice_global = []
    val_pixel_acc_global = []
    val_IOU_score_global = []
    val_loss_global = []

    # Training loop
    for epoch in range(args.epochs):
        print(epoch)
        # model.train()  # Put model in train mode

        # Define arrays to store metrics
        dice_vec = []
        train_loss = []
        pixel_acc_metric = []
        iou_metric = []
        dice_vec_own = []
        dice_vec_own_confuse = []
        train_soft_dice = []
        train_IOU_own = []

        train_loop = tqdm(train_loader)  # Progress bar for the training data
        print("Model In Train", model.training)
        for batch_number, batch in enumerate(train_loop):
            images, masks = batch
            images = images.unsqueeze(1)
            images = images.to(device=DEVICE, dtype=torch.float32)
            masks = masks.type(torch.LongTensor)
            if model.out_ch > 1:
                masks = masks.squeeze(1)
            masks = masks.to(DEVICE)

            # get predictions
            optimizer.zero_grad()
            predictions = model(images)

            # Save dice score
            dice_vec.append(dice(predictions.squeeze(1), masks.type(torch.float32)).item())
            dice_vec_own.append(
                metrics.Dice_Coef(predictions.squeeze(1), masks.type(torch.float32)).item()
            )
            dice_vec_own_confuse.append(
                metrics.Dice_Coef_Confusion(
                    predictions.squeeze(1), masks.type(torch.float32)
                ).item()
            )
            train_soft_dice.append(torch_their_dice(predictions, masks).item())
            train_IOU_own.append(
                metrics.IOU_(predictions.squeeze(1), masks.type(torch.float32)).item()
            )

            loss = loss_fn(predictions.squeeze(1), masks.float())  # Calculate loss
            pixel_acc_metric.append(pixel_acc(predictions.squeeze(1), masks).item())
            iou_metric.append(IOU_score(predictions.squeeze(1), masks).item())
            train_loss.append(loss.item())

            # backpropagation
            loss.backward()
            optimizer.step()

            # For the first batch in each epoch, some image predictions are logged to wandb
            # if batch_number == 0:
            #     if (epoch + 1) == args.epochs:
            #         save_ = True
            #     fig = plot_prediction_batch(images, masks, predictions, save_=save_)
            #     wandb.log({"Training Predictions": fig})
            #     if save_:
            #         wandb.log({"Final High Resolution Training Example": wandb.Image("pred.png")})
            #         os.remove("pred.png")

            # update tqdm loop
            train_loop.set_postfix({"loss": loss.item(), "train_dice": dice_vec[-1]})

        # Log metrics to wandb
        # TO DO: Other Metrics, ECE ??
        wandb.log({"Train loss (epoch):": torch.tensor(train_loss).mean()})
        wandb.log({"Train dice (epoch):": torch.tensor(dice_vec).mean()})
        wandb.log({"Train pixel accuracy (epoch):": torch.tensor(pixel_acc_metric).mean()})
        wandb.log({"Train IOU (epoch):": torch.tensor(iou_metric).mean()})
        wandb.log({"Train dice own:": torch.tensor(dice_vec_own).mean()})
        wandb.log({"Train dice confuse:": torch.tensor(dice_vec_own_confuse).mean()})
        wandb.log({"Train soft dice": torch.tensor(train_soft_dice).mean()})
        wandb.log({"Train IOU own": torch.tensor(train_IOU_own).mean()})

        # Store to global for each epoch
        train_dice_global.append(torch.tensor(dice_vec).mean().item())
        train_pixel_acc_global.append(torch.tensor(pixel_acc_metric).mean().item())
        train_IOU_score_global.append(torch.tensor(iou_metric).mean().item())
        train_loss_global.append(torch.tensor(train_loss).mean().item())

        # Create arrays to store metrics
        val_dice_vec = []  #
        val_dice_vec_own = []  #
        val_dice_vec_own_confuse = []  #
        val_loss = []  #
        val_iou_metric = []  #
        val_accuracy = []  #
        val_soft_dice = []  #
        baseline = []
        val_IOU_own = []

        # define progress bar for validation lopp
        val_loop = tqdm(val_loader)

        with torch.no_grad():
            model.eval()
            print("Model In Train", model.training)
            for batch_number, batch in enumerate(val_loop):
                images, masks = batch
                images = images.unsqueeze(1)
                images = images.to(device=DEVICE, dtype=torch.float32)
                masks = masks.type(torch.LongTensor)
                if model.out_ch > 1:
                    masks = masks.squeeze(1)
                masks = masks.to(device)

                predictions = model(images)  # .to(device)

                # Save dice score
                val_dice_vec.append(dice(predictions.squeeze(1), masks.type(torch.float32)).item())
                val_dice_vec_own.append(
                    metrics.Dice_Coef(predictions.squeeze(1), masks.type(torch.float32)).item()
                )
                val_dice_vec_own_confuse.append(
                    metrics.Dice_Coef_Confusion(
                        predictions.squeeze(1), masks.type(torch.float32)
                    ).item()
                )
                val_soft_dice.append(torch_their_dice(predictions, masks).item())
                val_IOU_own.append(
                    metrics.IOU_(predictions.squeeze(1), masks.type(torch.float32)).item()
                )

                loss = loss_fn(predictions.squeeze(1), masks.float())  # Calculate loss
                val_accuracy.append(pixel_acc(predictions.squeeze(1), masks).item())
                val_iou_metric.append(IOU_score(predictions.squeeze(1), masks).item())
                val_loss.append(loss.item())

                val_loop.set_postfix(**{"loss (batch)": loss.item(), "val_dice": val_dice_vec[-1]})
                baseline.append(
                    torch.max(masks.sum() / masks.numel(), 1 - masks.sum() / masks.numel()).item()
                )

                # if batch_number == 0:
                #     fig = plot_prediction_batch(images, masks, predictions, save_=save_)
                #     wandb.log({"Validation Predictions": fig})
                #     if save_:
                #         wandb.log(
                #             {"Final High Resolution Validation Example": wandb.Image("pred.png")}
                #         )
                #         os.remove("pred.png")

        val_soft_dice = torch.tensor(val_soft_dice).mean().item()
        val_dice_vec_own = torch.tensor(val_dice_vec_own).mean().item()
        val_dice_vec_own_confuse = torch.tensor(val_dice_vec_own_confuse).mean().item()
        baseline = torch.tensor(baseline).mean().item()
        val_dice = torch.tensor(val_dice_vec).mean().item()

        val_iou = torch.tensor(val_iou_metric).mean().item()
        val_loss = torch.tensor(val_loss).mean().item()
        val_accuracy = torch.tensor(val_accuracy).mean().item()
        val_IOU_own = torch.tensor(val_IOU_own).mean().item()

        wandb.log({"Validation their dice (epoch):": val_soft_dice})
        wandb.log({"Validation acc baseline (epoch):": baseline})
        wandb.log({"Validation loss (epoch):": val_loss})
        wandb.log({"Validation dice (epoch):": val_dice})
        wandb.log({"Validation pixel accuracy (epoch):": val_accuracy})
        wandb.log({"Validation IOU (epoch):": val_iou})
        wandb.log({"Validation dice own (epoch):": val_dice_vec_own})
        wandb.log({"Validation dice own confuse (epoch):": val_dice_vec_own_confuse})
        wandb.log({"Validation IOU own (epoch):": val_IOU_own})

        print("IOU HERE:", val_iou, val_IOU_own)

        # Store to global for each epoch
        val_dice_global.append(val_dice)
        val_pixel_acc_global.append(val_accuracy)
        val_IOU_score_global.append(val_iou)
        val_loss_global.append(val_loss)
        scheduler.step(val_loss)  # Change to val_loss ???

        # Put model in train mode
        model.train()

    # store the metrics in one array
    # Train: loss | dice | pixel | iou
    # Val: loss | dice | pixel | iou
    # Other: datasize (train size) | method (droput, batchnorm) etc ??
    data_to_store = {
        "train_loss": [train_loss_global],
        "train_dice": [train_dice_global],
        "train_pixel_accuracy": [train_pixel_acc_global],
        "train_IOU": [train_IOU_score_global],
        "val_loss": [val_loss_global],
        "val_dice": [val_dice_global],
        "val_pixel_accuracy": [val_pixel_acc_global],
        "val_IOU": [val_IOU_score_global],
        "train_size": args.train_size,
        "method": args.model_method,
        "Experiment Number": (args.iter + 1),
    }
    return data_to_store


if __name__ == "__main__":
    get_data()
    sweep_configuration = {
        "method": "bayes",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "val_dice"},
        "parameters": {
            "momentum": {"min": 0.1, "max": 0.99},
            "beta0": {"min": 0.4, "max": 0.9},
            "interpolate_image": {"min": 0, "max": 2},
            "interpolate_mask": {"min": 0, "max": 1},
            "bias": {"min": 0, "max": 1},
            "init_": {"min": 0, "max": 3},
        },
    }
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        entity="andr_dl_projects",
        project="Active Bayesian Deep Learning For Image Segmentation",
    )
    wandb.agent(sweep_id=sweep_id, function=train, count=20)
