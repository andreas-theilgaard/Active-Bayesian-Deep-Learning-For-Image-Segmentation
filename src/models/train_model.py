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

# from sklearn.model_selection import train_test_split
# import numpy as np

# Import modules
from src.models.model import UNET, init_weights
from src.visualization.plot import plot_prediction_batch
from src.config import Config
from src.data.dataloader import train_split
import pandas as pd

# Initialize wandb
import wandb

# init wandb


@torch.inference_mode()
def validate(val_data, model, loss, device, save_):
    model.eval()  # Put model in eval mode

    # Define metrics to use
    dice = (
        BinaryF1Score().to(device)  # Dice(average="macro", validate_args=True).to(device) #micro
        if model.out_ch == 1
        else Dice(average="micro", ignore_index=0, validate_args=True).to(device)
    )
    pixel_acc = (
        MulticlassAccuracy(num_classes=model.out_ch, validate_args=True).to(device)
        if model.out_ch > 1
        else BinaryAccuracy(validate_args=True).to(device)
    )
    IOU_score = BinaryJaccardIndex().to(device)

    # Create arrays to store metrics
    dice_vec = []
    picel_acc_torch_vec = []
    val_loss = []
    iou_metric = []

    # define progress bar for validation lopp
    val_loop = tqdm(val_data, leave=False, desc="Validation round", unit="batch")

    with torch.no_grad():
        for batch_number, batch in enumerate(val_loop):
            images, masks = batch
            images = images.to(device, dtype=torch.float32)
            masks = masks.type(torch.LongTensor)
            if model.out_ch > 1:
                masks = masks.squeeze(1)
            masks = masks.to(device)

            preds = model(images)  # .to(device)
            # dice_vec.append(dice(preds, masks).item())  # Dice Score
            dice_vec.append(dice(preds.squeeze(1), masks).item())  # Dice Score

            if batch_number == 0:
                fig = plot_prediction_batch(images, masks, preds, save_=save_)
                wandb.log({"Validation Predictions": fig})
                if save_:
                    wandb.log({"Final High Resolution Validation Example": wandb.Image("pred.png")})
                    os.remove("pred.png")

            if model.out_ch == 1:
                masks = masks.float()
                loss_i = loss(preds.squeeze(1), masks)  # Get loss
                picel_acc_torch_vec.append(
                    pixel_acc(preds.squeeze(1), masks).item()
                )  # Pixel Accuracy torchmetrics
                iou_metric.append(IOU_score(preds.squeeze(1), masks).item())
            else:
                loss_i = loss(preds, masks)
                # pixel_acc_vec.append(get_pixel_accuracy(model.out_ch,preds,masks).item())
                picel_acc_torch_vec.append(pixel_acc(preds, masks).item())

            val_loss.append(loss_i.item())
            val_loop.set_postfix(**{"loss (batch)": loss_i.item(), "val_dice": dice_vec[-1]})

    val_dice = torch.tensor(dice_vec).mean().item()
    val_pix_acc = torch.tensor(picel_acc_torch_vec).mean().item()
    val_iou = torch.tensor(iou_metric).mean().item()
    val_loss = torch.tensor(val_loss).mean().item()

    wandb.log({"Validation loss (epoch):": val_loss})
    wandb.log({"Validation dice (epoch):": val_dice})
    wandb.log({"Validation pixel accuracy (epoch):": val_pix_acc})
    wandb.log({"Validation IOU (epoch):": val_iou})

    model.train()
    return (val_dice, val_pix_acc, val_iou, val_loss)


def train(
    dataset="membrane",
    train_size=0.99,
    epochs=10,
    lr=0.0001,
    momentum=0.1,
    device="cpu",
    batch_size=4,
    validation_size=0.33,
    binary=True,
    iter=10,
    experiment_name=None,
    job_type=None,
    enable_dropout=False,
    dropout_prob=0.5,
    enable_pool_dropout=False,
    pool_dropout_prob=0.5,
    bilinear_method=True,
    model_method=None,
):
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

    # init weights & biases
    if experiment_name and job_type:
        run_tracker = wandb.init(
            entity="andr_dl_projects",
            reinit=False,
            project="Active Bayesian Deep Learning For Image Segmentation",
            resume="allow",
            group=experiment_name,
            job_type=job_type,
        )
    else:
        run_tracker = wandb.init(
            entity="andr_dl_projects",
            reinit=False,
            project="Active Bayesian Deep Learning For Image Segmentation",
            resume="allow",
        )

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
    )

    # Get model and intilize weights
    model = UNET(
        in_ch=3,
        out_ch=n_classes,
        bilinear_method=bilinear_method,
        momentum=args.momentum,
        enable_dropout=enable_dropout,
        dropout_prob=dropout_prob,
        enable_pool_dropout=enable_pool_dropout,
        pool_dropout_prob=pool_dropout_prob,
    ).to(args.device)
    model = init_weights(model)

    # Intilize criterion and optimizer
    loss_fn = (
        nn.BCEWithLogitsLoss().to(DEVICE) if model.out_ch == 1 else nn.CrossEntropyLoss().to(DEVICE)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # eps=1e-07,weight_decay=1e-7
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=5, factor=0.1, threshold=0.001  # maybe reduce threshold
    )  # goal: maximize Dice score € change to track val loss

    # Define metricsthat will be used
    dice = (
        BinaryF1Score().to(args.device)  # Dice(average="macro").to(DEVICE) # micro
        if model.out_ch == 1
        else Dice(average="micro", ignore_index=0).to(DEVICE)
    )
    pixel_acc = (
        MulticlassAccuracy(num_classes=model.out_ch, validate_args=True).to(args.device)
        if model.out_ch > 1
        else BinaryAccuracy(validate_args=True).to(args.device)
    )
    IOU_score = BinaryJaccardIndex().to(args.device)

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
        model.train()  # Put model in train mode

        # Define arrays to store metrics
        dice_vec = []
        train_loss = []
        pixel_acc_metric = []
        iou_metric = []

        train_loop = tqdm(train_loader)  # Progress bar for the training data

        for batch_number, batch in enumerate(train_loop):
            images, masks = batch
            images = images.to(device=DEVICE, dtype=torch.float32)
            masks = masks.type(torch.LongTensor)
            if model.out_ch > 1:
                masks = masks.squeeze(1)
            masks = masks.to(DEVICE)

            # get predictions
            optimizer.zero_grad()
            predictions = model(images)

            # Save dice score
            dice_vec.append(
                dice(predictions.squeeze(1), masks).item()
            )  # squeeze(1) added for binarydice

            # For the first batch in each epoch, some image predictions are logged to wandb
            if batch_number == 0:
                if (epoch + 1) == args.epochs:
                    save_ = True
                fig = plot_prediction_batch(images, masks, predictions, save_=save_)
                wandb.log({"Training Predictions": fig})
                if save_:
                    wandb.log({"Final High Resolution Training Example": wandb.Image("pred.png")})
                    os.remove("pred.png")

            if model.out_ch == 1:
                loss = loss_fn(predictions.squeeze(1), masks.float())  # Calculate loss
                # log metrics
                pixel_acc_metric.append(pixel_acc(predictions.squeeze(1), masks).item())
                iou_metric.append(IOU_score(predictions.squeeze(1), masks).item())

            elif model.out_ch > 1:
                loss = loss_fn(predictions, masks)
                # loss += dice_loss(predictions,masks,model)
                # their_dice.append(torch_their_dice(predictions,masks,model).item())
            train_loss.append(loss.item())

            # backpropagation
            loss.backward()
            optimizer.step()

            # update tqdm loop
            train_loop.set_postfix({"loss": loss.item(), "train_dice": dice_vec[-1]})

        # Log metrics to wandb
        # TO DO: Other Metrics, ECE ??
        wandb.log({"Train loss (epoch):": torch.tensor(train_loss).mean()})
        wandb.log({"Train dice (epoch):": torch.tensor(dice_vec).mean()})
        wandb.log({"Train pixel accuracy (epoch):": torch.tensor(pixel_acc_metric).mean()})
        wandb.log({"Train IOU (epoch):": torch.tensor(iou_metric).mean()})

        # Store to global for each epoch
        train_dice_global.append(torch.tensor(dice_vec).mean().item())
        train_pixel_acc_global.append(torch.tensor(pixel_acc_metric).mean().item())
        train_IOU_score_global.append(torch.tensor(iou_metric).mean().item())
        train_loss_global.append(torch.tensor(train_loss).mean().item())

        # Validate model
        model.eval()
        val_dice, val_pix_acc, val_iou, val_loss = validate(
            val_loader, model, loss_fn, device=DEVICE, save_=save_
        )

        # Store to global for each epoch
        val_dice_global.append(val_dice)
        val_pixel_acc_global.append(val_pix_acc)
        val_IOU_score_global.append(val_iou)
        val_loss_global.append(val_loss)

        scheduler.step(val_dice)  # Change to val_loss ???

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
    wandb.finish()
    return data_to_store


# if __name__=="__main__":
#     train()
