# Import libraries
import torch
from tqdm import tqdm
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
from src.models.model_utils import SegmentationMetrics, Calibration_Scoring_Metrics
from torchmetrics.classification import BinaryCalibrationError

# Import modules
from src.models.model import UNET, init_weights

# from src.models.new_model import UNET, init_weights
from src.visualization.plot import plot_prediction_batch
from src.config import Config
from src.data.dataloader import train_split
import pandas as pd

# Initialize wandb
import wandb
import time


def train(
    dataset="PhC-C2DH-U373",
    train_size=0.99,
    epochs=5,
    lr=0.001,
    momentum=0.9,
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
    bilinear_method=False,
    model_method=None,
    seed=261,
    beta0=0.9,
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
    parser.add_argument("--beta0", default=beta0)  #
    args = parser.parse_args()
    print(args)
    torch.manual_seed(
        17
    )  # seed used for weight initialization, this should be stochastic when using deep ensemble
    # init weights & biases
    if experiment_name and job_type:
        run_tracker = wandb.init(
            entity="andr_dl_projects",
            reinit=False,
            project="Active Bayesian Deep Learning For Image Segmentation",
            resume="allow",
            group=experiment_name,
            job_type=job_type,
            anonymous="allow",
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
        seed=seed,
    )

    in_ch = 1 if args.dataset != "warwick" else 3
    print(in_ch)

    start = time.time()
    # Get model and intilize weights
    model = UNET(
        in_ch=in_ch,
        out_ch=n_classes,
        bilinear_method=bilinear_method,
        momentum=args.momentum,
        enable_dropout=enable_dropout,
        dropout_prob=dropout_prob,
        enable_pool_dropout=enable_pool_dropout,
        pool_dropout_prob=pool_dropout_prob,
    )
    model.apply(init_weights)
    model.to(args.device)

    # Intilize criterion and optimizer
    loss_fn = (
        nn.BCEWithLogitsLoss().to(DEVICE) if model.out_ch == 1 else nn.CrossEntropyLoss().to(DEVICE)
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(args.beta0, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.1, threshold=0.001
    )

    # Define metrics that will be used
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
    calibration_metrics = Calibration_Scoring_Metrics(
        nbins=15, multiclass=False, device=args.device
    )
    torch_ECE = BinaryCalibrationError(n_bins=15, norm="l1").to(args.device)
    torch_MCE = BinaryCalibrationError(n_bins=15, norm="max").to(args.device)

    # define arrays for global storage
    train_dice_global = []
    train_pixel_acc_global = []
    train_IOU_score_global = []
    train_loss_global = []
    train_NLL = []
    train_ECE = []
    train_MCE = []
    train_brier = []
    train_soft_dice = []

    val_dice_global = []
    val_pixel_acc_global = []
    val_IOU_score_global = []
    val_loss_global = []
    val_NLL = []
    val_ECE = []
    val_MCE = []
    val_brier = []
    validation_soft_dice = []

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
        train_NLL_local = []
        train_ECE_local = []
        train_MCE_local = []
        train_brier_local = []

        train_loop = tqdm(train_loader)  # Progress bar for the training data
        print("Model In Train", model.training)
        for batch_number, batch in enumerate(train_loop):
            images, masks = batch
            images = images.unsqueeze(1) if args.dataset != "warwick" else images
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

            train_soft_dice.append(metrics.torch_their_dice(predictions, masks).item())
            train_IOU_own.append(
                metrics.IOU_(predictions.squeeze(1), masks.type(torch.float32)).item()
            )

            loss = loss_fn(predictions.squeeze(1), masks.float())  # Calculate loss
            pixel_acc_metric.append(pixel_acc(predictions.squeeze(1), masks).item())
            iou_metric.append(IOU_score(predictions.squeeze(1), masks).item())
            train_loss.append(loss.item())
            calib_metrics = calibration_metrics.metrics(predictions, masks)
            train_NLL_local += calib_metrics["NLL"]
            train_ECE_local.append(
                torch_ECE(predictions.squeeze(1), masks).item()
            )  # calib_metrics['ECE']
            train_MCE_local.append(
                torch_MCE(predictions.squeeze(1), masks).item()
            )  # calib_metrics['MCE']
            train_brier_local += calib_metrics["brier_score"]

            # backpropagation
            loss.backward()
            optimizer.step()

            # For the first batch in each epoch, some image predictions are logged to wandb
            if batch_number == 0:
                if (epoch + 1) == args.epochs:
                    save_ = True
                fig = plot_prediction_batch(
                    images, masks, predictions, save_=save_, dataset=args.dataset
                )
                wandb.log({"Training Predictions": fig})
                if save_:
                    wandb.log({"Final High Resolution Training Example": wandb.Image("pred.png")})
                    os.remove("pred.png")

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
        wandb.log({"Train ECE": torch.tensor(train_ECE_local).mean()})
        wandb.log({"Train MCE": torch.tensor(train_MCE_local).mean()})
        wandb.log({"Train Brier": torch.tensor(train_brier_local).mean()})

        # Store to global for each epoch
        train_dice_global.append(torch.tensor(dice_vec).mean().item())
        train_pixel_acc_global.append(torch.tensor(pixel_acc_metric).mean().item())
        train_IOU_score_global.append(torch.tensor(iou_metric).mean().item())
        train_loss_global.append(torch.tensor(train_loss).mean().item())
        train_NLL.append(torch.tensor(train_NLL_local).mean().item())
        train_ECE.append(torch.tensor(train_ECE_local).mean().item())
        train_MCE.append(torch.tensor(train_MCE_local).mean().item())
        train_brier.append(torch.tensor(train_brier_local).mean().item())
        train_soft_dice.append(torch.tensor(train_soft_dice).mean())

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
        val_NLL_local = []
        val_ECE_local = []
        val_MCE_local = []
        val_brier_local = []

        # define progress bar for validation lopp
        val_loop = tqdm(val_loader)

        with torch.no_grad():
            model.eval()
            print("Model In Train", model.training)
            for batch_number, batch in enumerate(val_loop):
                images, masks = batch
                images = images.unsqueeze(1) if args.dataset != "warwick" else images
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
                val_soft_dice.append(metrics.torch_their_dice(predictions, masks).item())
                val_IOU_own.append(
                    metrics.IOU_(predictions.squeeze(1), masks.type(torch.float32)).item()
                )

                loss = loss_fn(predictions.squeeze(1), masks.float())  # Calculate loss
                val_accuracy.append(pixel_acc(predictions.squeeze(1), masks).item())
                val_iou_metric.append(IOU_score(predictions.squeeze(1), masks).item())
                val_loss.append(loss.item())

                calib_metrics = calibration_metrics.metrics(predictions, masks)
                val_NLL_local += calib_metrics["NLL"]
                val_ECE_local.append(
                    torch_ECE(predictions.squeeze(1), masks).item()
                )  # calib_metrics['ECE']
                val_MCE_local.append(
                    torch_MCE(predictions.squeeze(1), masks).item()
                )  # calib_metrics['MCE']
                val_brier_local += calib_metrics["brier_score"]

                val_loop.set_postfix(**{"loss (batch)": loss.item(), "val_dice": val_dice_vec[-1]})
                baseline.append(
                    torch.max(masks.sum() / masks.numel(), 1 - masks.sum() / masks.numel()).item()
                )

                if batch_number == 0:
                    fig = plot_prediction_batch(images, masks, predictions, save_=save_)
                    wandb.log({"Validation Predictions": fig})
                    if save_:
                        wandb.log(
                            {"Final High Resolution Validation Example": wandb.Image("pred.png")}
                        )
                        os.remove("pred.png")

        val_soft_dice = torch.tensor(val_soft_dice).mean().item()
        val_dice_vec_own = torch.tensor(val_dice_vec_own).mean().item()
        val_dice_vec_own_confuse = torch.tensor(val_dice_vec_own_confuse).mean().item()
        baseline = torch.tensor(baseline).mean().item()
        val_dice = torch.tensor(val_dice_vec).mean().item()

        val_iou = torch.tensor(val_iou_metric).mean().item()
        val_loss = torch.tensor(val_loss).mean().item()
        val_accuracy = torch.tensor(val_accuracy).mean().item()
        val_IOU_own = torch.tensor(val_IOU_own).mean().item()

        val_NLL_local = torch.tensor(val_NLL_local).mean().item()
        val_ECE_local = torch.tensor(val_ECE_local).mean().item()
        val_MCE_local = torch.tensor(val_MCE_local).mean().item()
        val_brier_local = torch.tensor(val_brier_local).mean().item()

        wandb.log({"Validation soft dice (epoch):": val_soft_dice})
        wandb.log({"Validation acc baseline (epoch):": baseline})
        wandb.log({"Validation loss (epoch):": val_loss})
        wandb.log({"Validation dice (epoch):": val_dice})
        wandb.log({"Validation pixel accuracy (epoch):": val_accuracy})
        wandb.log({"Validation IOU (epoch):": val_iou})
        wandb.log({"Validation dice own (epoch):": val_dice_vec_own})
        wandb.log({"Validation dice own confuse (epoch):": val_dice_vec_own_confuse})
        wandb.log({"Validation IOU own (epoch):": val_IOU_own})
        wandb.log({"Validation ECE": val_ECE_local})
        wandb.log({"Validation MCE": val_MCE_local})
        wandb.log({"Validation Brier": val_brier_local})

        # Store to global for each epoch
        val_dice_global.append(val_dice)
        val_pixel_acc_global.append(val_accuracy)
        val_IOU_score_global.append(val_iou)
        val_loss_global.append(val_loss)
        val_NLL.append(val_NLL_local)
        val_ECE.append(val_ECE_local)
        val_MCE.append(val_MCE_local)
        val_brier.append(val_brier_local)
        validation_soft_dice.append(val_soft_dice)
        scheduler.step(val_loss)

        # Put model in train mode
        model.train()

    # store the metrics in one array
    # Train: loss | dice | pixel | iou | NLL | ECE | MCE | Brier | Soft Dice
    # Val: loss | dice | pixel | iou   | NLL | ECE | MCE | Brier | Soft Dice
    # Other: datasize (train size) | method (batchmorm baseline, dropout, dropout pool, both dropout and pool dropout)  | Experiment number | Execution Time
    # Total Columns: 9+9+4=22
    end = time.time()
    execution_time = end - start
    data_to_store = {
        "train_loss": [train_loss_global],
        "train_dice": [train_dice_global],
        "train_pixel_accuracy": [train_pixel_acc_global],
        "train_IOU": [train_IOU_score_global],
        "train_NLL": [train_NLL],
        "train_ECE": [train_ECE],
        "train_MCE": [train_MCE],
        "train_brier": [train_brier],
        "train_soft_dice": [train_soft_dice],
        "val_loss": [val_loss_global],
        "val_dice": [val_dice_global],
        "val_pixel_accuracy": [val_pixel_acc_global],
        "val_IOU": [val_IOU_score_global],
        "val_NLL": [val_NLL],
        "val_ECE": [val_ECE],
        "val_MCE": [val_MCE],
        "val_brier": [val_brier],
        "val_soft_dice": [validation_soft_dice],
        "train_size": args.train_size,
        "method": args.model_method,
        "Experiment Number": (args.iter + 1),
        "Execution Time": execution_time,  # Measured in seconds
    }
    wandb.finish()
    return data_to_store


if __name__ == "__main__":
    train()
