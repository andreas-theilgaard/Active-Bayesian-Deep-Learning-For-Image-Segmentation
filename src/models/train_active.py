# Import libraries
import math
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
import hydra

# Import modules
from src.models.model import UNET, init_weights

# from src.models.new_model import UNET, init_weights
from src.visualization.plot import plot_prediction_batch
from src.config import Config
from src.data.dataloader import train_split, data_from_index
import pandas as pd
from src.experiments.experiment_utils import arrayify_results
import numpy as np

# Initialize wandb
import wandb
import time
from src.models.active_learning_utils import ActiveLearningAcquisitions
from src.experiments.experiment_utils import arrayify_results


def init_params(
    momentum=0.9,
    in_ch=1,
    out_ch=1,
    bilinear_method=False,
    enable_dropout=False,
    dropout_prob=0.5,
    enable_pool_dropout=False,
    pool_dropout_prob=0.5,
):
    print(
        f""" U-Net Model Parameters: \n ---------------------- \n momentum: {momentum} \n in_ch: {in_ch} \n out_ch: {out_ch}\n bilinear_method: {bilinear_method}\n enable_dropout: {enable_dropout} \n dropout_prob: {dropout_prob}\n enable_pool_dropout: {enable_pool_dropout}\n pool_dropout_prob: {pool_dropout_prob} \n ----------------------"""
    )


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


def store_results(MetricsTrain, MetricsValidate, query_id, execution_time, save_path):
    data_to_store = {
        "train_loss": [MetricsTrain.loss_global],
        "train_dice": [MetricsTrain.dice_global],
        "train_pixel_accuracy": [MetricsTrain.pixel_acc_global],
        "train_IOU": [MetricsTrain.IOU_score_global],
        "train_NLL": [MetricsTrain.NLL_global],
        "train_ECE": [MetricsTrain.ECE_global],
        "train_MCE": [MetricsTrain.MCE_global],
        "train_brier": [MetricsTrain.brier_global],
        "train_soft_dice": [MetricsTrain.soft_dice_global],
        "val_loss": [MetricsValidate.loss_global],
        "val_dice": [MetricsValidate.dice_global],
        "val_pixel_accuracy": [MetricsValidate.pixel_acc_global],
        "val_IOU": [MetricsValidate.IOU_score_global],
        "val_NLL": [MetricsValidate.NLL_global],
        "val_ECE": [MetricsValidate.ECE_global],
        "val_MCE": [MetricsValidate.MCE_global],
        "val_brier": [MetricsValidate.brier_global],
        "val_soft_dice": [MetricsValidate.soft_dice_global],
        "Query ID": (query_id),
        "Execution Time": execution_time,  # Measured in seconds
    }
    arrayify_results(data_to_store, save_path=save_path)


def active_train(
    batch_size=4,
    learning_rate=0.001,
    epochs=100,
    momentum=0.9,
    beta0=0.5,
    train_size=0.99,
    dataset="PhC-C2DH-U373",
    device="cpu",
    validation_size=0.33,
    binary=True,
    enable_dropout=False,
    dropout_prob=0.5,
    enable_pool_dropout=False,
    pool_dropout_prob=0.5,
    bilinear_method=False,
    model_method="PoolDropout",
    seed=261,
    torch_seed=17,
    first_train=True,
    train_loader=None,
    val_loader=None,
    unlabeled_loader=None,
    train_idx=None,
    val_idx=None,
    unlabeled_pool_idx=None,
    query_id=None,
    AcquisitionFunction=None,
):

    torch.manual_seed(torch_seed)

    # Load training and validation data
    if first_train:
        (
            train_loader,
            val_loader,
            unlabeled_loader,
            train_idx,
            val_idx,
            unlabeled_pool_idx,
        ) = train_split(
            train_size=train_size,
            dataset=dataset,
            batch_size=batch_size,
            to_binary=binary,
            num_workers=0,
            seed=seed,
        )
    else:
        (
            train_loader,
            val_loader,
            unlabeled_loader,
            train_idx,
            val_idx,
            unlabeled_pool_idx,
        ) = data_from_index(
            dataset,
            batch_size,
            binary,
            train_idx,
            val_idx,
            unlabeled_pool_idx,
            num_workers=0,
            seed=seed,
        )

    n_classes = 1 if binary else Config.n_classes[dataset]  # out channels to use for U-Net
    DEVICE = device  # Device to use
    in_ch = 1 if dataset != "warwick" else 3

    start = time.time()
    # Get model and intilize weights
    model = UNET(
        in_ch=in_ch,
        out_ch=n_classes,
        bilinear_method=bilinear_method,
        momentum=momentum,
        enable_dropout=enable_dropout,
        dropout_prob=dropout_prob,
        enable_pool_dropout=enable_pool_dropout,
        pool_dropout_prob=pool_dropout_prob,
    )
    model.apply(init_weights)
    model.to(device)
    if first_train:
        init_params(
            momentum=momentum,
            in_ch=in_ch,
            out_ch=n_classes,
            bilinear_method=bilinear_method,
            enable_dropout=enable_dropout,
            dropout_prob=dropout_prob,
            enable_pool_dropout=enable_pool_dropout,
            pool_dropout_prob=pool_dropout_prob,
        )

    # Intilize criterion and optimizer
    loss_fn = (
        nn.BCEWithLogitsLoss().to(DEVICE) if model.out_ch == 1 else nn.CrossEntropyLoss().to(DEVICE)
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta0, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.1, threshold=0.001
    )

    MetricsTrain = CollectMetrics(device=device, out_ch=model.out_ch)
    MetricsValidate = CollectMetrics(device=device, out_ch=model.out_ch)
    # Now train loop
    for epoch in range(epochs):
        train_loop = tqdm(train_loader, leave=False)  # Progress bar for the training data
        for batch_number, batch in enumerate(train_loop):
            images, masks, idx = batch
            images = images.unsqueeze(1) if dataset != "warwick" else images
            images = images.to(device=DEVICE, dtype=torch.float32)
            masks = masks.type(torch.LongTensor)
            if model.out_ch > 1:
                masks = masks.squeeze(1)
            masks = masks.to(DEVICE)
            # get predictions
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions.squeeze(1), masks.float())  # Calculate loss

            # Collect metrics
            out_loss, out_Dice = MetricsTrain.GetMetrics(predictions, masks, loss)
            loss.backward()
            optimizer.step()
            train_loop.set_postfix({"loss": out_loss, "train_dice": out_Dice})
            MetricsTrain.AppendToGlobal()

        if epoch > -1:  # epoch%10==0:
            val_loop = tqdm(val_loader, leave=False)
            with torch.no_grad():
                model.eval()
                for batch_number, batch in enumerate(val_loop):
                    images, masks, idx = batch
                    images = images.unsqueeze(1) if dataset != "warwick" else images
                    images = images.to(device=DEVICE, dtype=torch.float32)
                    masks = masks.type(torch.LongTensor)
                    if model.out_ch > 1:
                        masks = masks.squeeze(1)
                    masks = masks.to(DEVICE)
                    # get predictions
                    predictions = model(images)
                    loss = loss_fn(predictions.squeeze(1), masks.float())  # Calculate loss

                    # Collect metrics
                    out_loss, out_Dice = MetricsValidate.GetMetrics(predictions, masks, loss)
                    val_loop.set_postfix({"loss": out_loss, "val_dice": out_Dice})
                    MetricsValidate.AppendToGlobal()

        model.train()
    end = time.time()
    execution_time = end - start
    store_results(
        MetricsTrain,
        MetricsValidate,
        query_id,
        execution_time,
        f"results/active_learning/train_val_{dataset}_{model_method}_{AcquisitionFunction}_{seed}_{torch_seed}",
    )
    torch.save(model.state_dict(), f"models/{dataset}_{model_method}_{seed}_{torch_seed}.pth")
    # train and validate
    return (train_loader, val_loader, unlabeled_loader, train_idx, val_idx, unlabeled_pool_idx)


def process(input, first_run, file_path):
    file_exists = os.path.isfile(f"results/active_learning/test.json")
    if first_run:
        if not file_exists:
            df = pd.DataFrame(
                {"Train Round": input["epoch"], "Mu": input["mu"], "idx": input["idx"]}
            )
            df.to_json("results/active_learning/test.json")
        else:
            df = pd.read_json("results/active_learning/test.json")
            df_new = pd.DataFrame(
                {"Train Round": input["epoch"], "Mu": input["mu"], "idx": input["idx"]}
            )
            df_out = pd.concat([df, df_new], ignore_index=True)
            df_out.to_json("results/active_learning/test.json")
    else:
        df = pd.read_json(f"results/active_learning/test.json")
        df_new = pd.DataFrame(
            {"Train Round": input["epoch"], "Mu": input["mu"], "idx": input["idx"]}
        )
        df_out = pd.concat([df, df_new], ignore_index=True)
        df_out.to_json("results/active_learning/test.json")


def enable_MCDropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def binarize(predictions):
    if len(predictions.shape) == 4:
        return torch.cat([predictions, 1 - predictions], dim=3)
    elif len(predictions.shape) == 5:
        return torch.cat([predictions, 1 - predictions], dim=4)


def unbinarize(predictions):
    return predictions[:, :, :, :, 0].unsqueeze(4)


def find_next(
    models: list,
    model_method=None,
    first_train=True,
    train_loader=None,
    val_loader=None,
    unlabeled_loader=None,
    train_idx=None,
    val_idx=None,
    unlabeled_pool_idx=None,
    momentum=None,
    enable_dropout=False,
    dropout_prob=0.5,
    enable_pool_dropout=False,
    pool_dropout_prob=0.5,
    bilinear_method=False,
    dataset=None,
    binary=None,
    device=None,
    AcquisitionFunction=None,
    torch_seed=None,
):

    np.random.seed(torch_seed)
    Acq_Func = ActiveLearningAcquisitions()

    if AcquisitionFunction == "Random":
        next_labels = Acq_Func.ApplyAcquisition(unlabeled_pool_idx, method=AcquisitionFunction, n=2)
        return next_labels

    n_classes = 1 if binary else Config.n_classes[dataset]  # out channels to use for U-Net
    DEVICE = device  # Device to use
    in_ch = 1 if dataset != "warwick" else 3
    loss_fn = (
        nn.BCEWithLogitsLoss().to(DEVICE) if n_classes == 1 else nn.CrossEntropyLoss().to(DEVICE)
    )

    # Get model and intilize weights
    model = UNET(
        in_ch=in_ch,
        out_ch=n_classes,
        bilinear_method=bilinear_method,
        momentum=momentum,
        enable_dropout=enable_dropout,
        dropout_prob=dropout_prob,
        enable_pool_dropout=enable_pool_dropout,
        pool_dropout_prob=pool_dropout_prob,
    )
    # Create shapes here based on sizes above
    for model_path in models:
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        Unlabeled_loop = tqdm(unlabeled_loader)  # unlabelled pool
        if model_method == "MCD":
            with torch.no_grad():
                model.eval()
                model.apply(enable_MCDropout)
                for batch_number, batch in enumerate(Unlabeled_loop):
                    images, masks, idx = batch
                    images = images.unsqueeze(1) if dataset != "warwick" else images
                    images = images.to(device=DEVICE, dtype=torch.float32)
                    # get predictions
                    predictions = torch.stack(
                        [
                            binarize(torch.sigmoid(model(images).permute(0, 2, 3, 1)))
                            for x in range(50)
                        ],
                        dim=1,
                    )
                    if batch_number == 0:
                        prediction_tensor = predictions
                        prediction_idx = idx.clone().detach()
                    else:
                        prediction_tensor = torch.vstack((prediction_tensor, predictions))
                        prediction_idx = torch.cat([prediction_idx, idx.clone().detach()])

    scores = Acq_Func.ApplyAcquisition(prediction_tensor, method=AcquisitionFunction)
    if len(unlabeled_pool_idx) < 2:
        top_values, top_indicies = torch.topk(scores, k=1)
    else:
        top_values, top_indicies = torch.topk(scores, k=2)
    next_labels = prediction_idx[top_indicies]
    return next_labels.tolist()


def how_many_iters(dataset):
    total_images = [x for x in os.listdir(f"data/raw/{dataset}/image") if x != ".DS_Store"]
    validation_images = math.ceil(len(total_images) * 0.33)
    training_unlabeled_images = len(total_images) - validation_images
    train_size = training_unlabeled_images * 0.01
    if (training_unlabeled_images * 0.01) < 1.0:
        train_size = 1
    unlabeled_pool_size = training_unlabeled_images - train_size
    out = math.ceil(unlabeled_pool_size / 2)
    return out


def run_active(
    batch_size=4,
    learning_rate=0.001,
    epochs=3,
    momentum=0.9,
    beta0=0.9,
    train_size=0.01,
    dataset="PhC-C2DH-U373",
    device="cpu",
    validation_size=0.33,
    binary=True,
    enable_dropout=False,
    dropout_prob=0.5,
    enable_pool_dropout=True,
    pool_dropout_prob=0.5,
    bilinear_method=False,
    model_method="MCD",
    seed=261,
    torch_seeds=[17],
    first_train=True,
    AcquisitionFunction="BALD",  # [Random,ShanonEntropy,BALD]
    train_loader=None,
    val_loader=None,
    unlabeled_loader=None,
    train_idx=None,
    val_idx=None,
    unlabeled_pool_idx=None,
):
    print(f"Running on device {device}")
    models_list = [
        f"models/{dataset}_{model_method}_{seed}_{torch_seed}.pth" for torch_seed in torch_seeds
    ]
    # loop_length = how_many_iters(dataset)

    active_df = pd.DataFrame(
        {"Query_id": [], "labels_added": [], "Train_size": [], "Unlabeled_size": []}, dtype=object
    )
    for torch_seed in torch_seeds:
        for i in tqdm(range(3)):
            (
                train_loader,
                val_loader,
                unlabeled_loader,
                train_idx,
                val_idx,
                unlabeled_pool_idx,
            ) = active_train(
                batch_size=batch_size,
                learning_rate=learning_rate,
                epochs=epochs,
                momentum=momentum,
                beta0=beta0,
                train_size=train_size,
                dataset=dataset,
                device=device,
                validation_size=validation_size,
                binary=binary,
                enable_dropout=enable_dropout,
                dropout_prob=dropout_prob,
                enable_pool_dropout=enable_pool_dropout,
                pool_dropout_prob=pool_dropout_prob,
                bilinear_method=bilinear_method,
                model_method=model_method,
                seed=seed,
                torch_seed=torch_seed,
                first_train=first_train,
                train_loader=train_loader,
                val_loader=val_loader,
                unlabeled_loader=unlabeled_loader,
                train_idx=train_idx,
                val_idx=val_idx,
                unlabeled_pool_idx=unlabeled_pool_idx,
                query_id=i,
                AcquisitionFunction=AcquisitionFunction,
            )

            next_labels = find_next(
                models=models_list,
                model_method=model_method,
                first_train=first_train,
                train_loader=train_loader,
                val_loader=val_loader,
                unlabeled_loader=unlabeled_loader,
                train_idx=train_idx,
                val_idx=val_idx,
                unlabeled_pool_idx=unlabeled_pool_idx,
                momentum=momentum,
                enable_dropout=enable_dropout,
                dropout_prob=dropout_prob,
                enable_pool_dropout=enable_pool_dropout,
                pool_dropout_prob=pool_dropout_prob,
                bilinear_method=bilinear_method,
                dataset=dataset,
                binary=binary,
                device=device,
                AcquisitionFunction=AcquisitionFunction,
                torch_seed=torch_seed,
            )
            # Add labels to train pool
            train_idx += next_labels
            # Remove labels from unlabeled pool
            [unlabeled_pool_idx.remove(x) for x in next_labels]
            first_train = False
            print(f"Adding {','.join(map(str,next_labels))} to the training pool.")
            print(
                f"Size of training set {len(train_idx)}, Size of unlabeled pool {len(unlabeled_pool_idx)}"
            )
            active_df.loc[len(active_df)] = [
                i + 1,
                np.array(next_labels, dtype=object),
                len(train_idx),
                len(unlabeled_pool_idx),
            ]
            print(active_df)
        # Final Train on the last data
        (
            train_loader,
            val_loader,
            unlabeled_loader,
            train_idx,
            val_idx,
            unlabeled_pool_idx,
        ) = active_train(
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            momentum=momentum,
            beta0=beta0,
            train_size=train_size,
            dataset=dataset,
            device=device,
            validation_size=validation_size,
            binary=binary,
            enable_dropout=enable_dropout,
            dropout_prob=dropout_prob,
            enable_pool_dropout=enable_pool_dropout,
            pool_dropout_prob=pool_dropout_prob,
            bilinear_method=bilinear_method,
            model_method=model_method,
            seed=seed,
            torch_seed=torch_seed,
            first_train=first_train,
            train_loader=train_loader,
            val_loader=val_loader,
            unlabeled_loader=unlabeled_loader,
            train_idx=train_idx,
            val_idx=val_idx,
            unlabeled_pool_idx=unlabeled_pool_idx,
            query_id=i + 1,
            AcquisitionFunction=AcquisitionFunction,
        )
        active_df.to_json(
            f"results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{torch_seed}.json"
        )


# if __name__ == "__main__":
#     run_active()
