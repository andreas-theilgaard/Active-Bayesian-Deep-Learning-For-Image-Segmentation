import torch
from src.Metrics.CollectMetrics import CollectMetrics, store_results, WandBLogger
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from src.models.model import UNET, init_weights
from src.config import Config
from src.data.dataloader import train_split
import wandb
import time
import hydra
from src.models.util_models import init_params
from omegaconf import OmegaConf
import random
import numpy as np
from src.config import Config

# Wand b here
def train(
    dataset="warwick",
    train_size=0.61,
    epochs=10,
    device="cpu",
    binary=True,
    iter=10,
    experiment_name=None,
    job_type=None,
    enable_dropout=False,
    enable_pool_dropout=False,
    model_method=None,
    seed=261,
    turn_off_wandb=False,
    torch_seed=17,
    save_model=None,  # Model Path To Where Model Should Be Saved
):

    cfg = OmegaConf.load("src/configs/base.yaml")  # Load configurations from yaml file
    print(init_params(locals(), cfg))  # Print Configurations

    # Set Seeds
    random.seed(torch_seed)
    np.random.seed(seed)
    torch.manual_seed(torch_seed)  # Set Torch Seed
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    n_classes = 1 if binary else Config.n_classes[dataset]  # out channels to use for U-Net
    in_ch = (
        1 if dataset != "warwick" else 3
    )  # Only warwick is RGB color coded : TODO adjust such that agnostic

    start = time.time()  # initialize time to track how long training takes

    # Get model and initialize weights
    model = UNET(
        in_ch=in_ch,
        out_ch=n_classes,
        bilinear_method=cfg.bilinear_method,
        momentum=cfg.momentum,
        enable_dropout=enable_dropout,
        dropout_prob=cfg.dropout_prob,
        enable_pool_dropout=enable_pool_dropout,
        pool_dropout_prob=cfg.pool_dropout_prob,
    )
    model.apply(init_weights)
    model.to(device)

    # Initialize criterion and optimizer
    loss_fn = (
        nn.BCEWithLogitsLoss().to(device) if model.out_ch == 1 else nn.CrossEntropyLoss().to(device)
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta0, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=10, factor=0.1, threshold=0.001
    )
    MetricsTrain = CollectMetrics(validation=False, device=device, out_ch=model.out_ch)
    MetricsValidate = CollectMetrics(validation=True, device=device, out_ch=model.out_ch)

    # Load Data
    train_loader, val_loader, _, _, _, _ = train_split(
        train_size=train_size,
        dataset=dataset,
        batch_size=cfg.batch_size,
        to_binary=binary,
        num_workers=0,
        seed=seed,
    )

    # Initialize Weights & Biases
    if turn_off_wandb == False:
        WB_Logger = WandBLogger(job_type=job_type, experiment_name=experiment_name)
        WB_Logger.init_wandb()
    save_ = False
    batches = [0]  # Random batches to save prediction images from

    # Training Loop
    for epoch in range(epochs):
        train_loop = tqdm(train_loader)  # Progress bar for the training data
        for batch_number, batch in enumerate(train_loop):
            images, masks, _ = batch
            images = images.unsqueeze(1) if dataset != "warwick" else images
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.type(torch.LongTensor)
            if model.out_ch > 1:
                masks = masks.squeeze(1)
            masks = masks.to(device)

            # Get predictions
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions.squeeze(1), masks.float())  # Calculate loss

            # Collect metrics
            out_loss, out_Dice = MetricsTrain.GetMetrics(predictions, masks, loss.item())
            loss.backward()
            optimizer.step()
            train_loop.set_postfix({"loss": out_loss, "train_dice": out_Dice})

            # Log
            if batch_number in batches and turn_off_wandb == False:
                WB_Logger.log_images(
                    type_="Training",
                    save_=save_,
                    images=images,
                    masks=masks,
                    predictions=predictions,
                    dataset=dataset,
                    save_path=f"Assets/{dataset}/{train_size}_{seed}_{torch_seed}_{batch_number}_{epoch}",
                )
        MetricsTrain.AppendToGlobal()

        if (epoch + 1) == epochs:
            save_ = True
            batches = [0, 5, 15]

        predictions_vec = []  # Array To Collect All Validation Predictions
        masks_vec = []  # Array To Collect All Validation Masks
        val_loop = tqdm(val_loader)  # Progress Bar For Validation Data

        with torch.no_grad():
            model.eval()  # Put Model In Validation Mode
            for batch_number, batch in enumerate(val_loop):
                images, masks, _ = batch
                images = images.unsqueeze(1) if dataset != "warwick" else images
                images = images.to(device=device, dtype=torch.float32)
                masks = masks.type(torch.LongTensor)
                if model.out_ch > 1:
                    masks = masks.squeeze(1)
                masks = masks.to(device)

                # Get Predictions
                predictions = model(images)
                loss = loss_fn(predictions.squeeze(1), masks.float())  # Calculate loss
                out_loss, out_Dice = MetricsValidate.GetMetrics(predictions, masks, loss.item())
                val_loop.set_postfix({"loss": out_loss, "val_dice": out_Dice})
                predictions_vec.append(predictions)
                masks_vec.append(masks)

                # Log
                if batch_number in batches and turn_off_wandb == False:
                    WB_Logger.log_images(
                        type_="Validation",
                        save_=save_,
                        images=images,
                        masks=masks,
                        predictions=predictions,
                        dataset=dataset,
                        save_path=f"Assets/{dataset}/{train_size}_{seed}_{torch_seed}_{batch_number}_{epoch}",
                    )

        # Turn Arrays In Tensors
        predictions_vec = torch.vstack(predictions_vec)
        masks_vec = torch.vstack(masks_vec)
        # Log Metrics To Global
        MetricsValidate.AppendToGlobal({"Logits": predictions_vec.squeeze(1), "Masks": masks_vec})
        # Scheduler step
        scheduler.step(MetricsValidate.loss_global[-1])

        # Put Model Back In Train Mode
        model.train()

    MetricsValidate.plots(
        predictions_vec,
        masks_vec,
        title=f"Reliability Diagram - {Config.title_mapper[dataset]}",
        save_path=f"Assets/{dataset}/{train_size}_{seed}_{torch_seed}_{batch_number}_{epoch+1}_Diagram",
        dataset=dataset,
    )

    if save_model:
        torch.save(
            model.state_dict(),
            f"models/MAP/{dataset}_{train_size}_{seed}_{torch_seed}_{enable_pool_dropout}.pth",
        )

    end = time.time()  # Get end time
    execution_time = end - start  # Calculate execution time
    # Store metrics collected for each epoch to one big array
    data_to_store = store_results(
        MetricsTrain, MetricsValidate, execution_time=execution_time, ActiveResults=False
    )
    data_to_store["method"] = model_method
    data_to_store["Experiment Number"] = iter + 1
    data_to_store["train_size"] = train_size
    return data_to_store


if __name__ == "__main__":
    # from src.experiments.experiment_utils import arrayify_results
    res = train(
        dataset="warwick",
        train_size=0.61,
        epochs=20,
        turn_off_wandb=True,
        save_model=False,
        enable_pool_dropout=False,
    )
    # arrayify_results(res,'results/test_frame')
