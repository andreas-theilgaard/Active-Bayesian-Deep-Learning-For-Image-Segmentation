import torch
from src.Metrics.CollectMetrics import CollectMetrics
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import argparse
import os

# Import modules
from src.models.model import UNET, init_weights

# from src.models.new_model import UNET, init_weights
from src.visualization.plot import plot_prediction_batch
from src.config import Config
from src.data.dataloader import train_split

# Initialize wandb
import wandb
import time
import hydra

# Wand b here
@hydra.main(config_path="src/configs", config_name="base.yaml", version_base="1.1")
def active_train(cfg):
    print(cfg.hyperparameters.batch_size)

    # torch.manual_seed(torch_seed)

    # n_classes = 1 if binary else Config.n_classes[dataset]  # out channels to use for U-Net
    # DEVICE = device  # Device to use
    # in_ch = 1 if dataset != "warwick" else 3

    # start = time.time()
    # # Get model and intilize weights
    # model = UNET(
    #     in_ch=in_ch,
    #     out_ch=n_classes,
    #     bilinear_method=bilinear_method,
    #     momentum=momentum,
    #     enable_dropout=enable_dropout,
    #     dropout_prob=dropout_prob,
    #     enable_pool_dropout=enable_pool_dropout,
    #     pool_dropout_prob=pool_dropout_prob,
    # )
    # model.apply(init_weights)
    # model.to(device)
    # if first_train:
    #     init_params(
    #         momentum=momentum,
    #         in_ch=in_ch,
    #         out_ch=n_classes,
    #         bilinear_method=bilinear_method,
    #         enable_dropout=enable_dropout,
    #         dropout_prob=dropout_prob,
    #         enable_pool_dropout=enable_pool_dropout,
    #         pool_dropout_prob=pool_dropout_prob,
    #     )

    # # Intilize criterion and optimizer
    # loss_fn = (
    #     nn.BCEWithLogitsLoss().to(DEVICE) if model.out_ch == 1 else nn.CrossEntropyLoss().to(DEVICE)
    # )
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta0, 0.999))
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, "min", patience=5, factor=0.1, threshold=0.001
    # )

    # MetricsTrain = CollectMetrics(device=device, out_ch=model.out_ch)
    # MetricsValidate = CollectMetrics(device=device, out_ch=model.out_ch)
    # # Now train loop
    # for epoch in range(epochs):
    #     train_loop = tqdm(train_loader)  # Progress bar for the training data
    #     for batch_number, batch in enumerate(train_loop):
    #         images, masks, idx = batch
    #         images = images.unsqueeze(1) if dataset != "warwick" else images
    #         images = images.to(device=DEVICE, dtype=torch.float32)
    #         masks = masks.type(torch.LongTensor)
    #         if model.out_ch > 1:
    #             masks = masks.squeeze(1)
    #         masks = masks.to(DEVICE)
    #         # get predictions
    #         optimizer.zero_grad()
    #         predictions = model(images)
    #         loss = loss_fn(predictions.squeeze(1), masks.float())  # Calculate loss

    #         # Collect metrics
    #         out_loss, out_Dice = MetricsTrain.GetMetrics(predictions, masks, loss)
    #         loss.backward()
    #         optimizer.step()
    #         train_loop.set_postfix({"loss": out_loss, "train_dice": out_Dice})
    #         MetricsTrain.AppendToGlobal()

    #     if epoch > -1:  # epoch%10==0:
    #         val_loop = tqdm(val_loader)
    #         with torch.no_grad():
    #             model.eval()
    #             for batch_number, batch in enumerate(val_loop):
    #                 images, masks, idx = batch
    #                 images = images.unsqueeze(1) if dataset != "warwick" else images
    #                 images = images.to(device=DEVICE, dtype=torch.float32)
    #                 masks = masks.type(torch.LongTensor)
    #                 if model.out_ch > 1:
    #                     masks = masks.squeeze(1)
    #                 masks = masks.to(DEVICE)
    #                 # get predictions
    #                 predictions = model(images)
    #                 loss = loss_fn(predictions.squeeze(1), masks.float())  # Calculate loss

    #                 # Collect metrics
    #                 out_loss, out_Dice = MetricsValidate.GetMetrics(predictions, masks, loss)
    #                 val_loop.set_postfix({"loss": out_loss, "val_dice": out_Dice})
    #                 MetricsValidate.AppendToGlobal()

    #     model.train()
    # end = time.time()
    # execution_time = end - start
    # store_results(
    #     MetricsTrain,
    #     MetricsValidate,
    #     query_id,
    #     execution_time,
    #     f"results/active_learning/train_val___{dataset}_{model_method}_{AcquisitionFunction}_{seed}_{torch_seed}",
    # )
    # torch.save(model.state_dict(), f"models/{dataset}_{model_method}_{seed}_{torch_seed}.pth")
    # # train and validate
    # return (train_loader, val_loader, unlabeled_loader, train_idx, val_idx, unlabeled_pool_idx)


if __name__ == "__main__":
    active_train()
