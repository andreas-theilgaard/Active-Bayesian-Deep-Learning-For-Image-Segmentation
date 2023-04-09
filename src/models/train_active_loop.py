import torch
from src.ActiveLearning.AcquisitionFunctions import ActiveLearningAcquisitions
from src.ActiveLearning.AL_utils import binarize, unbinarize
from src.models.inference import inference
from src.models.model import UNET, init_weights
from src.Metrics.CollectMetrics import CollectMetrics, store_results, WandBLogger
from src.experiments.experiment_utils import arrayify_results
from src.models.util_models import init_params, EarlyStopping
from src.config import Config
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from src.data.dataloader import train_split, data_from_index
import time
from omegaconf import OmegaConf
import random
import pandas as pd
import os
import numpy as np
import math
from src.visualization.viz_tools import viz_batch

# Wand b here
def train(
    dataset="warwick",
    train_size=0.61,
    epochs=10,
    device="cpu",
    binary=True,
    enable_pool_dropout=False,
    seed=261,
    turn_off_wandb=False,
    torch_seed=17,
    train_loader=None,
    val_loader=None,
    first_train=True,
    unlabeled_loader=None,
    train_idx=None,
    val_idx=None,
    unlabeled_pool_idx=None,
    query_id=None,
    AcquisitionFunction=None,
    model_method="MCD",
    Earlystopping_=False,
    model_params=None,
):

    cfg = OmegaConf.load("src/configs/base.yaml")  # Load configurations from yaml file
    if first_train:
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
        in_ch=model_params["in_ch"],
        out_ch=model_params["n_classes"],
        bilinear_method=model_params["bilinear_method"],
        momentum=model_params["momentum"],
        enable_dropout=model_params["enable_dropout"],
        dropout_prob=model_params["dropout_prob"],
        enable_pool_dropout=enable_pool_dropout,
        pool_dropout_prob=model_params["pool_dropout_prob"],
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
    EarlyStopper = EarlyStopping(tolerance=20)

    MetricsTrain = CollectMetrics(validation=False, device=device, out_ch=model.out_ch)
    MetricsValidate = CollectMetrics(validation=True, device=device, out_ch=model.out_ch)

    # Load Data
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
            batch_size=cfg.batch_size,
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
            dataset=dataset,
            batch_size=cfg.batch_size,
            to_binary=binary,
            train_idx=train_idx,
            val_idx=val_idx,
            unlabeled_pool_idx=unlabeled_pool_idx,
            num_workers=0,
            seed=seed,
        )

    # Initialize Weights & Biases
    if turn_off_wandb == False:
        WB_Logger = WandBLogger(
            job_type=1,
            experiment_name=f"{dataset}_{model_method}_{seed}_{torch_seed}_{AcquisitionFunction}",
        )
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
                    save_path=f"Assets/{dataset}/{AcquisitionFunction}_{seed}_{torch_seed}_{batch_number}_{epoch}",
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
                        save_path=f"Assets/{dataset}/{AcquisitionFunction}_{seed}_{torch_seed}_{batch_number}_{epoch}",
                    )

        # Turn Arrays In Tensors
        predictions_vec = torch.vstack(predictions_vec)
        masks_vec = torch.vstack(masks_vec)
        # Log Metrics To Global
        MetricsValidate.AppendToGlobal({"Logits": predictions_vec.squeeze(1), "Masks": masks_vec})
        # Scheduler step
        scheduler.step(MetricsValidate.loss_global[-1])
        if Earlystopping_:
            EarlyStopper(MetricsValidate.loss_global[-1])
            if EarlyStopper.early_stop:
                print(f"Early Stopping At epoch {epoch}/{epochs}")
                break

        # Put Model Back In Train Mode
        model.train()

    torch.save(
        model.state_dict(),
        f"models/{dataset}_{model_method}_{AcquisitionFunction}_{seed}_{torch_seed}_{enable_pool_dropout}.pth",
    )

    end = time.time()  # Get end time
    execution_time = end - start  # Calculate execution time
    # Store metrics collected for each epoch to one big array
    data_to_store = store_results(
        MetricsTrain,
        MetricsValidate,
        execution_time=execution_time,
        ActiveResults=True,
        query_id=query_id,
    )
    data_to_store["method"] = model_method
    # Progress
    arrayify_results(
        data_to_store=data_to_store,
        save_path=f"results/active_learning/train_val_{dataset}_{model_method}_{AcquisitionFunction}_{seed}_{torch_seed}",
    )
    return train_loader, val_loader, unlabeled_loader, train_idx, val_idx, unlabeled_pool_idx


def find_next(
    models,
    model_method,
    unlabeled_loader,
    unlabeled_pool_idx,
    torch_seeds,
    seed,
    AcquisitionFunction,
    dataset,
    device,
    model_params,
    n_items_to_label=2,
):
    """
    models: list of model_paths to saved models
    model_method: 'MCD, DeepEnsemble,Laplace'
    unlabeled_loader: unlabeled_loader
    torch_seeds: list of torch_seeds
    seed: int for numpy
    AcquisitionFunction: [ShanonEntropy,BALD,Random,JensenDivergence]
    #enable_pool_dropout: bool
    dataset: name of dataset
    #binary: Boll
    device: [cpu,mps,cuda]
    model_params: dict of model_params
    """
    np.random.seed(seed)

    Acq_Func = ActiveLearningAcquisitions()
    if AcquisitionFunction == "Random":
        next_labels = Acq_Func.ApplyAcquisition(
            unlabeled_pool_idx, method=AcquisitionFunction, n=n_items_to_label
        )
        return next_labels

    _, _, predictions, prediction_idx = inference(
        models=models,
        model_params=model_params,
        data_loader=unlabeled_loader,
        method=model_method,
        dataset=dataset,
        device=device,
        seed=seed,
        torch_seeds=torch_seeds,
    )
    scores = Acq_Func.ApplyAcquisition(
        binarize(torch.sigmoid(predictions)), method=AcquisitionFunction
    )
    if len(unlabeled_pool_idx) < 2:
        top_values, top_indicies = torch.topk(scores, k=1)
    else:
        top_values, top_indicies = torch.topk(scores, k=n_items_to_label)
    next_labels = prediction_idx[top_indicies]
    return next_labels.tolist()


# Main Loop


def how_many_iters(dataset, start_size, binary, seed):
    _, _, _, train_idx, val_idx, unlabel_idx = train_split(
        dataset=dataset, train_size=start_size, batch_size=4, to_binary=binary, seed=seed
    )
    n_start = int(start_size.split("-")[0])
    N = len(train_idx) + len(unlabel_idx)
    start_percent = float(n_start) / float((N - n_start))
    iters_to_run = math.ceil((N - n_start) / (n_start))
    track_progress = np.linspace(0, N, 5)
    return (iters_to_run, start_percent, N, n_start, track_progress)


def information_message(models_list, dataset, AcquisitionFunction, seed, torch_seeds, model_method):
    indent = "     "
    print(
        f"\n \n \n ################ Important Files Will Be saved at the following location ################### \n"
    )
    print(f"{indent}: Model Checkpoints\n")
    for model_path in models_list:
        print(f"{indent} {indent} - {model_path}\n")
    print(f"{indent}: Json File with metrics for Train & Validation\n")
    for torch_seed in torch_seeds:
        print(
            f"{indent} {indent} - results/active_learning/train_val_{dataset}_{model_method}_{AcquisitionFunction}_{seed}_{torch_seed}\n"
        )
    print(f"{indent}: Json File Query Information\n")
    print(
        f"{indent} {indent} - results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}_{model_method}.json\n"
    )
    print(f"{indent}: Plot Savings can be found in\n")
    print(
        f"{indent} {indent} - Assets/{dataset}/{AcquisitionFunction}_{model_method}_{seed}_{torch_seed}_xxxx_pred_yyyy"
    )
    if model_method != "BatchNorm":
        print(
            f"{indent} {indent} - Assets/{dataset}/{AcquisitionFunction}_{model_method}_{seed}_{torch_seed}_xxxx_uncertain_yyyy"
        )
    if model_method == "DeepEnsemble":
        print(f"{indent}: JsonFile with metrics specific for deep ensemble\n")
        print(
            f"{indent} {indent} - results/active_learning/val_DeepEnsemble_{dataset}_{model_method}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}"
        )

    print(
        f"\n \n \n ############################################################################################ \n"
    )


def run_active(
    model_params,
    dataset="warwick",
    epochs=100,
    start_size="2-Samples",
    binary=True,
    model_method="MCD",
    seed=261,
    torch_seeds=[17],
    first_train=True,
    AcquisitionFunction="ShanonEntropy",
    device="cpu",
    n_items_to_label=2,
    turn_off_wandb=True,
    Earlystopping_=False,
    testing=False,
):
    assert model_method in ["BatchNorm", "MCD", "DeepEnsemble", "Laplace"]
    assert (
        model_params["enable_pool_dropout"] == False
        if model_method != "MCD"
        else model_params["enable_pool_dropout"] == True
    )
    # init loaders
    train_loader, val_loader, unlabeled_loader = None, None, None
    train_idx, val_idx, unlabeled_pool_idx = None, None, None
    print(f"Running on device {device}")
    models_list = [
        f"models/{dataset}_{model_method}_{AcquisitionFunction}_{seed}_{torch_seed}_{model_params['enable_pool_dropout']}.pth"
        for torch_seed in torch_seeds
    ]
    active_df = pd.DataFrame(
        {"Query_id": [], "labels_added": [], "Train_size": [], "Unlabeled_size": []}, dtype=object
    )
    iters_to_run, _, N, n_start, track_progress = how_many_iters(
        dataset=dataset, start_size=start_size, binary=binary, seed=seed
    )

    information_message(models_list, dataset, AcquisitionFunction, seed, torch_seeds, model_method)

    active_df.loc[len(active_df)] = [0, np.asarray([], dtype=object), n_start, N - n_start]

    MetricsCalulator = CollectMetrics(
        validation=True, device=device, out_ch=model_params["n_classes"]
    )

    MetricsValidatation = CollectMetrics(
        validation=True, device=device, out_ch=model_params["n_classes"]
    )

    active_train_bar = tqdm(range(iters_to_run))
    for i in active_train_bar:
        for torch_seed in torch_seeds:
            # Get DataLoaders & Train Network
            (
                train_loader,
                val_loader,
                unlabeled_loader,
                train_idx,
                val_idx,
                unlabeled_pool_idx,
            ) = train(
                dataset=dataset,
                train_size=start_size,
                epochs=epochs,
                device=device,
                binary=binary,
                enable_pool_dropout=model_params["enable_pool_dropout"],
                seed=seed,
                turn_off_wandb=turn_off_wandb,
                torch_seed=torch_seed,
                train_loader=train_loader,
                val_loader=val_loader,
                unlabeled_loader=unlabeled_loader,
                train_idx=train_idx,
                val_idx=val_idx,
                unlabeled_pool_idx=unlabeled_pool_idx,
                query_id=i,
                AcquisitionFunction=AcquisitionFunction,
                model_method=model_method,
                first_train=first_train,
                Earlystopping_=Earlystopping_,
                model_params=model_params,
            )

        # Inference On Validation Data
        if i in track_progress:
            progress_i = int([x for x in track_progress if x == i][0])  # find the progress number
            print("\n#################### Inference Plotting ####################")
            images, masks, predictions, prediction_idx = inference(
                models=models_list,
                model_params=model_params,
                data_loader=val_loader,
                method=model_method,
                seed=seed,
                torch_seeds=torch_seeds,
                dataset=dataset,
                device=device,
            )
            if model_method != "BatchNorm":
                mean_predictions = torch.mean(torch.sigmoid(predictions), dim=1)
                mean_predictions = mean_predictions.permute(0, 3, 1, 2)
                assert (
                    np.sum(
                        np.array(list(mean_predictions.shape))
                        == np.array([mean_predictions.shape[0], 1, 64, 64])
                    )
                    == 4
                )
            else:
                mean_predictions = torch.sigmoid(predictions)
            MetricsCalulator.plots(
                mean_predictions,
                masks,
                show=False,
                dataset=dataset,
                from_logits=False,
                title=f"Reliability Diagram - {Config.title_mapper[dataset]}",
                save_path=f"Assets/{dataset}/{AcquisitionFunction}_{model_method}_{seed}_{torch_seed}_{progress_i}_Diagram",
            )
            #
            fig = viz_batch(
                images[0:4],
                masks[0:4],
                predictions[0:4],
                cols=["img", "mask", "pred", "err"],
                from_logits=True,
                reduction=True if model_method != "BatchNorm" else False,
                save_=True,
                dataset=dataset,
                save_path=f"Assets/{dataset}/{AcquisitionFunction}_{model_method}_{seed}_{torch_seed}_{progress_i}_pred_04",
            )
            if model_method != "BatchNorm":
                fig = viz_batch(
                    images[0:4],
                    masks[0:4],
                    predictions[0:4],
                    cols=["var", "entropy", "mut_info", "jsd"],
                    from_logits=True,
                    reduction=True,
                    save_=True,
                    dataset=dataset,
                    save_path=f"Assets/{dataset}/{AcquisitionFunction}_{model_method}_{seed}_{torch_seed}_{progress_i}_uncertain_04",
                )
            #
            fig = viz_batch(
                images[16:20],
                masks[16:20],
                predictions[16:20],
                cols=["img", "mask", "pred", "err"],
                from_logits=True,
                reduction=True if model_method != "BatchNorm" else False,
                save_=True,
                dataset=dataset,
                save_path=f"Assets/{dataset}/{AcquisitionFunction}_{model_method}_{seed}_{torch_seed}_{progress_i}_pred_1620",
            )

            if model_method != "BatchNorm":
                fig = viz_batch(
                    images[16:20],
                    masks[16:20],
                    predictions[16:20],
                    cols=["var", "entropy", "mut_info", "jsd"],
                    from_logits=True,
                    reduction=True,
                    save_=True,
                    dataset=dataset,
                    save_path=f"Assets/{dataset}/{AcquisitionFunction}_{model_method}_{seed}_{torch_seed}_{progress_i}_uncertain_1620",
                )
            print("\n#################### Finish Inference Plotting ####################")

        if model_method == "DeepEnsemble":
            # Run Inference on the validation data
            images, masks, predictions, prediction_idx = inference(
                models=models_list,
                model_params=model_params,
                data_loader=val_loader,
                method=model_method,
                seed=seed,
                torch_seeds=torch_seeds,
                dataset=dataset,
                device=device,
            )
            # Get Mean Predictions For Ensemble
            mean_predictions = torch.mean(torch.sigmoid(predictions), dim=1)
            mean_predictions = mean_predictions.permute(0, 3, 1, 2)
            # Calculate Metrics
            NLL, Brier, ECE, MCE, Dice, IOU, Acc, Soft_Dice = MetricsValidatation.Metrics(
                mean_predictions, masks
            )
            # Store Results
            data_to_store = {
                "NLL": [NLL],
                "Brier": [Brier],
                "ECE": [ECE],
                "MCE": [MCE],
                "Dice": [Dice],
                "IOU": [IOU],
                "Acc": [Acc],
                "Soft_Dice": [Soft_Dice],
                "Query ID": [i],
            }
            arrayify_results(
                data_to_store=data_to_store,
                save_path=f"results/active_learning/val_DeepEnsemble_{dataset}_{model_method}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}",
            )

        # Run Inference with given Acquisition Function
        next_labels = find_next(
            models=models_list,
            model_method=model_method,
            unlabeled_loader=unlabeled_loader,
            unlabeled_pool_idx=unlabeled_pool_idx,
            torch_seeds=torch_seeds,
            seed=seed,
            AcquisitionFunction=AcquisitionFunction,
            dataset=dataset,
            device=device,
            model_params=model_params,
            n_items_to_label=n_items_to_label,
        )

        # Add Label To train pool
        train_idx += next_labels
        # Remove label from train pool
        [unlabeled_pool_idx.remove(x) for x in next_labels]
        first_train = False
        print(f"Adding {','.join(map(str,next_labels))} to the training pool.")
        active_train_bar.set_postfix({"Size training set": f"{len(train_idx)}/{N}"})
        active_df.loc[len(active_df)] = [
            i + 1,
            np.asarray(next_labels, dtype=object),
            len(train_idx),
            len(unlabeled_pool_idx),
        ]
        print(active_df)
        if testing and i == 1:
            print("Breaking Due To 'testing=True'")
            break
    # Save results
    active_df.to_json(
        f"results/active_learning/{dataset}_{AcquisitionFunction}_{seed}_{','.join(map(str,torch_seeds))}_{model_method}.json"
    )


# if __name__ == "__main__":
#     model_params = {
#         "in_ch": 1,
#         "n_classes": 1,
#         "bilinear_method": False,
#         "momentum": 0.9,
#         "enable_dropout": False,
#         "dropout_prob": 0.5,
#         "enable_pool_dropout": False,
#         "pool_dropout_prob": 0.5,
#     }
#     #[17,8,42,19,5]
#     run_active(model_params=model_params, dataset="PhC-C2DH-U373",epochs=3,start_size='2-Samples',model_method='DeepEnsemble',AcquisitionFunction='ShanonEntropy',torch_seeds=[17,8,42,19,5],seed=261)
