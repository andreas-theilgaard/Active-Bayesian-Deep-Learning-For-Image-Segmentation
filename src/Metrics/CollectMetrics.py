import torch
from src.Metrics.CalibrationMetrics import Calibration_Scoring_Metrics
from src.Metrics.SegmentationMetrics import SegmentationMetrics
from src.experiments.experiment_utils import arrayify_results
import os
import wandb

# from src.visualization.plot import plot_prediction_batch
from src.visualization.viz_tools import viz_batch


class CollectMetrics:
    def __init__(self, validation, device, out_ch, nbins=10, torchmetrics=False):
        self.device = device
        self.out_ch = out_ch
        self.torchmetrics = torchmetrics
        self.nbins = nbins
        self.validation = validation

        self.multiclass = True if self.out_ch > 1 else False

        self.Calib_Metrics = Calibration_Scoring_Metrics(
            nbins=self.nbins,
            torchmetrics=self.torchmetrics,
            multiclass=self.multiclass,
            device=self.device,
        )
        self.Seg_Metrics = SegmentationMetrics(
            device=self.device, multiclass=self.multiclass, torch_metrics=self.torchmetrics
        )

        # Define Arrays For Storing
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

        if self.validation:
            self.val_dice_all = []
            self.val_pixel_accuracy_all = []
            self.val_IOU_all = []
            self.val_NLL_all = []
            self.val_ECE_all = []
            self.val_MCE_all = []
            self.val_brier_all = []
            self.val_soft_dice_all = []

        # Locals
        self.dice_vec_local = []
        self.loss_local = []
        self.pixel_acc_local = []
        self.iou_local = []
        self.soft_dice_local = []
        self.NLL_local = []
        self.ECE_local = []
        self.MCE_local = []
        self.brier_local = []

    def GetMetrics(self, y_hat, y_true, loss):
        NLL, Brier, ECE, MCE = self.Calib_Metrics.Calculate_Calibration_Metrics(
            y_hat.squeeze(1), y_true
        )
        Dice, IOU, Acc, Soft_Dice = self.Seg_Metrics.Calculate_Segmentation_Metrics(
            y_hat.squeeze(1), y_true
        )

        # Append To Local Arrays
        self.dice_vec_local.append(Dice)
        self.loss_local.append(loss)
        self.pixel_acc_local.append(Acc)
        self.iou_local.append(IOU)
        self.soft_dice_local.append(Soft_Dice)
        self.NLL_local.append(NLL)
        self.ECE_local.append(ECE)
        self.MCE_local.append(MCE)
        self.brier_local.append(Brier)

        return (self.loss_local[-1], self.dice_vec_local[-1])

    def Metrics(self, y_hat, y_true):
        NLL, Brier, ECE, MCE = self.Calib_Metrics.Calculate_Calibration_Metrics(
            y_hat.squeeze(1), y_true
        )
        Dice, IOU, Acc, Soft_Dice = self.Seg_Metrics.Calculate_Segmentation_Metrics(
            y_hat.squeeze(1), y_true
        )
        return (NLL, Brier, ECE, MCE, Dice, IOU, Acc, Soft_Dice)

    def Store_Auxilary(self, data):
        NLL, Brier, ECE, MCE, Dice, IOU, Acc, Soft_Dice = self.Metrics(
            data["Logits"], data["Masks"]
        )
        self.val_dice_all.append(Dice)
        self.val_pixel_accuracy_all.append(Acc)
        self.val_IOU_all.append(IOU)
        self.val_NLL_all.append(NLL)
        self.val_ECE_all.append(ECE)
        self.val_MCE_all.append(MCE)
        self.val_brier_all.append(Brier)
        self.val_soft_dice_all.append(Soft_Dice)

    def AppendToGlobal(self, data=None):
        self.dice_global.append(torch.tensor(self.dice_vec_local).mean().item())
        self.pixel_acc_global.append(torch.tensor(self.pixel_acc_local).mean().item())
        self.IOU_score_global.append(torch.tensor(self.iou_local).mean().item())
        self.loss_global.append(torch.tensor(self.loss_local).mean().item())
        self.NLL_global.append(torch.tensor(self.NLL_local).mean().item())
        self.ECE_global.append(torch.tensor(self.ECE_local).mean().item())
        self.MCE_global.append(torch.tensor(self.MCE_local).mean().item())
        self.brier_global.append(torch.tensor(self.brier_local).mean().item())
        self.brier_local.append(torch.tensor(self.soft_dice_local).mean())

        if self.validation and data:
            self.Store_Auxilary(data)

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

    def plots(
        self, predictions, masks, dataset, title=None, from_logits=True, save_path=None, show=False
    ):
        if not os.path.exists(f"Assets/{dataset}"):
            os.mkdir(f"Assets/{dataset}")
        self.Calib_Metrics.PlotRealiabilityDiagram(
            predictions, masks, title=title, save_path=save_path, show=show, from_logits=from_logits
        )


def store_results(
    MetricsTrain, MetricsValidate, execution_time, ActiveResults, save_path=None, query_id=None
):
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
        "Execution Time": execution_time,  # Measured in seconds
    }
    if MetricsValidate.validation:
        data_to_store["val_dice_all"] = [MetricsValidate.val_dice_all]
        data_to_store["val_pixel_accuracy_all"] = [MetricsValidate.val_pixel_accuracy_all]
        data_to_store["val_IOU_all"] = [MetricsValidate.val_IOU_all]
        data_to_store["val_NLL_all"] = [MetricsValidate.val_NLL_all]
        data_to_store["val_ECE_all"] = [MetricsValidate.val_ECE_all]
        data_to_store["val_MCE_all"] = [MetricsValidate.val_MCE_all]
        data_to_store["val_brier_all"] = [MetricsValidate.val_brier_all]
        data_to_store["val_soft_dice_all"] = [MetricsValidate.val_soft_dice_all]

    if ActiveResults:
        data_to_store["Query ID"] = query_id
    return data_to_store


import os


class WandBLogger:
    def __init__(self, job_type, experiment_name):
        self.job_type = job_type
        self.experiment_name = experiment_name

    def init_wandb(self):
        if self.experiment_name and self.job_type:
            run_tracker = wandb.init(
                entity="andr_dl_projects",
                reinit=False,
                project="Active Bayesian Deep Learning For Image Segmentation",
                resume="allow",
                group=self.experiment_name,
                job_type=self.job_type,
            )
        else:
            run_tracker = wandb.init(
                entity="andr_dl_projects",
                reinit=False,
                project="Active Bayesian Deep Learning For Image Segmentation",
                resume="allow",
            )

    def log_metrics(self, type, Metrics):
        """
        type: ['Train','Validation']
        Metrics: An instance of the CollectMetrics class
        """
        wandb.log({f"{type} loss (epoch):": Metrics.loss_global[-1]})
        wandb.log({f"{type} dice (epoch):": Metrics.dice_global[-1]})
        wandb.log({f"{type} pixel accuracy (epoch):": Metrics.pixel_acc_global[-1]})
        wandb.log({f"{type} soft dice (epoch):": Metrics.soft_dice_global[-1]})
        wandb.log({f"{type} IOU (epoch):": Metrics.IOU_score_global[-1]})
        wandb.log({f"{type} ECE": Metrics.ECE_global[-1]})
        wandb.log({f"{type} MCE": Metrics.MCE_global[-1]})
        wandb.log({f"{type} Brier": Metrics.brier_global[-1]})
        wandb.log({f"{type} NLL": Metrics.NLL_global[-1]})

    def log_images(self, type_, save_, images, masks, predictions, dataset, save_path):
        if not os.path.exists(f"Assets/{dataset}"):
            os.mkdir(f"Assets/{dataset}")
        # fig = plot_prediction_batch(
        #    images, masks, predictions, save_=save_, dataset=dataset, save_path=save_path
        # )
        fig = viz_batch(
            images,
            masks,
            predictions,
            cols=["img", "mask", "pred", "err"],
            from_logits=True,
            reduction=False,
            save_=save_,
            dataset=dataset,
            save_path=save_path,
        )
        if save_ and type_ == "Validation":
            wandb.log(
                {
                    "Final High Resolution Validation Example": wandb.Image(
                        f"{save_path}_predictions.png"
                    )
                }
            )
        wandb.log({f"{type_} Predictions": fig})
