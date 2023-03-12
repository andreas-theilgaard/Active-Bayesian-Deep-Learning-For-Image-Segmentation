import torch

# ECE: Expected Calibration Error
# MCE: Maximum Calibration
# BRIOR Score:

import numpy as np
from numpy import ndarray as array
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from typing import Tuple, List, Union, Dict
import uncertainty_toolbox as uct
from torch.nn.functional import softmax
from torch.nn.functional import one_hot


def calibration_curve(
    probabilities: array, labels: array, bins: int = 20
) -> Tuple[float, array, array, array]:
    r"""Computes the Expected Calibration Error (ECE) of the predicted class probabilities.
    With accuracy `acc` and confidence `conf`, it is defined as
    :math:`ECE=\sum_{m=1}^M\frac{\left|B_m\right|}{n}\left|\mathrm{acc}(B_M)-\mathrm{conf}(B_m)\right|`
    where `n` is the number of samples and `B_m` are all samples in bin `m` from `M` with equal number of samples.
    Source: `A Simple Baseline for Bayesian Neural Networks <https://arxiv.org/abs/1902.02476>`_.
    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
        bins: The number of bins into which the probabilities are discretized.
    Returns:
        The ECE alongside the average confidence, accuracy and proportion of data points in each bin respectively.
    """
    confidences = np.max(probabilities, 1)
    # confidences = probabilities
    step = (confidences.shape[0] + bins - 1) // bins
    bins = np.sort(confidences)[::step]
    if confidences.shape[0] % step != 1:
        bins = np.concatenate((bins, [np.max(confidences)]))
    # bins = np.linspace(0.1, 1.0, 30)
    predictions = np.argmax(probabilities, 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    accuracies = predictions == labels

    xs = []
    ys = []
    zs = []

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) * (confidences < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            xs.append(avg_confidence_in_bin)
            ys.append(accuracy_in_bin)
            zs.append(prop_in_bin)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    return ece, xs, ys, zs


def accuracy(probabilities: array, labels: array) -> float:
    """Computes the top 1 accuracy of the predicted class probabilities in percent.
    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
    Returns:
        The top 1 accuracy in percent.
    """
    return 100.0 * np.mean(np.argmax(probabilities, axis=1) == labels)


def calibration(
    probabilities,
    labels,
    bins=20,
    swag=True,
    axis=None,
    label=None,
    linestyle="-",
    alpha=1.0,
    color="crimson",
    path="",
):
    ece, bin_confs, bin_accs, _ = calibration_curve(probabilities, labels, bins)
    bin_aces = bin_confs - bin_accs

    if axis is None:
        fig, ax = plt.subplots(figsize=(12, 7), tight_layout=True)
    else:
        ax = axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", labelsize=14, right=False, top=False)
    ax.set_xlabel("Confidence", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)
    if swag:
        ax.set_ylabel("Confidence - Accuracy", fontsize=16)

        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.plot(
            bin_confs,
            bin_aces,
            marker="o",
            label=f"{label} | ECE: {100 * ece:.2f}%" if label is not None else None,
            linewidth=2,
            linestyle=linestyle,
            alpha=alpha,
            color=color,
        )

        ax.set_xscale("logit")
        ax.set_xlim(0.1, 0.999999)
        ax.minorticks_off()
        plt.xticks(
            [0.2, 0.759, 0.927, 0.978, 0.993, 0.998, 0.999999],
            labels=[0.2, 0.759, 0.927, 0.978, 0.993, 0.998, 1],
        )

        if label is not None:
            ax.legend(fontsize=16, frameon=False)
    else:
        ax.set_ylim(0.2, 1)
        ax.plot(
            ax.get_xlim(),
            ax.get_ylim(),
            color="black",
            linestyle="dashed",
            linewidth=1,
            dashes=(5, 10),
        )
        ax.plot(bin_confs, bin_accs, color="blueviolet", marker="o", linewidth=2)
    plt.show()
    # if axis is None:
    #    plt.savefig(path + "_calibration.pdf", format='pdf', dpi=1200)


def confidence(probabilities: array, mean: bool = True) -> Union[float, array]:
    """The confidence of a prediction is the maximum of the predicted class probabilities.
    Args:
        probabilities: The predicted class probabilities.
        mean: If True, returns the average confidence over all provided predictions.
    Returns:
        The confidence.
    """
    if mean:
        return np.mean(np.max(probabilities, axis=1))
    return np.max(probabilities, axis=1)


def expected_calibration_error(
    probabilities: array, labels: array, bins: int = 10
) -> Tuple[float, array, array, array]:
    r"""Computes the Expected Calibration Error (ECE) of the predicted class probabilities.
    With accuracy `acc` and confidence `conf`, it is defined as
    :math:`ECE=\sum_{m=1}^M\frac{\left|B_m\right|}{n}\left|\mathrm{acc}(B_M)-\mathrm{conf}(B_m)\right|`
    where `n` is the number of samples and `B_m` are all samples in bin `m` from `M` equally-spaced bins.
    Source: `On Calibration of Modern Neural Networks <https://arxiv.org/pdf/1706.04599.pdf)?>`_.
    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
        bins: The number of bins into which the probabilities are discretized.
    Returns:
        The ECE alongside the average confidence, accuracy and proportion of data points in each bin respectively.
    """
    conf = confidence(probabilities, mean=False)
    edges = np.linspace(0, 1, bins + 1)
    bin_ace = list()
    bin_accuracy = list()
    bin_confidence = list()
    ece = 0
    for i in range(bins):
        mask = np.logical_and(conf > edges[i], conf <= edges[i + 1])
        if any(mask):
            bin_acc = accuracy(probabilities[mask], labels[mask]) / 100
            bin_conf = conf[mask].mean()
            ace = bin_conf - bin_acc
            ece += mask.mean() * np.abs(ace)

            bin_ace.append(ace)
            bin_accuracy.append(bin_acc)
            bin_confidence.append(bin_conf)
        else:
            bin_ace.append(0)
            bin_accuracy.append(0)
            bin_confidence.append(0)
    return ece, np.array(bin_ace), np.array(bin_accuracy), np.array(bin_confidence)


def reliability_diagram(probabilities, labels, path="", bins=10, axis=None):
    ece, bin_aces, bin_accs, bin_confs = expected_calibration_error(
        probabilities, labels, bins=bins
    )
    if axis is None:
        text = offsetbox.AnchoredText(
            f"ECE: {(ece * 100):.2f}%\nAccuracy: {accuracy(probabilities, labels):.2f}%\nConfidence: {100 * confidence(probabilities):.2f}%",
            loc="upper left",
            frameon=False,
            prop=dict(fontsize=12),
        )
        fig, ax = plt.subplots(figsize=(9, 9), tight_layout=True)
        ax.add_artist(text)
    else:
        ax = axis
    ax.bar(
        x=np.arange(0, 1, 0.1),
        height=bin_accs,
        width=0.1,
        linewidth=1,
        edgecolor="black",
        align="edge",
        color="dodgerblue",
    )
    ax.bar(
        x=np.arange(0, 1, 0.1),
        height=bin_aces,
        bottom=bin_accs,
        width=0.1,
        linewidth=1,
        edgecolor="crimson",
        align="edge",
        color="crimson",
        fill=False,
        hatch="/",
    )
    ax.bar(
        x=np.arange(0, 1, 0.1),
        height=bin_aces,
        bottom=bin_accs,
        width=0.1,
        linewidth=1,
        edgecolor="crimson",
        align="edge",
        color="crimson",
        alpha=0.3,
    )
    if axis is None:
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.plot(ax.get_xlim(), ax.get_ylim(), color="black", linestyle="dashed", linewidth=1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", labelsize=12, right=False, top=False)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_xlabel("Confidence", fontsize=14)
        # plt.savefig(path if path else 'reliability_diagram.pdf', format='pdf', dpi=1200)
    else:
        ax.tick_params(
            right=False,
            left=False,
            top=False,
            bottom=False,
            labelright=False,
            labelleft=False,
            labeltop=False,
            labelbottom=False,
        )
        ax.set_frame_on(False)
    plt.show()


def confidence_hist(probabilities, labels=None, path=""):
    _confidence = confidence(probabilities, mean=False)
    weights = np.ones_like(_confidence) / len(_confidence)
    mean_confidence = np.mean(_confidence)

    fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True)
    ax.hist(
        _confidence, bins=20, edgecolor="black", linewidth=1, weights=weights, color="dodgerblue"
    )
    conf_line = 0.72
    conf_text = 1.1
    if labels is not None:
        mean_accuracy = accuracy(probabilities, labels)
        if mean_confidence > mean_accuracy:
            acc_line = conf_line
            acc_text = 1.1
            conf_line = 0.69
            conf_text = acc_text
        else:
            acc_line = 0.8
            acc_text = 1.3
        ax.vlines(mean_accuracy, ymin=0, ymax=acc_line, linestyles="dashed")
        ax.scatter(
            mean_accuracy,
            acc_line,
            s=30,
            edgecolor="black",
            facecolor="white",
            marker="o",
            linewidth=1.5,
        )
        ax.text(
            mean_accuracy,
            acc_text,
            f"Accuracy: {100 * mean_accuracy:.2f}%",
            rotation=45,
            verticalalignment="top",
            fontsize=14,
        )
    ax.vlines(mean_confidence, ymin=0, ymax=conf_line, linestyles="dashed")
    ax.scatter(
        mean_confidence,
        conf_line,
        s=30,
        edgecolor="black",
        facecolor="white",
        marker="o",
        linewidth=1.5,
    )
    ax.text(
        mean_confidence,
        conf_text,
        f"Confidence: {100 * mean_confidence:.2f}%",
        rotation=45,
        verticalalignment="top",
        fontsize=14,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.tick_params(direction="out", labelsize=14, right=False, top=False)
    ax.set_ylabel("Frequency", fontsize=16)
    ax.set_xlabel("Confidence", fontsize=16)
    plt.show()
    # plt.savefig(path if path else 'confidence_hist.pdf', format='pdf', dpi=1200)


# expected_calibration_error(np.array([0.25,0.25,0.55,0.75,0.75]),np.array([0,0,1,1,1]),bins=10)
def ECE_error(probs, em_list, k=15):
    """
    Inputs:
        probs: list containing the confidence scores
        em_list: list containing the exact match (EM) scores
        n_bins: The number of intervals the prob_list is divided into
    Output:
        Return Expected Calibration Error
    """
    bin_bounds = np.linspace(np.min(probs), np.max(probs), k + 1)
    n = np.size(em_list)
    ECE = 0
    confs = []
    accs = []
    for i in range(len(bin_bounds) - 1):
        em_in_bin = em_list[
            (probs > bin_bounds[i]) & (probs < bin_bounds[i + 1])
        ]  # calculates accuracy in i'th bin
        conf_in_bin = probs[
            (probs > bin_bounds[i]) & (probs < bin_bounds[i + 1])
        ]  # calculates confidence in i'th bin

        # take then mean of Acc and Conf
        confs.append(np.mean(conf_in_bin))
        accs.append(np.mean(em_in_bin))
        numb_in_bin = np.size(em_in_bin)  # find the total number on questions in bin

        ECE_i = np.abs(np.mean(em_in_bin) - np.mean(conf_in_bin)) * (
            numb_in_bin / n
        )  # calculate calibration error for that bin

        ECE += ECE_i  # sum for all i'th calibration errors

    return (confs, accs, ECE)


# ECE_error(np.array([0.25,0.25,0.55,0.75,0.75]),np.array([0,0,1,1,1]),k=2)

# np.float32((np.array([0.25,0.25,0.55,0.75,0.75])>=0.5))

# import torchmetrics
# np.random.seed(0)
# labels = torch.tensor(np.round(np.random.uniform(0,1,200)))
# probabilities = torch.tensor(np.random.uniform(0,1,(200,1)))
# #probs = torch.tensor(np.max(probabilities,1))
# prob_std = torch.tensor(np.random.uniform(0,0.5,200))

# num_bins=10


def Calibration_Errors():
    pass


def accuracy(probabilities: array, labels: array) -> float:
    """Computes the top 1 accuracy of the predicted class probabilities in percent.
    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
    Returns:
        The top 1 accuracy in percent.
    """
    return 100.0 * np.mean(np.argmax(probabilities, axis=1) == labels)


def get_max_prob(y_hat):
    # if self.multiclass:
    #    print("not implemented yet")
    # return torch.max(torch.softmax(y_hat,dim=1),dim=1)[0]
    # else:
    return torch.sigmoid(y_hat).squeeze(1)


def get_prediction(y_hat):
    return y_hat >= 0.5


def Calibration_Errors(y_hat, y_true, num_bins, is_prob=False):
    confidence = y_hat
    if is_prob == False:
        confidence = get_max_prob(y_hat)
    bin_bounds = np.linspace(0, 1, num_bins + 1)
    bin_ace = []  # Absolut calibration error
    bin_accuracy = []  # Accuracy in bin
    bin_confidence = []  # Confidence in bin

    ECE = 0  # ECE

    for i in range(num_bins):
        mask = np.logical_and((confidence > bin_bounds[i]), (confidence < bin_bounds[i + 1]))
        if any(mask):
            bin_acc = np.mean(
                get_prediction(y_hat)[mask] == y_true[mask]
            )  # calculates accuracy in i'th bin
            bin_conf = np.mean(confidence[mask])  # calculates confidence in i'th bin

            # absoulut calibration
            ace = np.abs(bin_conf - bin_acc)  # *****
            ECE += np.mean(mask) * ace

            # store values
            bin_ace.append(ace)
            bin_accuracy.append(bin_acc)
            bin_confidence.append(bin_conf)
        else:
            bin_ace.append(0)
            bin_accuracy.append(0)
            bin_confidence.append(0)
    MCE = np.max(ace)
    return ECE, MCE, np.array(bin_ace), np.array(bin_accuracy), np.array(bin_confidence)


# Calibration_Errors(np.array([0.25,0.25,0.55,0.75,0.75]),np.array([0,0,1,1,1]),num_bins=2,is_prob=True)


# (0.75*1)*(0.4)
# (0.31666667*1)*(0.6)

# np.mean(np.array([(0.19000000199999997),(0.31666667)]))

# np.logical_and((probs>bin_bounds[i]) & (probs<bin_bounds[i+1]))


# metrics = uct.metrics.get_all_metrics(probs,prob_std,labels,num_bins=10)

# expected_calibration_error(probabilities,labels,bins=10)

# torch_labels = torch.tensor(labels,dtype=torch.float32)
# torch_probs = torch.tensor(probs,dtype=torch.float32)

# ECE = torchmetrics.CalibrationError(task='binary',n_bins=10,norm="l1")
# ECE(torch_probs,torch_labels)

# torch.manual_seed(70)
# model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True))
# x =torch.rand((4,1,64,64))
# #y_true = ((0-3)*torch.rand((4,64,64))+3).round().to(dtype=torch.int64)
# y_true = torch.rand((4,64,64)).round()
# y_hat = model(x)
# #criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCEWithLogitsLoss()
# print(criterion(y_hat.squeeze(1),y_true))


class Calibration_Scoring_Metrics:
    def __init__(self, nbins, multiclass, device, is_prob=False):

        self.nbins = nbins
        self.multiclass = multiclass
        self.device = device
        self.is_prob = is_prob

    def brier_score(self, y_hat, y_true, sample_wise=True):
        if isinstance(y_hat, array) and isinstance(y_true, array):
            y_hat, y_true = torch.tensor(y_hat).to(self.device), torch.tensor(y_true).to(
                self.device
            )
        if self.multiclass:
            n_classes = y_hat.shape[1]
            p = softmax(y_hat, dim=1)
            y_true = one_hot(y_true.to(dtype=torch.int64), n_classes).permute(0, 3, 1, 2)
            brier = torch.mean(
                torch.sum(((p - y_true) ** 2), dim=1).view(y_hat.shape[0], -1), dim=1, keepdim=True
            )
            if not sample_wise:
                brier = torch.mean(brier)
            return brier.detach().cpu().numpy().ravel()
        else:
            p = torch.sigmoid(y_hat)
            brier = torch.mean(((p - y_true) ** 2).view(y_hat.shape[0], -1), dim=1, keepdim=True)
            if not sample_wise:
                brier = torch.mean(brier)
            return brier.detach().cpu().numpy().ravel()

    def get_max_prob(self, y_hat):
        # if self.multiclass:
        #    print("not implemented yet")
        # return torch.max(torch.softmax(y_hat,dim=1),dim=1)[0]
        # else:
        return torch.sigmoid(y_hat).squeeze(1)

    def get_prediction(self, y_hat):
        return y_hat >= 0.5

    def Calibration_Errors(self, y_hat, y_true, num_bins, is_prob=False):
        confidence = y_hat
        if is_prob == False:
            confidence = self.get_max_prob(y_hat)
        bin_bounds = np.linspace(0, 1, num_bins + 1)
        bin_ace = []  # Absolut calibration error
        bin_accuracy = []  # Accuracy in bin
        bin_confidence = []  # Confidence in bin

        ECE = 0  # ECE

        for i in range(num_bins):
            mask = (confidence > bin_bounds[i]) & (confidence < bin_bounds[i + 1])
            if torch.any(mask):
                bin_acc = torch.mean(
                    (
                        (self.get_prediction(y_hat)[mask]).float()
                        == (y_true.unsqueeze(0)[mask]).float()
                    ).float()
                )  # calculates accuracy in i'th bin
                bin_conf = torch.mean(confidence[mask].float())  # calculates confidence in i'th bin

                # absoulut calibration
                ace = abs(bin_conf - bin_acc)  # *****
                ECE += torch.mean(mask.float()) * ace

                # store values
                bin_ace.append(ace.item())
                bin_accuracy.append(bin_acc.item())
                bin_confidence.append(bin_conf.item())
            else:
                bin_ace.append(0)
                bin_accuracy.append(0)
                bin_confidence.append(0)
        MCE = np.max(np.array(bin_ace))
        out = {"ECE": ECE.item(), "MCE": MCE}
        return out
        # return ECE.item(), MCE,np.array(bin_ace), np.array(bin_accuracy), np.array(bin_confidence)

    def NLL(self, y_hat, y_true, sample_wise=True):
        if not self.multiclass:
            if isinstance(y_hat, array) and isinstance(y_true, array):
                y_hat, y_true = torch.tensor(y_hat).to(self.device), torch.tensor(y_true).to(
                    self.device
                )
            # y_hat,y_true = y_hat.ravel(),y_true.ravel()
            p = torch.sigmoid(y_hat)
            # nll = -torch.mean(torch.sum(torch.log(p)*y_true.unsqueeze(1),dim=1).view(y_hat.shape[0],-1),dim=1,keepdim=True)
            # nll = -torch.mean((y_true*torch.log(p) + (1-y_true)*torch.log(1-p)))
            nll = -torch.mean(
                (
                    y_true.unsqueeze(1) * torch.log(p)
                    + (1 - y_true.unsqueeze(1)) * torch.log(1 - p)
                ).view(y_hat.shape[0], -1),
                dim=1,
                keepdim=True,
            )
            if not sample_wise:
                nll = torch.mean(nll)
            return nll.detach().cpu().numpy().ravel()
        else:
            if torch.is_tensor(y_hat) and torch.is_tensor(y_true):
                p = softmax(y_hat, dim=1)
                y_true = one_hot(y_true, y_hat.shape[1]).permute(0, 3, 1, 2)
                nll = -torch.mean(
                    torch.sum(torch.log(p) * y_true, dim=1).view(y_hat.shape[0], -1),
                    dim=1,
                    keepdim=True,
                )
                if not sample_wise:
                    nll = torch.mean(nll)
                return nll.detach().cpu().numpy().ravel()

    def metrics(self, y_hat, y_true, sample_wise=True):
        """
        Samplewise get metrics for each sample in batch
        """
        NLL = self.NLL(y_hat, y_true, sample_wise=sample_wise)
        brier_score = self.brier_score(y_hat, y_true, sample_wise=sample_wise)
        calib_metrics = [
            self.Calibration_Errors(y_hat[x], y_true[x], num_bins=self.nbins, is_prob=self.is_prob)
            for x in range(y_hat.shape[0])
        ]
        return {
            "NLL": NLL,
            "brier_score": brier_score,
            "ECE": [calib_metrics[x]["ECE"] for x in range(len(calib_metrics))],
            "MCE": [calib_metrics[x]["MCE"] for x in range(len(calib_metrics))],
        }


if __name__ == "__main__":
    ged = Calibration_Scoring_Metrics(nbins=10, multiclass=False, device="cpu")
    print(ged.metrics(y_hat, y_true))
    # print(Calibration_Errors(torch.tensor(np.array([0.25,0.25,0.55,0.75,0.75])),torch.tensor(np.array([0,0,1,1,1])),num_bins=2,is_prob=True))
    # print([Calibration_Errors(y_hat=y_hat[x],y_true=y_true[x],num_bins=15,is_prob=False) for x in range(y_hat.shape[0])])
    # Calibration_Errors()


# if __name__ == "__main__":
#     np.random.seed(0)
#     labels = np.round(np.random.uniform(0,1,200))
#     probabilities = np.random.uniform(0,1,(200,2))

#     #calibration(probabilities, labels, bins=20, swag=True, axis=None, label=None, linestyle='-', alpha=1.0,color='crimson', path="")
#     #confidence_hist(probabilities,labels)
#     reliability_diagram(probabilities, labels, path="", bins=10, axis=None)
