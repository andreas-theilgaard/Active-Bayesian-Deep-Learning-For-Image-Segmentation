import torch
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch import cat, zeros, stack
from math import *
import random
import pytest
from src.models.laplace_utils import exact_hessian
from backpack import extend, backpack, extensions
from torch.distributions.multivariate_normal import MultivariateNormal

######################### 2D Case #############################
seed = 7777

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(99999)  # Set Torch Seed
torch.cuda.manual_seed_all(99999)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
train_size = 90
train_range = (4, 7.5)
test_size = 50
test_range = (0, 10)


def MakeData(size, n_classes, train_range, test_size, test_range):
    if n_classes == 2:
        X_train, y_train = datasets.make_blobs(
            n_samples=size,
            centers=n_classes,
            cluster_std=0.7,
            center_box=train_range,
            random_state=62,
        )
        test_rng = np.linspace(*test_range, test_size)
        X1_test, X2_test = np.meshgrid(test_rng, test_rng)
        X_test = np.stack([X1_test.ravel(), X2_test.ravel()]).T
        X_test = torch.from_numpy(X_test).float()
        return (X_train, y_train, X_test, X1_test, X2_test)
    else:
        X_train, y_train = datasets.make_blobs(
            n_samples=size,
            centers=n_classes,
            cluster_std=1.2,
            center_box=train_range,
            random_state=37,
        )
        test_rng = np.linspace(*test_range, test_size)
        X1_test, X2_test = np.meshgrid(test_rng, test_rng)
        X_test = np.stack([X1_test.ravel(), X2_test.ravel()]).T
        X_test = torch.from_numpy(X_test).float()
        return (X_train, y_train, X_test, X1_test, X2_test)


def plotBinary(X, Y, X1_test, X2_test, py_list, conf_list, size=120, test_range=None, show=False):
    ims = []
    cmap = "Blues"

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(6, 5)

    for i in range(len(py_list)):
        py = py_list[i]
        conf = conf_list[i]
        # Decision boundary contour
        axes[i].contour(
            X1_test, X2_test, py.reshape(size, size), levels=[0.5], colors="black", linewidths=[3]
        )

        # Background shade, representing confidence
        conf = np.clip(conf, 0, 0.999999)
        im = axes[i].contourf(
            X1_test,
            X2_test,
            conf.reshape(size, size),
            alpha=0.7,
            levels=np.arange(0.5, 1.01, 0.1),
            cmap=cmap,
            vmin=0.5,
            vmax=1,
        )
        fig.colorbar(im, ax=axes[i])

        # Scatter plot the training data
        axes[i].scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], c="coral", edgecolors="k", linewidths=0.5)
        axes[i].scatter(
            X[Y == 1][:, 0], X[Y == 1][:, 1], c="yellow", edgecolors="k", linewidths=0.5
        )

        axes[i].set_xlim(test_range)
        axes[i].set_ylim(test_range)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        if i == 0:
            axes[i].set_title("MAP")
        if i == 1:
            axes[i].set_title("Laplace")

    if show:
        plt.show()


def plotMutliClass(X, Y, X1_test, X2_test, Z_list, test_range=None, show=False):
    ims = []
    cmap = "Blues"

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(6, 5)

    for i in range(len(Z_list)):
        Z = Z_list[i]

        im = axes[i].contourf(
            X1_test, X2_test, Z, alpha=0.7, cmap=cmap, levels=np.arange(0.3, 1.01, 0.1)
        )
        fig.colorbar(im, ax=axes[i])

        axes[i].scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], c="coral", edgecolors="k", linewidths=0.5)
        axes[i].scatter(
            X[Y == 1][:, 0], X[Y == 1][:, 1], c="yellow", edgecolors="k", linewidths=0.5
        )
        axes[i].scatter(
            X[Y == 2][:, 0], X[Y == 2][:, 1], c="yellowgreen", edgecolors="k", linewidths=0.5
        )
        axes[i].scatter(
            X[Y == 3][:, 0], X[Y == 2][:, 1], c="violet", edgecolors="k", linewidths=0.5
        )

        axes[i].set_xlim(test_range)
        axes[i].set_ylim(test_range)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        if i == 0:
            axes[i].set_title("MAP")
        if i == 1:
            axes[i].set_title("Laplace")

    if show:
        plt.show()


class ToyModel(nn.Module):
    def __init__(self, X, out=1, hidden=20):
        super().__init__()
        self.n = X.shape[1]
        self.hidden = hidden
        self.out = out

        self.features = nn.Sequential(
            nn.Linear(self.n, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
        )
        self.last_layer = nn.Linear(self.hidden, self.out, bias=True)

    def forward(self, x):
        features = self.features(x)
        return self.last_layer(features)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


########## Binary Case ############
@pytest.mark.skipif(1 == 1, reason="Too Computationally Expensive - Run Local !!!")
def test_binary():
    X, Y, X_test, X1_test, X2_test = MakeData(
        size=train_size,
        n_classes=2,
        train_range=train_range,
        test_size=test_size,
        test_range=test_range,
    )

    X_train, y_train = torch.from_numpy(X).float(), torch.from_numpy(Y).float()

    model = ToyModel(X=X, out=1, hidden=20)
    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    # Fit MAP
    for it in tqdm(range(5000)):
        y_pred = model(X_train).squeeze()
        l = F.binary_cross_entropy_with_logits(y_pred, y_train)
        l.backward()
        opt.step()
        opt.zero_grad()

    print(f"Loss: {l.item():.3f}")

    # Use MAP on Validation
    with torch.no_grad():
        model.eval()
        py = sigmoid(model(X_test).squeeze().numpy())

    conf = np.maximum(py, 1 - py)

    # Using Laplace

    # Get Weights For Last Layer
    Weights_last = list(model.parameters())[-2:]
    W = Weights_last[0]
    bias = Weights_last[1]
    Weights_map = W.view(-1).data.numpy()

    def neg_log_posterior(var0, W):
        # Negative-log-likelihood
        nll = F.binary_cross_entropy_with_logits(model(X_train).squeeze(), y_train, reduction="sum")
        # Negative-log-prior
        nlp = 1 / 2 * W.flatten() @ (1 / var0 * torch.eye(W.numel())) @ W.flatten()
        return nll + nlp

    def get_covariance(var0, W):
        # Outputs the inverse-Hessian of the negative-log-posterior at the MAP estimate
        # This is the posterior covariance
        loss = neg_log_posterior(var0, W=W)
        Lambda = exact_hessian(loss, [W])  # The Hessian of the negative log-posterior
        Sigma = torch.inverse(Lambda).detach().numpy()
        return Sigma

    @torch.no_grad()
    def predict(x, Sigma, w_map):
        phi = model.features(x).numpy()  # Feature vector of x
        m = (phi @ w_map) + bias.item()  # MAP prediction

        assert m[-1] == model(x)[-1].item()
        # "Moderate" the MAP prediction using the variance (see MacKay 1992 "Evidence Framework ...")
        # This is an approximation of the expected sigmoid (the so-called "probit approximation")
        v = np.diag(phi @ Sigma @ phi.T)
        py = sigmoid(m / np.sqrt(1 + pi / 8 * v))
        return py

    # prior precision is weight decay
    var0 = 1 / 5e-4
    # Posterior Covariance
    Sigma = get_covariance(var0=var0, W=W)

    # Make Predictions
    pyLaplace = predict(X_test, Sigma, w_map=Weights_map)
    conf_Laplace = np.maximum(pyLaplace, 1 - pyLaplace)

    plotBinary(
        X,
        Y,
        X1_test,
        X2_test,
        py_list=[py, pyLaplace],
        conf_list=[conf, conf_Laplace],
        size=test_size,
        show=False,
    )

    assert abs(Sigma[3, 2] - (49.63333)) < 0.0001
    assert abs(Sigma[-1, -1] - 532.9522) < 0.0001
    assert abs(Sigma[0, 0] - 475.2173) < 0.0001
    assert (
        np.sum(
            np.array(pyLaplace[0:5])
            - np.array([0.43713516, 0.43865892, 0.4402388, 0.44186565, 0.44406924])
        )
        < 0.0001
    )


########## Multiclass Case ############
@pytest.mark.skipif(1 == 1, reason="Too Computationally Expensive - Run Local !!!")
def test_multiclass():
    X, Y, X_test, X1_test, X2_test = MakeData(
        size=500, n_classes=4, train_range=(-10, 10), test_size=50, test_range=(-15, 15)
    )

    X_train, y_train = torch.from_numpy(X).float(), torch.from_numpy(Y).long()

    model = ToyModel(X=X, out=4, hidden=20)
    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    # Fit MAP
    for it in tqdm(range(5000)):
        y_pred = model(X_train).squeeze()
        l = F.cross_entropy(y_pred, y_train)
        l.backward()
        opt.step()
        opt.zero_grad()

    print(f"Loss: {l.item():.3f}")

    with torch.no_grad():
        model.eval()
        py = F.softmax(model(X_test), 1).squeeze().numpy()

    conf = py.max(1)

    # Laplace
    Weights_last = list(model.parameters())[-2]
    extend(model.last_layer)
    loss_func = extend(nn.CrossEntropyLoss(reduction="sum"))
    loss = loss_func(model(X_train), y_train)
    with backpack(extensions.KFAC()):
        loss.backward()
    A, B = Weights_last.kfac
    prec0 = 5e-4
    U = torch.inverse(A + sqrt(prec0) * torch.eye(Weights_last.shape[0]))
    V = torch.inverse(B + sqrt(prec0) * torch.eye(Weights_last.shape[1]))

    @torch.no_grad()
    def predict(x, W):
        phi = model.features(x)

        # MAP prediction
        m = phi @ W.T

        # v is the induced covariance.
        # See Appendix B.1 of https://arxiv.org/abs/2002.10118 for the detail of the derivation.
        v = torch.diag(phi @ V @ phi.T).reshape(-1, 1, 1) * U

        # The induced distribution over the output (pre-softmax)
        output_dist = MultivariateNormal(m, v)

        # MC-integral
        n_sample = 1000
        py = 0

        for _ in range(n_sample):
            out_s = output_dist.rsample()
            py += torch.softmax(out_s, 1)

        py /= n_sample

        return py.numpy()

    pyLaplace = predict(X_test, W=Weights_last)
    conf_Laplace = pyLaplace.max(1)

    plotMutliClass(
        X,
        Y,
        X1_test,
        X2_test,
        Z_list=[conf.reshape(50, 50), conf_Laplace.reshape(50, 50)],
        test_range=(-15, 15),
        show=False,
    )
    assert (
        abs(
            np.sum(
                conf_Laplace[0:5]
                - np.array([0.6981487, 0.7220149, 0.7056564, 0.7108208, 0.70511055])
            )
        )
        < 0.0001
    )


# if __name__ == "__main__":
#   test_binary()
#   test_multiclass()
