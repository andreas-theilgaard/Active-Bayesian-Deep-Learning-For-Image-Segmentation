##################################################################################################
#
#  Exact Hessian
#
##################################################################################################

##########################################################################
#
#  Courtesy of Felix Dangel: https://github.com/f-dangel/backpack
#
##########################################################################

"""Exact computation of full Hessian using autodiff."""
from torch import cat, zeros, stack
from torch.autograd import grad
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch
import numpy as np


def exact_hessian(f, parameters, show_progress=False):
    r"""Compute all second derivatives of a scalar w.r.t. `parameters`.
    ​
      The order of parameters corresponds to a one-dimensional
      vectorization followed by a concatenation of all tensors in
      `parameters`.
    ​
      Parameters
      ----------
      f : scalar torch.Tensor
        Scalar PyTorch function/tensor.
      parameters : list or tuple or iterator of torch.Tensor
        Iterable object containing all tensors acting as variables of `f`.
      show_progress : bool
        Show a progressbar while performing the computation.
    ​
      Returns
      -------
      torch.Tensor
        Hessian of `f` with respect to the concatenated version
        of all flattened quantities in `parameters`

      Note
      ----
      The parameters in the list are all flattened and concatenated
      into one large vector `theta`. Return the matrix :math:`d^2 E /
      d \theta^2` with

      .. math::
    ​
        (d^2E / d \theta^2)[i, j] = (d^2E / d \theta[i] d \theta[j]).
    ​
      The code is a modified version of
      https://discuss.pytorch.org/t/compute-the-hessian-matrix-of-a-
      network/15270/3
    """
    params = list(parameters)
    if not all(p.requires_grad for p in params):
        raise ValueError("All parameters have to require_grad")
    df = grad(f, params, create_graph=True)
    # flatten all parameter gradients and concatenate into a vector
    dtheta = None
    for grad_f in df:
        dtheta = (
            grad_f.contiguous().view(-1)
            if dtheta is None
            else cat([dtheta, grad_f.contiguous().view(-1)])
        )
    # compute second derivatives
    hessian_dim = dtheta.size(0)
    hessian = zeros(hessian_dim, hessian_dim)
    progressbar = tqdm(
        iterable=range(hessian_dim),
        total=hessian_dim,
        desc="[exact] Full Hessian",
        disable=(not show_progress),
    )
    for idx in progressbar:
        df2 = grad(dtheta[idx], params, create_graph=True)
        d2theta = None
        for d2 in df2:
            d2theta = (
                d2.contiguous().view(-1)
                if d2theta is None
                else cat([d2theta, d2.contiguous().view(-1)])
            )
        hessian[idx] = d2theta
    return hessian


def exact_hessian_diagonal_blocks(f, parameters, show_progress=True):
    """Compute diagonal blocks of a scalar function's Hessian.
    ​
        Parameters
        ----------
        f : scalar of torch.Tensor
            Scalar PyTorch function
        parameters : list or tuple or iterator of torch.Tensor
            List of parameters whose second derivatives are to be computed
            in a blockwise manner
        show_progress : bool, optional
            Show a progressbar while performing the computation.
    ​
        Returns
        -------
        list of torch.Tensor
            Hessian blocks. The order is identical to the order specified
            by `parameters`
    ​
        Note
        ----
        For each parameter, `exact_hessian` is called.
    """
    return [exact_hessian(f, [p], show_progress=show_progress) for p in parameters]


##################################################################################################
##################################################################################################


##################################################################################################
#
#  KFAC Laplace
#
##################################################################################################
##########################################################################
#
#  Taken with modifications from
#  https://github.com/wjmaddox/swa_gaussian/
#
##########################################################################


import torch
from torch import nn, optim, autograd
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from math import *
from tqdm import tqdm, trange
import numpy as np

# import laplace.util as lutil


class KFLA(nn.Module):
    """
    Taken, with modification, from:
    https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/diag_laplace.py
    """

    def __init__(self, base_model, device="cpu"):
        super().__init__()

        self.net = base_model
        self.params = []
        self.net.apply(lambda module: kfla_parameters(module, self.params))
        self.hessians = None
        self.device = device

    def forward(self, x):
        return self.net.forward(x)

    def forward_sample(self, x):
        self.sample()
        return self.net.forward(x)

    def sample(self, scale=1, require_grad=False):
        for module, name in self.params:
            mod_class = module.__class__.__name__
            if mod_class not in ["Linear", "Conv2d"]:
                continue

            if name == "bias":
                w = module.__getattr__(f"{name}_mean")
            else:
                M = module.__getattr__(f"{name}_mean")
                U_half = module.__getattr__(f"{name}_U_half")
                V_half = module.__getattr__(f"{name}_V_half")

                if len(M.shape) == 1:
                    M_ = M.unsqueeze(1)
                elif len(M.shape) > 2:
                    M_ = M.reshape(M.shape[0], np.prod(M.shape[1:]))
                else:
                    M_ = M

                E = torch.randn(*M_.shape, device=self.device)
                w = M_ + scale * U_half @ E @ V_half
                w = w.reshape(*M.shape)

            if require_grad:
                w.requires_grad_()

            module.__setattr__(name, w)

    def estimate_variance(self, var0, invert=True):
        tau = 1 / var0

        U, V = self.hessians

        for module, name in self.params:
            mod_class = module.__class__.__name__
            if mod_class not in ["Linear", "Conv2d"]:
                continue

            if name == "bias":
                continue

            U_ = U[(module, name)].clone()
            V_ = V[(module, name)].clone()

            if invert:
                m, n = int(U_.shape[0]), int(V_.shape[0])

                U_ += torch.sqrt(tau) * torch.eye(m, device=self.device)
                V_ += torch.sqrt(tau) * torch.eye(n, device=self.device)

                U_ = torch.cholesky(torch.inverse(U_), upper=False)
                V_ = torch.cholesky(torch.inverse(V_), upper=True)

            module.__getattr__(f"{name}_U_half").copy_(U_)
            module.__getattr__(f"{name}_V_half").copy_(V_)

    def get_hessian(self, train_loader, binary=False):
        criterion = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
        opt = KFAC(self.net)
        U = {}
        V = {}

        # Populate parameters with the means
        self.sample(scale=0, require_grad=True)

        for batch in tqdm(train_loader):
            # x = x.cuda(non_blocking=True)
            x, y = prepare_batch_data(batch)

            self.net.zero_grad()
            out = self(x).squeeze(1)

            if binary:
                distribution = torch.distributions.Binomial(logits=out)
            else:
                distribution = torch.distributions.Categorical(logits=out)

            y = distribution.sample()
            loss = criterion(out, y.float())
            loss.backward()
            opt.step()

        with torch.no_grad():
            for group in opt.param_groups:
                if len(group["params"]) == 2:
                    weight, bias = group["params"]
                else:
                    weight = group["params"][0]
                    bias = None

                module = group["mod"]
                state = opt.state[module]

                U_ = state["ggt"]
                V_ = state["xxt"]

                n_data = len(train_loader.dataset)

                U[(module, "weight")] = sqrt(n_data) * U_
                V[(module, "weight")] = sqrt(n_data) * V_

            self.hessians = (U, V)

    def gridsearch_var0(self, val_loader, ood_loader, interval, n_classes=10, lam=1):
        vals, var0s = [], []
        pbar = tqdm(interval)

        for var0 in pbar:
            self.estimate_variance(var0)

            if n_classes == 2:
                preds_in, y_in = predict_binary(val_loader, self, 3, return_targets=True)
                preds_out = predict_binary(ood_loader, self, 3)

                loss_in = F.binary_cross_entropy(preds_in.squeeze(), y_in.float())
                loss_out = F.binary_cross_entropy(
                    preds_out.squeeze(), torch.ones_like(y_in).float() * 0.5
                )
            else:
                preds_in, y_in = predict(val_loader, self, n_samples=5, return_targets=True)
                preds_out = predict(ood_loader, self, n_samples=5)

                loss_in = F.nll_loss(torch.log(preds_in + 1e-8), y_in)
                loss_out = -torch.log(preds_out + 1e-8).mean()

            loss = loss_in + lam * loss_out

            vals.append(loss)
            var0s.append(var0)

            pbar.set_description(
                f"var0: {var0:.5f}, Loss-in: {loss_in:.3f}, Loss-out: {loss_out:.3f}, Loss: {loss:.3f}"
            )

        best_var0 = var0s[np.argmin(vals)]
        return best_var0


def kfla_parameters(module, params, device="cpu"):
    mod_class = module.__class__.__name__
    if mod_class not in ["Linear", "Conv2d"]:
        return

    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            # print(module, name)
            continue

        data = module._parameters[name].data
        m, n = int(data.shape[0]), int(np.prod(data.shape[1:]))
        module._parameters.pop(name)
        module.register_buffer(f"{name}_mean", data)
        module.register_buffer(f"{name}_U_half", torch.zeros([m, m], device=device))
        module.register_buffer(f"{name}_V_half", torch.zeros([n, n], device=device))
        module.register_buffer(name, data.new(data.size()).zero_())

        params.append((module, name))


##########################################################################
#
#  Taken with modifications from
#  https://github.com/wjmaddox/swa_gaussian/
#
##########################################################################
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer


class KFAC(Optimizer):
    def __init__(self, net, alpha=0.95):
        """K-FAC Preconditionner for Linear and Conv2d layers.

        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.

        Args:
            net (torch.nn.Module): Network to precondition.
            alpha (float): Running average parameter (if == 1, no r. ave.).
        """
        self.alpha = alpha
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0

        for mod in net.modules():
            mod_class = mod.__class__.__name__

            if mod_class in ["Linear", "Conv2d"]:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)

                handle = mod.register_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)

                params = [mod.weight]

                if mod.bias is not None:
                    params.append(mod.bias)

                d = {"params": params, "mod": mod, "layer_type": mod_class}
                self.params.append(d)

        super(KFAC, self).__init__(self.params, {})

    def step(self):
        for group in self.param_groups:
            # Getting parameters
            if len(group["params"]) == 2:
                weight, bias = group["params"]
            else:
                weight = group["params"][0]
                bias = None

            state = self.state[group["mod"]]
            self._compute_covs(group, state)

        self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        self.state[mod]["x"] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        self.state[mod]["gy"] = grad_output[0] * grad_output[0].size(0)

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        mod = group["mod"]
        x = self.state[group["mod"]]["x"]
        gy = self.state[group["mod"]]["gy"]

        # Computation of xxt
        if group["layer_type"] == "Conv2d":
            x = F.unfold(x, mod.kernel_size, padding=mod.padding, stride=mod.stride)
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()

        # if mod.bias is not None:
        #     ones = torch.ones_like(x[:1])
        #     x = torch.cat([x, ones], dim=0)

        if self._iteration_counter == 0:
            state["xxt"] = torch.mm(x, x.t()) / float(x.shape[1])
        else:
            state["xxt"].addmm_(
                mat1=x, mat2=x.t(), beta=(1.0 - self.alpha), alpha=self.alpha / float(x.shape[1])
            )

        # Computation of ggt
        if group["layer_type"] == "Conv2d":
            gy = gy.data.permute(1, 0, 2, 3)
            state["num_locations"] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state["num_locations"] = 1

        if self._iteration_counter == 0:
            state["ggt"] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        else:
            state["ggt"].addmm_(
                mat1=gy, mat2=gy.t(), beta=(1.0 - self.alpha), alpha=self.alpha / float(gy.shape[1])
            )


import torch
from tqdm import tqdm


@torch.no_grad()
def predict(test_loader, model, n_samples=20, apply_softmax=True, return_targets=False, delta=1):
    py = []
    targets = []

    for x, y in test_loader:
        x, y = delta * x.cuda(), y.cuda()
        targets.append(y)

        # MC-integral
        py_ = 0
        for _ in range(n_samples):
            out = model.forward_sample(x)
            py_ += torch.softmax(out, 1) if apply_softmax else out

        py_ /= n_samples
        py.append(py_)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0)


@torch.no_grad()
def predict_binary(test_loader, model, n_samples=100, return_targets=False, delta=1):
    py = []
    targets = []

    for x, y in test_loader:
        x, y = delta * x.cuda(), y.cuda()
        targets.append(y)

        # MC-integral
        py_ = 0
        for _ in range(n_samples):
            out = model.forward_sample(x).squeeze()
            py_ += torch.sigmoid(out)

        py_ /= n_samples
        py.append(py_)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0)


##################################################################################################
##################################################################################################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def negative_log_posterior(prior, W, predictions, target, binary=True):
    """
    input:
        prior: is the prior variance, which is to be optimized for
        W: is the model weights of the last layer
        predictions: [batch_size,n_classes,img_width,img_height]
        target: [batch_size,img_width,img_height]
        binary: Bool indicating wheter binary classification or multiclass classification
    """
    if binary:
        nll = F.binary_cross_entropy_with_logits(predictions.squeeze(1), target, reduction="sum")
    else:
        nll = F.softmax(predictions, target, reduction="sum")
    # negative log prior is given by:
    P = (1 / prior) * torch.eye(W.numel())  # log prior in diagonal
    nlp = (1 / 2) * (W.flatten() @ P @ W.flatten())
    return nll + nlp

def optimize_prior(prior_prec,W,predictions,target,binary=True,n_steps=500):
    log_prior_prec = prior_prec.log()
    log_prior_prec.requires_grad = True
    optimizer = torch.optim.Adam([log_prior_prec], lr=1e-1)
    for _ in range(n_steps):
        optimizer.zero_grad()
        prior_prec = log_prior_prec.exp()
        neg_log_marglik = -negative_log_posterior(prior=prior_prec,W=W,predictions=predictions,target=target,binary=binary)
        neg_log_marglik.backward()
        optimizer.step()
    prior_prec = log_prior_prec.detach().exp()
    return prior_prec


def compute_covariance(prior, W, predictions, target, binary=True):
    loss = negative_log_posterior(prior, W, predictions, target, binary)
    # Compute Hessian
    Hes = exact_hessian(loss, [W])
    # The Variance-Covariance Matrix is the Inverse Hessian
    Sigma = torch.inverse(Hes).detach().cpu().numpy()
    return Sigma


def prepare_batch_data(batch, out_ch=1, device="cpu", dataset="warwick"):
    images, masks, _ = batch
    images = images.unsqueeze(1) if dataset != "warwick" else images
    images = images.to(device=device, dtype=torch.float32)
    masks = masks.type(torch.LongTensor)
    if out_ch > 1:
        masks = masks.squeeze(1)
    masks = masks.to(device)
    return images, masks

