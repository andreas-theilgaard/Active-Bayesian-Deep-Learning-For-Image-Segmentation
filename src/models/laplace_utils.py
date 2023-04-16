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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def negative_log_posterior(prior, W, predictions, target, device, binary=True):
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
    diag = torch.eye(W.numel()).to(device=device)
    P = (1 / prior) * diag  # log prior in diagonal
    nlp = (1 / 2) * (W.flatten() @ P @ W.flatten())
    return nll.cpu() + nlp


def log_marginal_likelihood(prior, W, device):
    # prior.requires_grad = True
    diag = torch.eye(W.numel()).to(device=device)
    P = (1 / prior) * diag  # log prior in diagonal
    nlp = -(1 / 2) * (W.flatten() @ P @ W.flatten())
    return nlp


def optimize_prior(prior_prec, W, device, n_steps=100):
    log_prior_prec = prior_prec.log()
    log_prior_prec.requires_grad = True
    optimizer = torch.optim.Adam([log_prior_prec], lr=1e-1)
    for _ in range(n_steps):
        optimizer.zero_grad()
        prior_prec = log_prior_prec.exp()
        neg_log_marglik = log_marginal_likelihood(
            prior=prior_prec, W=W, device=device
        )  # nll is constant at MAP
        neg_log_marglik.backward()
        optimizer.step()

    prior_prec = log_prior_prec.detach().exp()
    return prior_prec


def compute_covariance(prior, W, predictions, target, device, binary=True):
    loss = negative_log_posterior(prior, W, predictions, target, device, binary)
    # Compute Hessian
    Hes = exact_hessian(loss, [W])
    # The Variance-Covariance Matrix is the Inverse Hessian
    Sigma = torch.inverse(Hes).detach().cpu().numpy()
    return Sigma
