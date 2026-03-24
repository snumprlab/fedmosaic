from federated_methods.fedavg import LLaVATrainerFEDAVG

import contextlib
import copy
import functools
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import RandomSampler
from packaging import version
from torch import nn
from utils.train_utils import load_deepspeed
from models.llava.llava_trainer import LLaVATrainer
from transformers.utils import logging
import sys, os, time, shutil, datetime
import math
from typing import Optional, Dict, Union, Any
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.trainer_pt_utils import get_model_param_count, get_dataloader_sampler, reissue_pt_warnings
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers import Trainer
import bitsandbytes
from transformers.trainer import (
    is_sagemaker_mp_enabled, 
    _is_peft_model, 
    TRAINER_STATE_NAME,
    is_torch_xla_available,
    is_accelerate_available,
    is_deepspeed_available,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    SCHEDULER_NAME
)
import warnings
from transformers.integrations import hp_params
from transformers.trainer_callback import TrainerState, ExportableState
from transformers.training_args import ParallelMode

if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedType
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]
    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

logger = logging.get_logger(__name__)

from eval_VLM_CL import anytime_evaluation

def cka_create_trainer(model, tokenizer, training_args, data_module, model2):
    training_args.max_seq_length = training_args.model_max_length
    training_args.packing=False
    trainer = CKA_Feat_Extract(model=model,
        tokenizer=tokenizer,
        args=training_args,
        client_id = 0,
        curr_round = 0,
        test_datalist=None,
        processor=None,
        data_args=None,
        task_vector=None,
        fisher_old=None,
        fisher_freq= 5,
        model2=model2,
        **data_module,
        )
    return trainer


class CKA_Feat_Extract(LLaVATrainerFEDAVG):
    def __init__(self, client_id, curr_round, test_datalist, processor, data_args, task_vector=None, fisher_old=None, fisher_freq=5,model2=None, **kwargs):
        super(CKA_Feat_Extract, self).__init__(client_id=client_id,curr_round=curr_round,test_datalist=test_datalist,processor=processor,data_args=data_args,
                                                task_vector=task_vector,fisher_old=fisher_old,fisher_freq=fisher_freq,model2=model2,
                                                **kwargs)
        for hook in self.hooks:
            hook.remove()
        
        self.hidden_feat_1b = []
        self.hidden_feat_3b = []
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs['output_hidden_states'] = True
        # model: 3b
        # model2: 1b
        with torch.no_grad():
            loss2, outputs2 = super(CKA_Feat_Extract, self).compute_loss(self.model2, copy.deepcopy(inputs), return_outputs=True, num_items_in_batch=num_items_in_batch)
        
        loss, outputs = super(CKA_Feat_Extract, self).compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        
        labels = inputs['labels']
        shift_labels = labels[..., 1:].contiguous()
        print((shift_labels != -100).sum())
        if len(self.hidden_feat_1b) == 0:
            for feat in outputs.hidden_states:
                self.hidden_feat_3b.append(feat[..., :-1, :][shift_labels != -100].detach().cpu())
            for feat in outputs2.hidden_states:
                self.hidden_feat_1b.append(feat[..., :-1, :][shift_labels != -100].detach().cpu())
        else:
            for i,feat in enumerate(outputs.hidden_states):
                self.hidden_feat_3b[i] = torch.cat((self.hidden_feat_3b[i], feat[..., :-1, :][shift_labels != -100].detach().cpu()))
            for i,feat in enumerate(outputs2.hidden_states):
                self.hidden_feat_1b[i] = torch.cat((self.hidden_feat_1b[i], feat[..., :-1, :][shift_labels != -100].detach().cpu()))
        loss = loss*0
        return (loss, outputs) if return_outputs else loss
    


from typing import Literal

"""
Code from https://github.com/RistoAle97/centered-kernel-alignment/blob/main/src/ckatorch/core.py
Module that implements both base and mini-batch CKA.
"""

"""Module for computing HSIC (Hilbert-Schmidt Independence Criterion), both its standard and mini-batch versions."""

def hsic0(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """Compute the Hilbert-Schmidt Independence Criterion on two given Gram matrices.

    Args:
        gram_x: Gram matrix of shape (n, n), this is equivalent to K from the original paper.
        gram_y: Gram matrix of shape (n, n), this is equivalent to L from the original paper.

    Returns:
        a tensor with the Hilbert-Schmidt Independence Criterion values.

    Raises:
        ValueError: if ``gram_x`` and ``gram_y`` are not symmetric.
    """
    if not torch.allclose(gram_x, gram_x.T) and not torch.allclose(gram_y, gram_y.T):
        raise ValueError("The given matrices must be symmetric.")

    # Build the identity matrix
    n = gram_x.shape[0]
    identity = torch.eye(n, n, dtype=gram_x.dtype, device=gram_x.device)

    # Build the centering matrix
    h = identity - torch.ones(n, n, dtype=gram_x.dtype, device=gram_x.device) / n

    # Compute k * h and l * h
    kh = torch.mm(gram_x, h)
    lh = torch.mm(gram_y, h)

    # Compute the trace of the product kh * lh
    trace = torch.trace(kh.mm(lh))
    return trace / (n - 1) ** 2


def hsic1(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """Compute the batched version of the Hilbert-Schmidt Independence Criterion on Gram matrices.

    This version is based on
    https://github.com/numpee/CKA.pytorch/blob/07874ec7e219ad29a29ee8d5ebdada0e1156cf9f/cka.py#L107.

    Args:
        gram_x: batch of Gram matrices of shape (bsz, n, n).
        gram_y: batch of Gram matrices of shape (bsz, n, n).

    Returns:
        a tensor with the unbiased Hilbert-Schmidt Independence Criterion values.

    Raises:
        ValueError: if ``gram_x`` and ``gram_y`` do not have the same shape or if they do not have exactly three
        dimensions.
    """
    if len(gram_x.size()) != 3 or gram_x.size() != gram_y.size():
        raise ValueError("Invalid size for one of the two input tensors.")

    n = gram_x.shape[-1]
    gram_x = gram_x.clone()
    gram_y = gram_y.clone()

    # Fill the diagonal of each matrix with 0
    gram_x.diagonal(dim1=-1, dim2=-2).fill_(0)
    gram_y.diagonal(dim1=-1, dim2=-2).fill_(0)

    # Compute the product between k (i.e.: gram_x) and l (i.e.: gram_y)
    kl = torch.bmm(gram_x, gram_y)

    # Compute the trace (sum of the elements on the diagonal) of the previous product, i.e.: the left term
    trace_kl = kl.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)

    # Compute the middle term
    middle_term = gram_x.sum((-1, -2), keepdim=True) * gram_y.sum((-1, -2), keepdim=True)
    middle_term /= (n - 1) * (n - 2)

    # Compute the right term
    right_term = kl.sum((-1, -2), keepdim=True)
    right_term *= 2 / (n - 2)

    # Put all together to compute the main term
    main_term = trace_kl + middle_term - right_term

    # Compute the hsic values
    out = main_term / (n**2 - 3 * n)
    return out.squeeze(-1).squeeze(-1)

"""Utilities for computing and centering Gram matrices."""
def linear_kernel(x: torch.Tensor) -> torch.Tensor:
    """Computes the Gram (kernel) matrix for a linear kernel.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        x: tensor of shape (n, m).

    Returns:
        a Gram matrix which is a tensor of shape (n, n).
    """
    return torch.mm(x, x.T)


def rbf_kernel(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Computes the Gram (kernel) matrix for an RBF kernel.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        x: tensor of shape (n, m).
        threshold: fraction of median Euclidean distance to use as RBF kernel bandwidth (default=1.0).

    Returns:
        a Gram matrix which is a tensor of shape (n, n).
    """
    dot_products = torch.mm(x, x.T)
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = torch.median(sq_distances)
    return torch.exp(-sq_distances / (2 * threshold**2 * sq_median_distance))


def center_gram_matrix(gram_matrix: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    """Centers a given Gram matrix.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        gram_matrix: tensor of shape (n, n).
        unbiased: whether to use the unbiased version of the centering (default=False).

    Returns:
        the centered version of the given Gram matrix.
    """
    if not torch.allclose(gram_matrix, gram_matrix.T):
        raise ValueError("The given matrix must be symmetric.")

    gram_matrix = gram_matrix.detach().clone()
    if unbiased:
        n = gram_matrix.shape[0]
        gram_matrix.fill_diagonal_(0)
        means = torch.sum(gram_matrix, dim=0, dtype=torch.float64) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram_matrix -= means[:, None]
        gram_matrix -= means[None, :]
        gram_matrix.fill_diagonal_(0)
    else:
        means = torch.mean(gram_matrix, dim=0, dtype=torch.float64)
        means -= torch.mean(means) / 2
        gram_matrix -= means[:, None]
        gram_matrix -= means[None, :]

    return gram_matrix


def cka_base(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: Literal["linear", "rbf"] = "linear",
    unbiased: bool = False,
    threshold: float = 1.0,
    method: Literal["fro_norm", "hsic"] = "fro_norm",
) -> torch.Tensor:
    """Computes the Centered Kernel Alignment (CKA) between two given matrices.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        x: tensor of shape (n, j).
        y: tensor of shape (n, k).
        kernel: the kernel used to compute the Gram matrices, must be "linear" or "rbf" (default="linear).
        unbiased: whether to use the unbiased version of CKA (default=False).
        threshold: the threshold used by the RBF kernel (default=1.0).
        method: the method used to compute the CKA value, must be "fro_norm" (Frobenius norm) or "hsic"
            (Hilbert-Schmidt Independence Criterion). Note that the choice does not influence the output
            (default="fro_norm").

    Returns:
        a float tensor in [0, 1] that is the CKA value between the two given matrices.

    Raises:
        ValueError: if ``kernel`` is not "linear" or "rbf" or if ``method`` is not "fro_norm" or "hsic".
    """
    if kernel not in ["linear", "rbf"]:
        raise ValueError("The chosen kernel must be either 'linear' or 'rbf'.")

    if method not in ["hsic", "fro_norm"]:
        raise ValueError("The chosen method must be either 'hsic' or 'fro_norm'.")

    x = x.type(torch.float64) if not x.dtype == torch.float64 else x
    y = y.type(torch.float64) if not y.dtype == torch.float64 else y

    # Build the Gram matrices by applying the kernel
    gram_x = linear_kernel(x) if kernel == "linear" else rbf_kernel(x, threshold)
    gram_y = linear_kernel(y) if kernel == "linear" else rbf_kernel(y, threshold)

    # Compute CKA by either using HSIC or the Frobenius norm
    if method == "hsic":
        hsic_xy = hsic0(gram_x, gram_y)
        hsic_xx = hsic0(gram_x, gram_x)
        hsic_yy = hsic0(gram_y, gram_y)
        cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
    else:
        gram_x = center_gram_matrix(gram_x, unbiased)
        gram_y = center_gram_matrix(gram_y, unbiased)
        norm_xy = gram_x.ravel().dot(gram_y.ravel())
        norm_xx = torch.linalg.norm(gram_x, ord="fro")
        norm_yy = torch.linalg.norm(gram_y, ord="fro")
        cka = norm_xy / (norm_xx * norm_yy)

    return cka


def cka_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the minibatch version of CKA from Nguyen et al. (https://arxiv.org/abs/2010.15327).

    This computation is performed with linear kernel and by calculating HSIC_1.

    Args:
        x: tensor of shape (bsz, n, j).
        y: tensor of shape (bsz, n, k).

    Returns:
        a float tensor in [0, 1] that is the CKA value between the two given tensors.
    """
    x = x.type(torch.float64) if not x.dtype == torch.float64 else x
    y = y.type(torch.float64) if not y.dtype == torch.float64 else y

    # Build the Gram matrices by applying the linear kernel
    gram_x = torch.bmm(x, x.transpose(1, 2))
    gram_y = torch.bmm(y, y.transpose(1, 2))

    # Compute the HSIC values for the entire batches
    hsic1_xy = hsic1(gram_x, gram_y)
    hsic1_xx = hsic1(gram_x, gram_x)
    hsic1_yy = hsic1(gram_y, gram_y)

    # Compute the CKA value
    cka = hsic1_xy.sum() / (hsic1_xx.sum() * hsic1_yy.sum()).sqrt()
    return cka


import numpy as np
from sklearn.cross_decomposition import CCA

def compute_pwcca(X, Y, n_components=None):
    """
    Compute Projection-Weighted CCA (PWCCA) between two datasets X and Y.

    Args:
        X (numpy.ndarray): Shape (num_samples, feature_dim_X).
        Y (numpy.ndarray): Shape (num_samples, feature_dim_Y).
        n_components (int, optional): Number of canonical dimensions to use.
            Defaults to min(X.shape[1], Y.shape[1]).

    Returns:
        pwcca_value (float): The PWCCA value (scalar).
        correlations (np.ndarray): Per-dimension canonical correlations (length = n_components).
    """
    # Defaults to the smaller feature dimension
    if n_components is None:
        n_components = min(X.shape[1], Y.shape[1])
    
    # 1. Center the data
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    
    # 2. Fit CCA
    cca = CCA(n_components=n_components)
    cca.fit(Xc, Yc)
    Xc_cca, Yc_cca = cca.transform(Xc, Yc)  # shape: (num_samples, n_components)
    
    # 3. Compute per-dimension correlation
    correlations = []
    for i in range(n_components):
        corr_matrix = np.corrcoef(Xc_cca[:, i], Yc_cca[:, i])
        corr = corr_matrix[0, 1]
        correlations.append(corr)
    correlations = np.array(correlations)
    
    # 4. Compute projection-based weights alpha_i
    #    Following Morcos et al., "Insights on representational similarity...", alpha is 
    #    based on how strongly X projects onto each canonical dimension.
    #    We can obtain the canonical directions in X-space from `cca.x_loadings_` (shape: [feature_dim_X, n_components]).
    
    comps = cca.x_loadings_        # (feature_dim_X, n_components)
    # Project the centered data Xc onto these components to see how "important" each component is
    X_projections = Xc @ comps     # (num_samples, n_components)
    # Sum of absolute values (or sums of squares) is often used; the official references use sum of absolute projections.
    alpha = np.sum(np.abs(X_projections), axis=0)
    alpha /= alpha.sum()  # normalize so that sum(alpha) = 1
    
    # 5. Weighted sum of correlations = PWCCA
    pwcca_value = np.sum(alpha * correlations)
    
    return pwcca_value, correlations

from scipy.stats import spearmanr

def compute_rdm(activations, metric='correlation'):
    """
    Given activations of shape (N, D), compute the RDM as an (N, N) matrix,
    where entry (i, j) is the dissimilarity between sample i and sample j.

    Supported metrics:
      - 'euclidean': Uses torch.cdist (L2 distance)
      - 'correlation': Uses correlation distance = 1 - Pearson correlation
    """
    if metric == 'euclidean':
        # Euclidean distance between all pairs
        rdm = torch.cdist(activations, activations, p=2)

    elif metric == 'correlation':
        # 1) Center each row
        X_centered = activations - activations.mean(dim=1, keepdim=True)
        
        # 2) Normalize each row to have unit standard deviation
        X_std = X_centered.std(dim=1, keepdim=True) + 1e-8
        X_normalized = X_centered / X_std  # shape: (N, D)
        
        # 3) Compute similarity = dot product = Pearson correlation
        #    (since each row is zero-mean and unit-variance)
        similarity = X_normalized @ X_normalized.t()  # shape: (N, N)
        
        # 4) Convert similarity to dissimilarity
        #    correlation distance = 1 - correlation
        rdm = 1 - similarity

    else:
        raise NotImplementedError(f"Metric '{metric}' not implemented.")

    return rdm

def rdm_similarity(rdm1, rdm2, method='spearman'):
    """
    Compare two RDMs via correlation.

    Args:
        rdm1 (torch.Tensor): RDM of shape (N, N)
        rdm2 (torch.Tensor): RDM of shape (N, N)
        method (str): 'spearman' or 'pearson'

    Returns:
        float: The correlation (similarity) between rdm1 and rdm2.
    """
    # Extract upper triangular entries (excluding diagonal)
    triu_indices = torch.triu_indices(rdm1.size(0), rdm1.size(1), offset=1)
    rdm1_flat = rdm1[triu_indices[0], triu_indices[1]]
    rdm2_flat = rdm2[triu_indices[0], triu_indices[1]]

    # Convert to numpy
    rdm1_flat_np = rdm1_flat.cpu().numpy()
    rdm2_flat_np = rdm2_flat.cpu().numpy()

    if method == 'spearman':
        corr, _ = spearmanr(rdm1_flat_np, rdm2_flat_np)
    elif method == 'pearson':
        corr = np.corrcoef(rdm1_flat_np, rdm2_flat_np)[0, 1]
    else:
        raise ValueError("method must be 'spearman' or 'pearson'")
    return corr