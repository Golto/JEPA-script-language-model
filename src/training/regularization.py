"""VICReg-style regularization losses to prevent representation collapse.

MSE alone (z_hat vs z_target) does not prevent collapse: the loss decreases
just as well if all programs collapse to the same point in latent space.
VICReg adds two explicit anti-collapse terms applied to the context encoder
output (z_context), which carries gradients:

    Variance term   -- penalizes any latent dimension whose per-batch std
                       falls below 1. Forces the encoder to use its full
                       representational capacity.

    Covariance term -- penalizes off-diagonal entries of the sample covariance
                       matrix. Decorrelates dimensions so information is spread
                       across the whole embedding rather than collapsed into a
                       few correlated directions.

NOTE: Both terms are applied to z_context (the context encoder output) and NOT
to z_target. z_target is produced by the EMA encoder which has no gradients;
regularizing it via a loss would have no effect. Regularizing z_context is
sufficient because z_target is an EMA of z_context weights -- preventing the
context encoder from collapsing indirectly stabilizes the target encoder.

Reference: Bardes et al. "VICReg: Variance-Invariance-Covariance Regularization
for Self-Supervised Learning", ICLR 2022.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

@dataclass
class VICRegConfig:
    """Weights for the two VICReg regularization terms.

    Args:
        lambda_var: Weight on the variance loss. The variance loss is in
            [0, 1] (ReLU-clamped), so lambda_var=1.0 makes it comparable in
            scale to a typical MSE loss around 0.5-1.0. Increase if collapse
            persists; decrease if the predictor stops learning.
        lambda_cov: Weight on the covariance loss. Covariance loss values tend
            to be smaller, so a lower default is appropriate.
    """

    lambda_var: float = 1.0
    lambda_cov: float = 0.1


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _mean_pool(z: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool a sequence tensor over non-padding positions.

    Args:
        z: Sequence of representations, shape (batch, seq_len, d_model).
        padding_mask: Boolean mask where True marks padding positions,
            shape (batch, seq_len). Pass None when there is no padding.

    Returns:
        Mean-pooled tensor of shape (batch, d_model).
    """
    if padding_mask is not None:
        active = (~padding_mask).float().unsqueeze(-1)      # (batch, seq_len, 1)
        pooled = (z * active).sum(dim=1)
        pooled = pooled / active.sum(dim=1).clamp(min=1.0)  # (batch, d_model)
    else:
        pooled = z.mean(dim=1)                              # (batch, d_model)

    return pooled


# ----------------------------------------------------------------
# VICReg loss
# ----------------------------------------------------------------

@dataclass
class VICRegLoss:
    """Decomposed VICReg regularization loss.

    Args:
        total: Weighted sum of variance and covariance losses (backprop target).
        variance: Raw variance loss before weighting.
        covariance: Raw covariance loss before weighting.
    """

    total: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor


def compute_vicreg_loss(
    z_context: torch.Tensor,
    config: VICRegConfig,
    padding_mask: torch.Tensor | None = None,
) -> VICRegLoss:
    """Compute VICReg variance and covariance regularization on context representations.

    Mean-pools z_context over non-padding positions to obtain one vector per
    program, then computes variance and covariance losses over the batch.

    Args:
        z_context: Context encoder output, shape (batch, seq_len, d_model).
            Must carry gradients (do not detach before calling).
        config: Regularization weights.
        padding_mask: Boolean mask where True marks padding positions,
            shape (batch, seq_len). Pass None when there is no padding.

    Returns:
        VICRegLoss with total, variance, and covariance components.
    """
    pooled = _mean_pool(z_context, padding_mask)    # (batch, d_model)
    batch_size, d_model = pooled.shape

    # ----------------------------------------------------------------
    # Variance term
    # ----------------------------------------------------------------
    # Per-dimension std across the batch. Penalize if below 1.
    # relu(1 - std) is in [0, 1]: zero when std >= 1, rises as std falls.
    per_dim_std = pooled.std(dim=0)                 # (d_model,)
    variance_loss = F.relu(1.0 - per_dim_std).mean()

    # ----------------------------------------------------------------
    # Covariance term
    # ----------------------------------------------------------------
    # Off-diagonal entries of the normalized sample covariance matrix.
    # Non-zero off-diagonal entries mean dimensions carry redundant information.
    z_centered = pooled - pooled.mean(dim=0)        # (batch, d_model)
    cov = (z_centered.T @ z_centered) / (batch_size - 1)  # (d_model, d_model)

    diagonal = torch.eye(d_model, dtype=torch.bool, device=z_context.device)
    off_diagonal_cov = cov.masked_fill(diagonal, 0.0)
    covariance_loss = off_diagonal_cov.pow(2).sum() / d_model

    total = config.lambda_var * variance_loss + config.lambda_cov * covariance_loss

    return VICRegLoss(total=total, variance=variance_loss, covariance=covariance_loss)
