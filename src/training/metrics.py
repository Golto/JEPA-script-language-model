"""Collapse detection metrics for JEPA pre-training.

Representation collapse occurs when the model learns to map all inputs to
the same (or nearly the same) point in latent space. Two complementary
signals are tracked:

    embedding_std   -- per-dimension standard deviation of pooled embeddings
                       across the batch. Drops toward 0 under collapse.

    mean_cosine_sim -- average pairwise cosine similarity of pooled embeddings.
                       Rises toward 1.0 under collapse.

Both are computed over z_target (the EMA encoder output) since it is the
stable reference. Computing them over z_context or z_hat would conflate
collapse with normal predictor behavior early in training.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_collapse_metrics(
    z_target: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute collapse indicators from a batch of target representations.

    Args:
        z_target: Target encoder output, shape (batch, seq_len, d_model).
        padding_mask: Boolean mask where True marks PAD positions,
            shape (batch, seq_len). Used to exclude padding from pooling.
            Pass None when there is no padding.

    Returns:
        Dictionary with two scalar float entries:
            'embedding_std':   mean per-dimension std across the batch.
                               Healthy range: > 0.1. Collapse: near 0.
            'mean_cosine_sim': mean pairwise cosine similarity.
                               Healthy range: < 0.9. Collapse: near 1.0.
    """
    # Pool each sample to a single vector, ignoring padding positions.
    if padding_mask is not None:
        active_positions = (~padding_mask).float().unsqueeze(-1)  # (batch, seq_len, 1)
        pooled = (z_target * active_positions).sum(dim=1)
        pooled = pooled / active_positions.sum(dim=1).clamp(min=1.0)
    else:
        pooled = z_target.mean(dim=1)  # (batch, d_model)

    # Per-dimension std across the batch (average over dimensions).
    # A healthy encoder spreads representations across all dimensions.
    embedding_std = pooled.std(dim=0).mean().item()

    # Mean pairwise cosine similarity (excluding self-similarity on diagonal).
    batch_size = pooled.shape[0]
    normalized = F.normalize(pooled, dim=-1)           # (batch, d_model)
    similarity_matrix = normalized @ normalized.T      # (batch, batch)
    off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=z_target.device)
    mean_cosine_sim = similarity_matrix[off_diagonal].mean().item()

    return {
        'embedding_std': embedding_std,
        'mean_cosine_sim': mean_cosine_sim,
    }
