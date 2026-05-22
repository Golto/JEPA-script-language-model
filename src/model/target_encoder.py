"""EMA target encoder for the JEPA architecture.

The target encoder is a frozen copy of the context encoder whose weights are
updated exclusively via exponential moving average. This asymmetry between the
two encoders is what prevents representation collapse during JEPA pre-training.
"""

import copy

import torch
import torch.nn as nn

from .context_encoder import ContextEncoder


class TargetEncoder(nn.Module):
    """Stable prediction target maintained as an EMA of the context encoder.

    At initialization, the target encoder is an exact copy of the context
    encoder. After each optimizer step, call update_from() to apply the EMA:

        theta_target = decay * theta_target + (1 - decay) * theta_context

    Parameters are frozen: no gradient ever flows through this module.
    The forward pass runs under torch.no_grad() for efficiency.
    """

    def __init__(self, context_encoder: ContextEncoder, ema_decay: float = 0.996):
        """Initialize from a context encoder.

        Args:
            context_encoder: The encoder to copy. The target encoder starts
                as an exact replica of this module.
            ema_decay: EMA coefficient. Closer to 1.0 means slower updates
                (more stable target). Typical range: 0.99 to 0.9999.
        """
        super().__init__()
        self.ema_decay = ema_decay
        self._encoder = copy.deepcopy(context_encoder)
        for param in self._encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_from(self, context_encoder: ContextEncoder) -> None:
        """Apply one EMA step, pulling weights toward the context encoder.

        Must be called after every optimizer step during pre-training.

        Args:
            context_encoder: The context encoder whose current weights are
                used as the EMA source.
        """
        for target_param, source_param in zip(
            self._encoder.parameters(), context_encoder.parameters()
        ):
            target_param.data.mul_(self.ema_decay).add_(
                source_param.data, alpha=1.0 - self.ema_decay
            )

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode token sequences without gradient tracking.

        Args:
            token_ids: Integer token indices, shape (batch, seq_len).
            padding_mask: Boolean padding mask, shape (batch, seq_len).
                Pass None when there is no padding.

        Returns:
            Target representations, shape (batch, seq_len, d_model).
        """
        with torch.no_grad():
            return self._encoder(token_ids, padding_mask)
