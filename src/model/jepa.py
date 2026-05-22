"""Top-level JEPA model assembling all components.

Wires the context encoder, target encoder, and predictor into a single
module with a clear forward interface. The training loop only needs to call
forward() and update_target_encoder() -- everything else is encapsulated.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import JEPAConfig
from .context_encoder import ContextEncoder
from .predictor import MLPPredictor, TransformerPredictor, build_predictor
from .target_encoder import TargetEncoder


@dataclass
class JEPAOutput:
    """Output tensors from a single JEPA forward pass.

    All three tensors share the same shape (batch, seq_len, d_model).
    To compute the JEPA loss over masked positions:

        loss = F.mse_loss(output.z_hat[structural_mask], output.z_target[structural_mask])

    where structural_mask is a boolean tensor of shape (batch, seq_len).
    """

    z_context: torch.Tensor
    z_hat: torch.Tensor
    z_target: torch.Tensor


class JEPAModel(nn.Module):
    """Joint Embedding Predictive Architecture for the embedding language.

    Training flow:
        1. context_encoder encodes the sequence with masked positions replaced
           by a learned mask embedding -> z_context.
        2. predictor maps z_context to predicted latent targets -> z_hat.
        3. target_encoder encodes the original unmasked sequence -> z_target.
        4. Loss = MSE(z_hat, z_target) at structurally masked positions.
        5. After each optimizer step, call update_target_encoder() to advance
           the EMA. The target encoder is never touched by the optimizer.

    After pre-training, only context_encoder is kept. predictor and
    target_encoder are discarded.
    """

    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config
        self.context_encoder = ContextEncoder(config.encoder)
        self.target_encoder = TargetEncoder(self.context_encoder, config.ema_decay)
        self.predictor: MLPPredictor | TransformerPredictor = build_predictor(
            config.predictor
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        structural_mask: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> JEPAOutput:
        """Run a full JEPA forward pass.

        Args:
            token_ids: Original unmasked token ids, shape (batch, seq_len).
            structural_mask: Boolean mask where True marks positions that the
                context encoder sees as the learned mask embedding,
                shape (batch, seq_len).
            padding_mask: Boolean mask where True marks PAD positions that
                attention should ignore, shape (batch, seq_len).
                Pass None when sequences are not padded.

        Returns:
            JEPAOutput containing z_context, z_hat, and z_target.
        """
        z_context = self.context_encoder(
            token_ids, token_mask=structural_mask, padding_mask=padding_mask
        )
        z_hat = self.predictor(z_context)
        z_target = self.target_encoder(token_ids, padding_mask=padding_mask)
        return JEPAOutput(z_context=z_context, z_hat=z_hat, z_target=z_target)

    def update_target_encoder(self) -> None:
        """Advance the target encoder by one EMA step.

        Must be called after every optimizer step during pre-training.
        Has no effect on gradients or the optimizer state.
        """
        self.target_encoder.update_from(self.context_encoder)
