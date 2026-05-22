"""Program classifier built on top of a pre-trained JEPA context encoder.

The encoder uses bidirectional attention (is_causal=False) so every position
can attend to the full sequence. Its output is mean-pooled over non-padding
positions to obtain one embedding per program, which is then projected to
class logits by a linear head.

Two training modes:
    freeze_encoder=False  -- end-to-end fine-tuning of encoder + head.
    freeze_encoder=True   -- linear probing: only the classification head
                             trains. Use this to evaluate raw JEPA quality.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import ClassificationHead, ContextEncoder, EncoderConfig


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

@dataclass
class ClassifierConfig:
    """Configuration for the program classifier model.

    Args:
        encoder: Encoder architecture config. is_causal is forced to False
            at construction since classification benefits from full-sequence
            bidirectional context.
        n_classes: Number of output classes.
        freeze_encoder: If True, encoder weights are frozen and only the
            classification head trains. If False, the full model fine-tunes.
    """

    encoder: EncoderConfig
    n_classes: int
    freeze_encoder: bool = False


# ----------------------------------------------------------------
# Model
# ----------------------------------------------------------------

class ProgramClassifier(nn.Module):
    """Bidirectional encoder with a linear classification head.

    Encodes a full program, mean-pools the token representations over all
    non-padding positions, and projects the result to n_classes logits.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ClassifierConfig):
        super().__init__()

        # Force bidirectional attention: classification uses the full sequence.
        config.encoder.is_causal = False

        self.encoder = ContextEncoder(config.encoder)
        self.head = ClassificationHead(config.encoder.d_model, config.n_classes)

        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @classmethod
    def from_pretrained(
        cls,
        encoder_path: Path,
        n_classes: int,
        freeze_encoder: bool = False,
    ) -> 'ProgramClassifier':
        """Load encoder weights from a JEPA checkpoint and build the classifier.

        Reads model_config.json from the same directory as the checkpoint,
        overrides is_causal=False, then loads the encoder state dict.

        Args:
            encoder_path: Path to context_encoder_final.pt (or any encoder .pt).
                model_config.json must be in the same directory.
            n_classes: Number of output classes.
            freeze_encoder: If True, encoder weights are frozen after loading.

        Returns:
            A ProgramClassifier ready for fine-tuning.

        Raises:
            FileNotFoundError: If the checkpoint or model_config.json is missing.
        """
        encoder_path = Path(encoder_path)
        config_path = encoder_path.with_name('model_config.json')

        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")
        if not config_path.exists():
            raise FileNotFoundError(
                f"model_config.json not found: {config_path}. "
                f"Re-run JEPA training to regenerate it."
            )

        with config_path.open(encoding='utf-8') as config_file:
            config_dict = json.load(config_file)

        encoder_config = EncoderConfig(**config_dict)
        config = ClassifierConfig(
            encoder=encoder_config,
            n_classes=n_classes,
            freeze_encoder=freeze_encoder,
        )
        model = cls(config)

        state_dict = torch.load(encoder_path, map_location='cpu', weights_only=True)
        model.encoder.load_state_dict(state_dict)

        return model

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce class logits for a batch of programs.

        Encodes the full sequence with bidirectional attention, mean-pools
        over active (non-padding) positions, and projects to n_classes logits.

        Args:
            token_ids: Integer token indices, shape (batch, seq_len).
            padding_mask: Boolean mask where True marks padding positions,
                shape (batch, seq_len). Pass None when there is no padding.

        Returns:
            Class logits, shape (batch, n_classes).
        """
        hidden = self.encoder(token_ids, padding_mask=padding_mask)  # (batch, seq_len, d_model)
        pooled = _mean_pool(hidden, padding_mask)                     # (batch, d_model)
        return self.head(pooled)

    def compute_loss(
        self,
        token_ids: torch.Tensor,
        labels: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute cross-entropy classification loss.

        Args:
            token_ids: Integer token indices, shape (batch, seq_len).
            labels: Integer class indices, shape (batch,).
            padding_mask: Padding indicator, shape (batch, seq_len).

        Returns:
            Scalar cross-entropy loss averaged over the batch.
        """
        logits = self.forward(token_ids, padding_mask)
        return F.cross_entropy(logits, labels)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _mean_pool(
    hidden: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Mean-pool hidden states over non-padding positions.

    Args:
        hidden: Encoder output, shape (batch, seq_len, d_model).
        padding_mask: True at padding positions, shape (batch, seq_len).
            Pass None when there is no padding.

    Returns:
        Pooled representation, shape (batch, d_model).
    """
    if padding_mask is not None:
        active = (~padding_mask).float().unsqueeze(-1)   # (batch, seq_len, 1)
        pooled = (hidden * active).sum(dim=1)
        pooled = pooled / active.sum(dim=1).clamp(min=1.0)
    else:
        pooled = hidden.mean(dim=1)
    return pooled
