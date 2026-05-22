"""Next-token prediction model built on top of a pre-trained JEPA context encoder.

The context encoder is loaded from a JEPA checkpoint and switched from
bidirectional to causal attention. Its weights are fully compatible: only
the attention mask changes, so the pre-trained representations serve as a
strong initialization for the language modeling task.

Two training modes:
    freeze_encoder=False  -- end-to-end fine-tuning of encoder + LM head.
    freeze_encoder=True   -- linear probing: only the LM head is trained.
                             Use this to evaluate raw JEPA representation quality.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import ContextEncoder, EncoderConfig, LMHead
from src.tokenizer import LanguageTokenizer


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

@dataclass
class NTPConfig:
    """Configuration for the next-token prediction model.

    Args:
        encoder: Encoder architecture. is_causal is forced to True at model
            construction regardless of the value set here.
        freeze_encoder: If True, encoder weights are frozen and only the LM
            head trains. If False, the full model is fine-tuned end-to-end.
    """

    encoder: EncoderConfig
    freeze_encoder: bool = False


# ----------------------------------------------------------------
# Model
# ----------------------------------------------------------------

class NextTokenPredictor(nn.Module):
    """Causal language model head stacked on a pre-trained JEPA context encoder.

    At each position i, produces logits over the vocabulary predicting the
    token at position i+1. Causal masking ensures position i cannot attend
    to positions j > i, preventing information leakage during training.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: NTPConfig):
        super().__init__()

        # Force causal attention regardless of what the loaded config says.
        # The pre-trained bidirectional weights are a valid initialization
        # for a causal encoder -- only the attention mask pattern changes.
        config.encoder.is_causal = True

        self.encoder = ContextEncoder(config.encoder)
        self.lm_head = LMHead(config.encoder.d_model, config.encoder.vocab_size)

        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @classmethod
    def from_pretrained(
        cls,
        encoder_path: Path,
        freeze_encoder: bool = False,
    ) -> 'NextTokenPredictor':
        """Load encoder weights from a JEPA checkpoint and build the model.

        Reads model_config.json from the same directory as the checkpoint,
        overrides is_causal=True, then loads the encoder state dict.

        Args:
            encoder_path: Path to context_encoder_final.pt (or any encoder .pt).
                model_config.json must be in the same directory.
            freeze_encoder: If True, encoder weights are frozen after loading.

        Returns:
            A NextTokenPredictor ready for fine-tuning.

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
        config = NTPConfig(encoder=encoder_config, freeze_encoder=freeze_encoder)
        model = cls(config)

        state_dict = torch.load(encoder_path, map_location='cpu', weights_only=True)
        model.encoder.load_state_dict(state_dict)

        return model

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce next-token logits for each position in the sequence.

        Args:
            token_ids: Integer token indices, shape (batch, seq_len).
            padding_mask: Boolean mask where True marks padding positions,
                shape (batch, seq_len). Pass None when there is no padding.

        Returns:
            Logits of shape (batch, seq_len, vocab_size). The logit at
            position i predicts the token at position i+1, so the useful
            range for loss computation is logits[:, :-1, :] against
            targets token_ids[:, 1:].
        """
        hidden_states = self.encoder(token_ids, padding_mask=padding_mask)
        return self.lm_head(hidden_states)

    def compute_loss(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute cross-entropy next-token prediction loss.

        Shifts predictions and targets by one position: the representation
        at position i predicts the token at position i+1. Padding positions
        are excluded from the loss via ignore_index.

        Args:
            token_ids: Integer token indices, shape (batch, seq_len).
            padding_mask: Padding indicator, shape (batch, seq_len). Pass None
                when there is no padding.

        Returns:
            Scalar cross-entropy loss averaged over all active (non-padding) positions.
        """
        logits = self.forward(token_ids, padding_mask)
        vocab_size = logits.shape[-1]

        shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
        shift_targets = token_ids[:, 1:].contiguous().view(-1)

        return F.cross_entropy(
            shift_logits,
            shift_targets,
            ignore_index=LanguageTokenizer.PAD_ID,
        )
