"""Transformer context encoder for the JEPA architecture.

Processes token sequences (with or without structural masking) and produces
contextual latent representations. Supports both bidirectional and causal
attention patterns.
"""

import torch
import torch.nn as nn

from .config import EncoderConfig


class ContextEncoder(nn.Module):
    """Transformer encoder producing contextual representations of token sequences.

    Uses pre-norm transformer blocks (LayerNorm before attention and FFN)
    for more stable training at depth. Positional information is encoded via
    learned position embeddings summed with token embeddings.

    The attention pattern is controlled by EncoderConfig.is_causal:
        - False (default): bidirectional, each token attends to all others.
          Better for masked prediction (JEPA pre-training).
        - True: causal, each token attends only to preceding positions.
          Required for next-token prediction fine-tuning.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            # NOTE: nested tensor optimization is disabled because it is
            # incompatible with key_padding_mask in some PyTorch versions.
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of token sequences into contextual representations.

        Args:
            token_ids: Integer token indices, shape (batch, seq_len).
            padding_mask: Boolean mask where True marks padding positions that
                should be ignored by attention, shape (batch, seq_len).
                Pass None when there is no padding.

        Returns:
            Contextual representations, shape (batch, seq_len, d_model).

        Raises:
            ValueError: If seq_len exceeds max_seq_len set in the config.
        """
        seq_len = token_ids.shape[1]

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Input length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.embedding_dropout(
            self.token_embedding(token_ids) + self.position_embedding(positions)
        )

        attn_mask: torch.Tensor | None = None
        if self.config.is_causal:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=token_ids.device
            )

        x = self.transformer(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        return self.output_norm(x)
