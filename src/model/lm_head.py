"""Language model head: projects encoder hidden states to vocabulary logits.

Kept as a thin, separate module so it can be swapped out for a more
complex head (e.g. MLP, tied embeddings) without touching the encoder.
"""

import torch
import torch.nn as nn


class LMHead(nn.Module):
    """Linear projection from encoder hidden states to vocabulary logits.

    No bias term, following the convention in most LM implementations where
    the output projection is bias-free to reduce parameter count.

    Args:
        d_model: Input hidden dimension (must match the encoder's d_model).
        vocab_size: Number of output classes (vocabulary size).
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits.

        Args:
            hidden_states: Encoder output, shape (batch, seq_len, d_model).

        Returns:
            Unnormalized logits, shape (batch, seq_len, vocab_size).
        """
        return self.projection(hidden_states)
