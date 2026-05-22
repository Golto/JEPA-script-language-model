"""Linear classification head for sequence-level program classification."""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Linear projection from a pooled embedding to class logits.

    Args:
        d_model: Dimensionality of the input embedding.
        n_classes: Number of output classes.
    """

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, n_classes, bias=True)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Compute class logits from a pooled sequence embedding.

        Args:
            pooled: Pooled representation, shape (batch, d_model).

        Returns:
            Class logits, shape (batch, n_classes).
        """
        return self.classifier(pooled)
