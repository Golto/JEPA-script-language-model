"""Latent predictors for the JEPA architecture.

The predictor takes the context encoder's output and predicts the target
encoder's output at masked positions. It is discarded after pre-training.

Two variants are provided:
    - MLPPredictor: lightweight per-position MLP. No cross-position attention.
      Faster and sufficient for short programs.
    - TransformerPredictor: shallow transformer operating on latent vectors.
      Can model interactions between context positions when predicting targets.
"""

import torch
import torch.nn as nn

from .config import MLPPredictorConfig, TransformerPredictorConfig


class MLPPredictor(nn.Module):
    """Per-position MLP mapping context representations to target predictions.

    The same MLP is applied independently at every sequence position.
    Architecture: Linear -> GELU -> ... -> Linear (n_layers total).
    Input and output dimension are both d_model.
    """

    def __init__(self, config: MLPPredictorConfig):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = config.d_model
        for _ in range(config.n_layers - 1):
            layers.append(nn.Linear(input_dim, config.d_hidden))
            layers.append(nn.GELU())
            input_dim = config.d_hidden
        layers.append(nn.Linear(input_dim, config.d_model))
        self.net = nn.Sequential(*layers)

    def forward(self, z_context: torch.Tensor) -> torch.Tensor:
        """Map context representations to predicted target representations.

        Args:
            z_context: Context encoder output, shape (batch, seq_len, d_model).

        Returns:
            Predicted target representations, shape (batch, seq_len, d_model).
        """
        return self.net(z_context)


class TransformerPredictor(nn.Module):
    """Shallow transformer predicting target representations from context.

    Unlike the context encoder, this module has no embedding layer: it
    operates directly on the latent vectors produced by the context encoder.
    Cross-position attention allows the predictor to use global context when
    estimating each masked position.
    """

    def __init__(self, config: TransformerPredictorConfig):
        super().__init__()
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            predictor_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, z_context: torch.Tensor) -> torch.Tensor:
        """Map context representations to predicted target representations.

        Args:
            z_context: Context encoder output, shape (batch, seq_len, d_model).

        Returns:
            Predicted target representations, shape (batch, seq_len, d_model).
        """
        return self.output_norm(self.transformer(z_context))


def build_predictor(
    config: MLPPredictorConfig | TransformerPredictorConfig,
) -> MLPPredictor | TransformerPredictor:
    """Instantiate the appropriate predictor from its config type.

    Args:
        config: An MLPPredictorConfig or TransformerPredictorConfig instance.

    Returns:
        The corresponding instantiated predictor module.

    Raises:
        TypeError: If config is not a recognized predictor config type.
    """
    if isinstance(config, MLPPredictorConfig):
        return MLPPredictor(config)
    if isinstance(config, TransformerPredictorConfig):
        return TransformerPredictor(config)
    raise TypeError(f"Unrecognized predictor config type: {type(config)}")
