"""Configuration dataclasses for the JEPA model architecture.

All architecture hyperparameters live here so the rest of the code
remains free of magic numbers and is easily serializable.
"""

from dataclasses import dataclass


@dataclass
class EncoderConfig:
    """Configuration for the transformer context and target encoders.

    Args:
        vocab_size: Total number of tokens in the vocabulary (including specials).
        d_model: Embedding and hidden dimension throughout the transformer.
        n_heads: Number of attention heads. Must divide d_model evenly.
        n_layers: Number of transformer blocks stacked in the encoder.
        d_feedforward: Inner dimension of each block's feedforward sublayer.
        dropout: Dropout probability applied to embeddings and attention weights.
        max_seq_len: Maximum sequence length supported by positional embeddings.
        is_causal: If True, attention is masked to prevent attending to future
            positions (required for next-token prediction). If False, all
            positions attend to each other (better for masked prediction tasks).
    """

    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_feedforward: int = 512
    dropout: float = 0.1
    max_seq_len: int = 512
    is_causal: bool = False


@dataclass
class MLPPredictorConfig:
    """Configuration for the MLP-based latent predictor.

    Args:
        d_model: Input and output dimension (must match the encoder's d_model).
        n_layers: Total number of linear layers, including the output layer.
        d_hidden: Width of each hidden layer. Only used when n_layers > 1.
    """

    d_model: int
    n_layers: int = 2
    d_hidden: int = 256


@dataclass
class TransformerPredictorConfig:
    """Configuration for the shallow transformer latent predictor.

    Args:
        d_model: Input and output dimension (must match the encoder's d_model).
        n_layers: Number of transformer blocks in the predictor.
        n_heads: Number of attention heads. Must divide d_model evenly.
        d_feedforward: Inner dimension of each block's feedforward sublayer.
        dropout: Dropout probability applied within the predictor.
    """

    d_model: int
    n_layers: int = 2
    n_heads: int = 4
    d_feedforward: int = 512
    dropout: float = 0.1


@dataclass
class JEPAConfig:
    """Top-level configuration for the full JEPA model.

    Args:
        encoder: Configuration shared by both the context and target encoders.
        predictor: Configuration for the latent predictor. Use
            MLPPredictorConfig for a lightweight per-position MLP or
            TransformerPredictorConfig for a cross-position predictor.
        ema_decay: Exponential moving average decay rate for the target encoder.
            Higher values make the target encoder change more slowly.
            Typical range: 0.99 to 0.9999.
    """

    encoder: EncoderConfig
    predictor: MLPPredictorConfig | TransformerPredictorConfig
    ema_decay: float = 0.996
