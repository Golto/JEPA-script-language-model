"""Shared helpers used by all CLI subcommands.

Provides source resolution (inline string vs file), model loading, and
the core embed_program routine so that embed and compare stay thin.
"""

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.model import ContextEncoder, EncoderConfig
from src.tokenizer import LanguageTokenizer

if TYPE_CHECKING:
    from src.tasks.classification.model import ProgramClassifier
    from src.tasks.next_token.model import NextTokenPredictor


# ----------------------------------------------------------------
# Source resolution
# ----------------------------------------------------------------

def resolve_sources(inline_sources: list[str], file_paths: list[str]) -> list[str]:
    """Combine inline strings and file contents into an ordered list of program texts.

    Inline strings come first, then file contents in the order they were given.
    This convention is documented in the CLI help so callers know the order.

    Args:
        inline_sources: Program texts provided directly on the command line.
        file_paths: Paths to files whose contents are read as program texts.

    Returns:
        List of program text strings, inline before file-sourced.

    Raises:
        SystemExit: If any file path does not exist or cannot be read.
    """
    texts: list[str] = list(inline_sources)

    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            sys.exit(f"Error: file not found: {path_str}")
        texts.append(path.read_text(encoding='utf-8'))

    return texts


# ----------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------

def load_context_encoder(model_path: str) -> tuple[ContextEncoder, LanguageTokenizer]:
    """Load a ContextEncoder from a .pt file and its sibling model_config.json.

    The config JSON is saved automatically by the training pipeline next to
    context_encoder_final.pt. If it is missing, re-run training to regenerate it.

    Args:
        model_path: Path to the context encoder .pt file.

    Returns:
        A tuple of (encoder in eval mode, tokenizer).

    Raises:
        SystemExit: If either the .pt file or model_config.json is missing.
    """
    pt_path = Path(model_path)
    config_path = pt_path.with_name('model_config.json')

    if not pt_path.exists():
        sys.exit(f"Error: model file not found: {model_path}")
    if not config_path.exists():
        sys.exit(
            f"Error: config not found: {config_path}\n"
            f"Re-run training to regenerate model_config.json alongside the checkpoint."
        )

    with config_path.open(encoding='utf-8') as config_file:
        config_dict = json.load(config_file)

    encoder_config = EncoderConfig(**config_dict)
    encoder = ContextEncoder(encoder_config)
    state_dict = torch.load(pt_path, map_location='cpu', weights_only=True)
    encoder.load_state_dict(state_dict)
    encoder.eval()

    return encoder, LanguageTokenizer()


# ----------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------

def load_next_token_predictor(model_path: str) -> tuple['NextTokenPredictor', LanguageTokenizer]:
    """Load a NextTokenPredictor from a .pt file and its sibling model_config.json.

    model_config.json is written automatically by the fine-tuning pipeline
    next to next_token_predictor_final.pt. If it is missing, re-run fine-tuning
    to regenerate it.

    Args:
        model_path: Path to next_token_predictor_final.pt.

    Returns:
        A tuple of (predictor in eval mode, tokenizer).

    Raises:
        SystemExit: If either the .pt file or model_config.json is missing.
    """
    from src.tasks.next_token.model import NextTokenPredictor, NTPConfig

    pt_path = Path(model_path)
    config_path = pt_path.with_name('model_config.json')

    if not pt_path.exists():
        sys.exit(f"Error: model file not found: {model_path}")
    if not config_path.exists():
        sys.exit(
            f"Error: model_config.json not found next to {pt_path.name}.\n"
            f"Re-run fine-tuning to regenerate it alongside the checkpoint."
        )

    with config_path.open(encoding='utf-8') as config_file:
        config_dict = json.load(config_file)

    encoder_config = EncoderConfig(**config_dict)
    config = NTPConfig(encoder=encoder_config)
    model = NextTokenPredictor(config)
    state_dict = torch.load(pt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model, LanguageTokenizer()


def load_program_classifier(
    model_path: str,
) -> tuple['ProgramClassifier', LanguageTokenizer, dict[str, int]]:
    """Load a ProgramClassifier from a .pt file and its sibling JSON files.

    Expects model_config.json and label_vocab.json in the same directory as
    the .pt file. Both are written automatically by the classification
    fine-tuning pipeline.

    Args:
        model_path: Path to classifier_final.pt.

    Returns:
        A tuple of (classifier in eval mode, tokenizer, label_vocab).
        label_vocab maps label strings to integer class indices.

    Raises:
        SystemExit: If the .pt file, model_config.json, or label_vocab.json
            is missing.
    """
    from src.tasks.classification.model import ClassifierConfig, ProgramClassifier

    pt_path = Path(model_path)
    config_path = pt_path.with_name('model_config.json')
    vocab_path = pt_path.with_name('label_vocab.json')

    if not pt_path.exists():
        sys.exit(f"Error: model file not found: {model_path}")
    if not config_path.exists():
        sys.exit(f"Error: model_config.json not found next to {pt_path.name}.")
    if not vocab_path.exists():
        sys.exit(f"Error: label_vocab.json not found next to {pt_path.name}.")

    with config_path.open(encoding='utf-8') as config_file:
        config_dict = json.load(config_file)
    with vocab_path.open(encoding='utf-8') as vocab_file:
        label_vocab: dict[str, int] = json.load(vocab_file)

    encoder_config = EncoderConfig(**config_dict)
    config = ClassifierConfig(encoder=encoder_config, n_classes=len(label_vocab))
    model = ProgramClassifier(config)
    state_dict = torch.load(pt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model, LanguageTokenizer(), label_vocab


def embed_program(
    encoder: ContextEncoder,
    tokenizer: LanguageTokenizer,
    source: str,
) -> tuple[torch.Tensor, int]:
    """Encode a program source into a mean-pooled embedding vector.

    Tokenizes the source, runs it through the context encoder without any
    structural masking, and mean-pools over all non-padding positions.

    Args:
        encoder: A ContextEncoder in eval mode (no gradient tracking).
        tokenizer: Tokenizer matching the encoder's vocabulary.
        source: Raw program source text. Leading/trailing whitespace is stripped.

    Returns:
        A tuple of (embedding vector of shape (d_model,), number of tokens).
    """
    token_ids = tokenizer.encode(source.strip(), add_special_tokens=True)
    token_ids_tensor = torch.tensor([token_ids], dtype=torch.long)
    padding_mask = torch.zeros(1, len(token_ids), dtype=torch.bool)

    with torch.no_grad():
        representations = encoder(token_ids_tensor, padding_mask=padding_mask)

    embedding = representations[0].mean(dim=0)
    return embedding, len(token_ids)
