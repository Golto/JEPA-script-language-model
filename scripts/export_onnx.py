"""Export a trained model to ONNX for use in Next.js (onnxruntime-web / onnxruntime-node).

Supported tasks:
    next-token  — export NextTokenPredictor (causal LM, default)
    embed       — export ContextEncoder only (bidirectional, mean-pooled output)

Outputs written to --output directory:
    model.onnx          ONNX graph with dynamic batch and sequence-length axes
    vocab.json          Full token↔id mapping + special token ids for the JS tokenizer
    tokenizer.json      Tokenizer rules metadata (number encoding, keyword list …)
    model_config.json   Copy of the encoder architecture config

Usage:
    uv run scripts/export_onnx.py \\
        --model .private/output/next_token/next_token_final.pt \\
        --output .private/output/onnx/next_token

    uv run scripts/export_onnx.py \\
        --task embed \\
        --model .private/output/jepa/context_encoder_final.pt \\
        --output .private/output/onnx/embed

    # Include onnxruntime verification pass
    uv run scripts/export_onnx.py --model ... --output ... --verify
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Ensure the project root (parent of scripts/) is on sys.path so that
# `import src.*` works regardless of how the script is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn


# ----------------------------------------------------------------
# ONNX wrapper modules
# ----------------------------------------------------------------

class _NTPWrapper(nn.Module):
    """Wraps NextTokenPredictor for clean ONNX tracing.

    Removes the optional padding_mask argument (not needed for single-sequence
    inference) and fixes the causal mask path so the ONNX graph is static.

    Inputs
    ------
    token_ids : (batch, seq_len)  int64

    Outputs
    -------
    logits : (batch, seq_len, vocab_size)  float32
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model(token_ids, padding_mask=None)


class _EmbedWrapper(nn.Module):
    """Wraps ContextEncoder for clean ONNX tracing.

    Returns mean-pooled sequence embedding (one vector per example) so the
    output shape is fixed to (batch, d_model) — easier to consume in JS.

    Inputs
    ------
    token_ids : (batch, seq_len)  int64

    Outputs
    -------
    embeddings : (batch, d_model)  float32
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model) → (batch, d_model)
        hidden = self.model(token_ids, token_mask=None, padding_mask=None)
        return hidden.mean(dim=1)


# ----------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------

def _load_ntp(model_path: Path) -> nn.Module:
    """Load a NextTokenPredictor from a .pt checkpoint.

    Reads model_config.json from the same directory, builds the model,
    loads state dict, and switches to eval mode.
    """
    import json as _json
    from src.model import ContextEncoder, EncoderConfig, LMHead
    from src.tasks.next_token.model import NextTokenPredictor, NTPConfig

    config_path = model_path.with_name('model_config.json')
    if not config_path.exists():
        raise FileNotFoundError(
            f"model_config.json not found next to {model_path}. "
            "Re-run fine-tuning to regenerate it."
        )

    with config_path.open(encoding='utf-8') as fh:
        raw = _json.load(fh)

    encoder_config = EncoderConfig(**raw)
    encoder_config.is_causal = True
    config = NTPConfig(encoder=encoder_config, freeze_encoder=False)
    model = NextTokenPredictor(config)

    state = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_encoder(model_path: Path) -> nn.Module:
    """Load a ContextEncoder from a .pt checkpoint."""
    import json as _json
    from src.model import ContextEncoder, EncoderConfig

    config_path = model_path.with_name('model_config.json')
    if not config_path.exists():
        raise FileNotFoundError(
            f"model_config.json not found next to {model_path}."
        )

    with config_path.open(encoding='utf-8') as fh:
        raw = _json.load(fh)

    config = EncoderConfig(**raw)
    model = ContextEncoder(config)

    state = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ----------------------------------------------------------------
# Vocabulary / tokenizer JSON helpers
# ----------------------------------------------------------------

def _build_vocab_json() -> dict:
    """Produce a serialisable vocabulary dict for the JS tokenizer."""
    from src.tokenizer.vocabulary import (
        VOCAB, TOKEN_TO_ID, BOS_ID, EOS_ID, PAD_ID,
    )

    return {
        'token_to_id': TOKEN_TO_ID,
        'id_to_token': {str(k): v for k, v in enumerate(VOCAB)},
        'special': {
            'bos': BOS_ID,
            'eos': EOS_ID,
            'pad': PAD_ID,
        },
        'vocab_size': len(VOCAB),
    }


def _build_tokenizer_json() -> dict:
    """Produce a serialisable tokenizer-rules dict for the JS side.

    The JS tokenizer must reproduce exactly the behaviour of LanguageTokenizer:
    1. Lex source into tokens (split on whitespace / operators).
    2. For each token: if every character is in digit_chars, encode
       character-by-character; otherwise look up the full token in token_to_id.
    3. Wrap with bos_id … eos_id.
    """
    from src.tokenizer.vocabulary import (
        VOCAB, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    )

    keywords = [
        t for t in VOCAB
        if t not in {BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
        and not all(c in '0123456789.' for c in t)
    ]

    return {
        'description': (
            'Tokenizer for the JEPA scripting language. '
            'Numbers are split digit-by-digit; keywords and operators are '
            'single tokens; newlines are preserved as \\n tokens.'
        ),
        'number_encoding': 'digit_by_digit',
        'digit_chars': list('0123456789.'),
        'newline_token': '\\n',
        'keywords_and_operators': sorted(keywords),
        'encoding_steps': [
            'Lex source into a flat list of token strings (same as the Python Lexer).',
            'For each token: if all characters are in digit_chars, map each char individually.',
            'Otherwise: look up the full token string in token_to_id.',
            'Prepend bos_id, append eos_id.',
        ],
    }


# ----------------------------------------------------------------
# ONNX export
# ----------------------------------------------------------------

def _export(
    wrapper: nn.Module,
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> None:
    """Run torch.onnx.export with a representative dummy input."""
    # Dummy input: batch=1, seq_len=16 — concrete shapes for tracing,
    # dynamic axes declared separately so the runtime accepts any size.
    dummy_token_ids = torch.zeros(1, 16, dtype=torch.long)

    # dynamo=False forces the legacy TorchScript-based exporter, which does
    # not require onnxscript and produces cleaner graphs for transformer models.
    torch.onnx.export(
        wrapper,
        (dummy_token_ids,),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


# ----------------------------------------------------------------
# Verification
# ----------------------------------------------------------------

def _verify(onnx_path: Path, task: str) -> None:
    """Run a quick onnxruntime inference pass and compare with PyTorch output."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("  onnxruntime not installed — skipping verification.")
        print("  Install with: pip install onnxruntime")
        return

    import onnx
    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)
    print("  ONNX checker: OK")

    sess = ort.InferenceSession(str(onnx_path))
    dummy = np.zeros((1, 8), dtype=np.int64)
    outputs = sess.run(None, {'token_ids': dummy})
    print(f"  onnxruntime output shape: {outputs[0].shape}  — OK")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export a JEPA model to ONNX for use in Next.js.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model',
        required=True,
        type=Path,
        metavar='PATH',
        help='Path to the .pt checkpoint (next_token_final.pt or context_encoder_final.pt).',
    )
    parser.add_argument(
        '--output',
        required=True,
        type=Path,
        metavar='DIR',
        help='Output directory. Created if it does not exist.',
    )
    parser.add_argument(
        '--task',
        choices=['next-token', 'embed'],
        default='next-token',
        help='Which model to export.',
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version.',
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Run onnxruntime inference to verify the exported model.',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model_path: Path = args.model.resolve()
    output_dir: Path = args.output.resolve()

    if not model_path.exists():
        print(f"Error: checkpoint not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    print(f"Loading {args.task} model from {model_path} …")
    if args.task == 'next-token':
        model = _load_ntp(model_path)
        wrapper = _NTPWrapper(model)
        output_names = ['logits']
        # logits: (batch, seq_len, vocab_size)
        dynamic_axes = {
            'token_ids': {0: 'batch', 1: 'seq_len'},
            'logits':    {0: 'batch', 1: 'seq_len'},
        }
    else:
        model = _load_encoder(model_path)
        wrapper = _EmbedWrapper(model)
        output_names = ['embeddings']
        # embeddings: (batch, d_model) — mean-pooled, no seq_len axis
        dynamic_axes = {
            'token_ids':  {0: 'batch', 1: 'seq_len'},
            'embeddings': {0: 'batch'},
        }

    wrapper.eval()

    # ---- Export ONNX ----
    onnx_path = output_dir / 'model.onnx'
    print(f"Exporting ONNX (opset {args.opset}) → {onnx_path} …")
    _export(
        wrapper=wrapper,
        output_path=onnx_path,
        input_names=['token_ids'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset=args.opset,
    )
    size_mb = onnx_path.stat().st_size / 1_048_576
    print(f"  Written: {onnx_path.name}  ({size_mb:.2f} MB)")

    # ---- Vocabulary ----
    vocab_path = output_dir / 'vocab.json'
    vocab_path.write_text(
        json.dumps(_build_vocab_json(), indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(f"  Written: {vocab_path.name}")

    # ---- Tokenizer metadata ----
    tokenizer_path = output_dir / 'tokenizer.json'
    tokenizer_path.write_text(
        json.dumps(_build_tokenizer_json(), indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    print(f"  Written: {tokenizer_path.name}")

    # ---- Model config ----
    src_config = model_path.with_name('model_config.json')
    if src_config.exists():
        dst_config = output_dir / 'model_config.json'
        shutil.copy(src_config, dst_config)
        print(f"  Written: {dst_config.name}")

    # ---- Verify ----
    if args.verify:
        print("Verifying with onnxruntime …")
        _verify(onnx_path, args.task)

    print("\nDone. Next.js usage:")
    print("  npm install onnxruntime-web   # or onnxruntime-node")
    print(f"  Copy {output_dir}/ into your Next.js public/ or api/ directory.")


if __name__ == '__main__':
    main()
