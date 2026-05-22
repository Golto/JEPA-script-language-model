"""CLI subcommand: embed a single program and print its representation.

Encodes the program through the context encoder (no structural masking),
mean-pools over token positions, and reports summary statistics. The full
vector can optionally be saved as a .npy file for downstream analysis.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from ._shared import embed_program, load_context_encoder, resolve_sources


def add_embed_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the embed subcommand and its arguments.

    Args:
        subparsers: The subparsers action from the top-level argument parser.
    """
    parser = subparsers.add_parser(
        'embed',
        help='Encode a program and print its embedding statistics.',
        description=(
            'Encode a single program through the JEPA context encoder and print '
            'the mean-pooled embedding statistics. Provide the program either as '
            'an inline string or via -f.'
        ),
    )
    parser.add_argument(
        'source',
        nargs='?',
        default=None,
        metavar='PROGRAM',
        help='Program source text as an inline string.',
    )
    parser.add_argument(
        '--file', '-f',
        default=None,
        metavar='FILE',
        help='Path to a file containing the program source (alternative to inline string).',
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        metavar='PATH',
        help='Path to context_encoder_final.pt. model_config.json must be in the same directory.',
    )
    parser.add_argument(
        '--save', '-s',
        default=None,
        metavar='PATH',
        help='If given, save the full embedding vector as a .npy file at this path.',
    )


def run_embed(args: argparse.Namespace) -> None:
    """Execute the embed subcommand.

    Loads the context encoder, resolves the program source, computes the
    mean-pooled embedding, prints statistics, and optionally saves the vector.

    Args:
        args: Parsed arguments from the embed subparser.
    """
    texts = resolve_sources(
        inline_sources=[args.source] if args.source is not None else [],
        file_paths=[args.file] if args.file is not None else [],
    )

    if len(texts) != 1:
        sys.exit(
            "Error: provide exactly one program source, either as an inline "
            "string or via --file (-f)."
        )

    source = texts[0]
    encoder, tokenizer = load_context_encoder(args.model)
    embedding, n_tokens = embed_program(encoder, tokenizer, source)

    norm = embedding.norm().item()
    mean = embedding.mean().item()
    std = embedding.std().item()
    d_model = embedding.shape[0]

    preview_values = embedding[:8].tolist()
    preview_str = '  '.join(f'{v:+.4f}' for v in preview_values)

    print(f"Tokens    : {n_tokens}")
    print(f"Embedding : d={d_model}  norm={norm:.4f}  mean={mean:.4f}  std={std:.4f}")
    print(f"Preview   : [ {preview_str} ... ]")

    if args.save is not None:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, embedding.numpy())
        print(f"Saved     : {save_path}")
