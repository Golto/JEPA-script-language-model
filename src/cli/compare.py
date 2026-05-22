"""CLI subcommand: compare two programs via cosine similarity.

Encodes both programs through the context encoder, mean-pools their
representations, and reports cosine similarity and L2 distance between
the two embedding vectors.

Input order convention
----------------------
When mixing inline strings and files, inline strings always come first.
For example:

    compare "prog_a" -f prog_b.txt      # A = string, B = file
    compare -f prog_a.txt -f prog_b.txt  # A = first file, B = second file
    compare "prog_a" "prog_b"            # A = first string, B = second string

To use a file as program A and a string as program B, pass the file first:

    compare -f prog_a.txt "prog_b"       # A = file, B = string (file before string)

NOTE: positional strings are collected before -f file contents, so mixing
their order on the command line does not change which is A and which is B.
Pass both as -f if ordering matters.
"""

import argparse
import sys

import torch
import torch.nn.functional as F

from ._shared import embed_program, load_context_encoder, resolve_sources


def add_compare_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the compare subcommand and its arguments.

    Args:
        subparsers: The subparsers action from the top-level argument parser.
    """
    parser = subparsers.add_parser(
        'compare',
        help='Compute cosine similarity between two program embeddings.',
        description=(
            'Encode two programs and compare their mean-pooled embeddings. '
            'Each program can be given as an inline string or via -f FILE. '
            'Inline strings are always treated as program A before program B; '
            'use two -f flags to provide both from files.'
        ),
    )
    parser.add_argument(
        'source',
        nargs='*',
        metavar='PROGRAM',
        help=(
            'Program source text(s) as inline strings (0 to 2). '
            'Inline strings are ordered before file inputs.'
        ),
    )
    parser.add_argument(
        '--file', '-f',
        action='append',
        default=[],
        metavar='FILE',
        help=(
            'Path to a file containing a program source. '
            'Can be specified once or twice to provide both programs from files.'
        ),
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        metavar='PATH',
        help='Path to context_encoder_final.pt. model_config.json must be in the same directory.',
    )


def run_compare(args: argparse.Namespace) -> None:
    """Execute the compare subcommand.

    Resolves both program sources, embeds them, and prints cosine similarity
    and L2 distance between their mean-pooled representations.

    Args:
        args: Parsed arguments from the compare subparser.
    """
    texts = resolve_sources(
        inline_sources=list(args.source),
        file_paths=list(args.file),
    )

    if len(texts) != 2:
        sys.exit(
            f"Error: expected exactly 2 program inputs, got {len(texts)}.\n"
            f"Provide two inline strings, two -f flags, or one of each."
        )

    source_a, source_b = texts
    encoder, tokenizer = load_context_encoder(args.model)

    embedding_a, n_tokens_a = embed_program(encoder, tokenizer, source_a)
    embedding_b, n_tokens_b = embed_program(encoder, tokenizer, source_b)

    cosine_similarity = F.cosine_similarity(
        embedding_a.unsqueeze(0),
        embedding_b.unsqueeze(0),
    ).item()
    l2_distance = (embedding_a - embedding_b).norm().item()

    print(f"Program A  : {n_tokens_a} tokens")
    print(f"Program B  : {n_tokens_b} tokens")
    print()
    print(f"Cosine similarity  : {cosine_similarity:+.4f}")
    print(f"L2 distance        :  {l2_distance:.4f}")
