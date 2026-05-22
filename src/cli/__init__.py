"""Command-line interface for the JEPA project.

Entry point: main(). Dispatches to subcommands:
    embed   -- encode a single program and print its embedding statistics
    compare -- compare two programs via cosine similarity
    train   -- launch a JEPA pre-training run

Usage:
    uv run main.py embed   --model PATH [PROGRAM] [-f FILE] [--save PATH]
    uv run main.py compare --model PATH [PROGRAM_A] [PROGRAM_B] [-f FILE]...
    uv run main.py train   --data DIR [options]
"""

import argparse

from .compare import add_compare_parser, run_compare
from .embed import add_embed_parser, run_embed
from .train import add_train_parser, run_train


def main() -> None:
    """Parse command-line arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        prog='jepa',
        description='JEPA: train and inspect program embeddings.',
    )

    subparsers = parser.add_subparsers(dest='command', required=True)
    add_embed_parser(subparsers)
    add_compare_parser(subparsers)
    add_train_parser(subparsers)

    args = parser.parse_args()

    if args.command == 'embed':
        run_embed(args)
    elif args.command == 'compare':
        run_compare(args)
    elif args.command == 'train':
        run_train(args)
