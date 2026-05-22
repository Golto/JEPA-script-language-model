"""CLI command: fine-tune a downstream task on top of a JEPA encoder.

Subcommands:
    next-token  -- causal language model, predicts the next token at each position
    classify    -- sequence classifier, predicts a structural property of the program

Usage:
    uv run main.py finetune next-token --pretrained PATH --data DIR [options]
    uv run main.py finetune classify   --pretrained PATH --data DIR --label-type TYPE [options]
"""

import argparse
from pathlib import Path


def add_finetune_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the finetune command and its task subcommands.

    Args:
        subparsers: The subparsers action from the top-level argument parser.
    """
    parser = subparsers.add_parser(
        'finetune',
        help='Fine-tune a downstream task on top of a pre-trained JEPA encoder.',
        description='Fine-tune the JEPA encoder for a specific downstream task.',
    )
    task_subparsers = parser.add_subparsers(dest='task', required=True)
    _add_next_token_parser(task_subparsers)
    _add_classify_parser(task_subparsers)


def run_finetune(args: argparse.Namespace) -> None:
    """Dispatch the finetune command to the appropriate task handler.

    Args:
        args: Parsed arguments from the finetune subparser.
    """
    if args.task == 'next-token':
        _run_next_token(args)
    elif args.task == 'classify':
        _run_classify(args)


# ----------------------------------------------------------------
# next-token subcommand
# ----------------------------------------------------------------

def _add_next_token_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the next-token fine-tuning subcommand and its arguments.

    Args:
        subparsers: The subparsers action from the finetune parser.
    """
    parser = subparsers.add_parser(
        'next-token',
        help='Fine-tune next-token prediction on top of the JEPA context encoder.',
        description=(
            'Load a pre-trained JEPA context encoder, switch it to causal attention, '
            'and fine-tune a linear LM head for next-token prediction.'
        ),
    )

    # ----------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------
    paths = parser.add_argument_group('paths')
    paths.add_argument(
        '--pretrained', '-p',
        required=True,
        metavar='PATH',
        help='Path to context_encoder_final.pt from JEPA training. model_config.json must be in the same directory.',
    )
    paths.add_argument(
        '--data', '-d',
        required=True,
        metavar='DIR',
        help='Directory containing snippet_* data files.',
    )
    paths.add_argument(
        '--output', '-o',
        default='.private/output/next_token',
        metavar='DIR',
        help='Output directory for checkpoints and plots (default: .private/output/next_token).',
    )

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    training = parser.add_argument_group('training')
    training.add_argument('--epochs',         type=int,   default=20,    help='Number of fine-tuning epochs (default: 20).')
    training.add_argument('--batch-size',     type=int,   default=32,    help='Programs per gradient step (default: 32).')
    training.add_argument('--lr',             type=float, default=5e-4,  help='AdamW learning rate (default: 5e-4).')
    training.add_argument('--weight-decay',   type=float, default=1e-2,  help='AdamW weight decay (default: 1e-2).')
    training.add_argument('--val-ratio',      type=float, default=0.1,   help='Fraction held out for validation (default: 0.1).')
    training.add_argument('--freeze-encoder', action='store_true',        help='Freeze encoder weights; train LM head only (linear probing).')
    training.add_argument('--grad-clip',      type=float, default=1.0,   help='Max gradient norm. 0.0 disables (default: 1.0).')
    training.add_argument('--max-seq-len',    type=int,   default=256,   help='Maximum token sequence length (default: 256).')
    training.add_argument('--log-every',      type=int,   default=10,    help='Steps between console log lines (default: 10).')
    training.add_argument('--save-every',     type=int,   default=5,     help='Epochs between checkpoints (default: 5).')
    training.add_argument('--seed',           type=int,   default=42,    help='Random seed for train/val split (default: 42).')
    training.add_argument('--device',         type=str,   default='cpu', help='Torch device string (default: cpu).')


# ----------------------------------------------------------------
# classify subcommand
# ----------------------------------------------------------------

def _add_classify_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the classify fine-tuning subcommand and its arguments.

    Args:
        subparsers: The subparsers action from the finetune parser.
    """
    parser = subparsers.add_parser(
        'classify',
        help='Fine-tune a program classifier on top of the JEPA context encoder.',
        description=(
            'Load a pre-trained JEPA context encoder and fine-tune a linear '
            'classification head to predict a structural property of each program.'
        ),
    )

    # ----------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------
    paths = parser.add_argument_group('paths')
    paths.add_argument(
        '--pretrained', '-p',
        required=True,
        metavar='PATH',
        help='Path to context_encoder_final.pt from JEPA training.',
    )
    paths.add_argument(
        '--data', '-d',
        required=True,
        metavar='DIR',
        help='Directory containing snippet_* data files.',
    )
    paths.add_argument(
        '--output', '-o',
        default='.private/output/classify',
        metavar='DIR',
        help='Output directory for checkpoints and plots (default: .private/output/classify).',
    )

    # ----------------------------------------------------------------
    # Task
    # ----------------------------------------------------------------
    task = parser.add_argument_group('task')
    task.add_argument(
        '--label-type',
        required=True,
        choices=['has_loop', 'has_conditional', 'has_input', 'return_type'],
        help=(
            'Which program property to classify. '
            'has_loop/has_conditional/has_input are binary. '
            'return_type is multiclass.'
        ),
    )

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    training = parser.add_argument_group('training')
    training.add_argument('--epochs',         type=int,   default=20,    help='Number of fine-tuning epochs (default: 20).')
    training.add_argument('--batch-size',     type=int,   default=32,    help='Programs per gradient step (default: 32).')
    training.add_argument('--lr',             type=float, default=5e-4,  help='AdamW learning rate (default: 5e-4).')
    training.add_argument('--weight-decay',   type=float, default=1e-2,  help='AdamW weight decay (default: 1e-2).')
    training.add_argument('--val-ratio',      type=float, default=0.1,   help='Fraction held out for validation (default: 0.1).')
    training.add_argument('--freeze-encoder', action='store_true',        help='Freeze encoder weights; train head only (linear probing).')
    training.add_argument('--grad-clip',      type=float, default=1.0,   help='Max gradient norm. 0.0 disables (default: 1.0).')
    training.add_argument('--max-seq-len',    type=int,   default=256,   help='Maximum token sequence length (default: 256).')
    training.add_argument('--log-every',      type=int,   default=10,    help='Steps between console log lines (default: 10).')
    training.add_argument('--save-every',     type=int,   default=5,     help='Epochs between checkpoints (default: 5).')
    training.add_argument('--seed',           type=int,   default=42,    help='Random seed for train/val split (default: 42).')
    training.add_argument('--device',         type=str,   default='cpu', help='Torch device string (default: cpu).')


def _run_classify(args: argparse.Namespace) -> None:
    """Execute the classify fine-tuning subcommand.

    Args:
        args: Parsed arguments from the classify subparser.
    """
    from src.tasks.classification.labels import LabelType
    from src.tasks.classification.train import ClassifierFinetuneConfig, finetune_classifier

    config = ClassifierFinetuneConfig(
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        pretrained_encoder=Path(args.pretrained),
        label_type=LabelType(args.label_type),
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_seq_len=args.max_seq_len,
        val_ratio=args.val_ratio,
        freeze_encoder=args.freeze_encoder,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        device=args.device,
    )

    finetune_classifier(config)


def _run_next_token(args: argparse.Namespace) -> None:
    """Execute the next-token fine-tuning subcommand.

    Args:
        args: Parsed arguments from the next-token subparser.
    """
    from src.tasks.next_token.train import NTPFinetuneConfig, finetune_next_token

    config = NTPFinetuneConfig(
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        pretrained_encoder=Path(args.pretrained),
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_seq_len=args.max_seq_len,
        val_ratio=args.val_ratio,
        freeze_encoder=args.freeze_encoder,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        device=args.device,
    )

    finetune_next_token(config)
