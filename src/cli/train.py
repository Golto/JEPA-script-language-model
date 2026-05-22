"""CLI subcommand: launch a JEPA pre-training run.

Exposes the full PretrainConfig surface as command-line flags so that
training can be started directly with uv run main.py train [flags]
without writing any Python.
"""

import argparse
from pathlib import Path

from src.model import EncoderConfig, JEPAConfig, MLPPredictorConfig
from src.tokenizer import LanguageTokenizer
from src.training.pretrain import PretrainConfig, pretrain
from src.training.regularization import VICRegConfig


def add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the train subcommand and its arguments.

    Args:
        subparsers: The subparsers action from the top-level argument parser.
    """
    parser = subparsers.add_parser(
        'train',
        help='Launch a JEPA pre-training run.',
        description='Pre-train the JEPA context encoder on a directory of snippet files.',
    )

    # ----------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------
    paths = parser.add_argument_group('paths')
    paths.add_argument(
        '--data', '-d',
        required=True,
        metavar='DIR',
        help='Directory containing snippet_* data files.',
    )
    paths.add_argument(
        '--output', '-o',
        default='.private/output/jepa',
        metavar='DIR',
        help='Output directory for checkpoints and plots (default: .private/output/jepa).',
    )

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    training = parser.add_argument_group('training')
    training.add_argument('--epochs',       type=int,   default=50,    help='Number of training epochs (default: 50).')
    training.add_argument('--batch-size',   type=int,   default=32,    help='Programs per gradient step (default: 32).')
    training.add_argument('--lr',           type=float, default=1e-3,  help='AdamW learning rate (default: 1e-3).')
    training.add_argument('--weight-decay', type=float, default=1e-2,  help='AdamW weight decay (default: 1e-2).')
    training.add_argument('--n-blocks',     type=int,   default=1,     help='Structural blocks masked per sample (default: 1). Increase to harden the task.')
    training.add_argument('--max-seq-len',  type=int,   default=256,   help='Maximum token sequence length (default: 256).')
    training.add_argument('--log-every',    type=int,   default=10,    help='Steps between console log lines (default: 10).')
    training.add_argument('--save-every',   type=int,   default=10,    help='Epochs between checkpoints (default: 10).')
    training.add_argument('--grad-clip',    type=float, default=1.0,   help='Max gradient norm for clipping. 0.0 disables (default: 1.0).')
    training.add_argument('--lambda-var',   type=float, default=0.25,  help='VICReg variance loss weight (default: 0.25). Increase if collapse persists; decrease if cos_sim drops too low.')
    training.add_argument('--lambda-cov',   type=float, default=0.04,  help='VICReg covariance loss weight (default: 0.04).')
    training.add_argument('--device',       type=str,   default='cpu', help='Torch device string, e.g. cpu or cuda (default: cpu).')

    # ----------------------------------------------------------------
    # Architecture
    # ----------------------------------------------------------------
    arch = parser.add_argument_group('architecture')
    arch.add_argument('--d-model',       type=int,   default=128,   help='Embedding and hidden dimension (default: 128).')
    arch.add_argument('--n-layers',      type=int,   default=4,     help='Transformer encoder layers (default: 4).')
    arch.add_argument('--n-heads',       type=int,   default=4,     help='Attention heads (default: 4).')
    arch.add_argument('--d-feedforward', type=int,   default=512,   help='FFN inner dimension (default: 512).')
    arch.add_argument('--dropout',       type=float, default=0.1,   help='Dropout probability (default: 0.1).')
    arch.add_argument('--ema-decay',     type=float, default=0.996, help='EMA decay for target encoder (default: 0.996).')
    arch.add_argument('--d-hidden',      type=int,   default=256,   help='MLP predictor hidden dimension (default: 256).')
    arch.add_argument('--predictor-layers', type=int, default=2,   help='MLP predictor number of layers (default: 2).')


def run_train(args: argparse.Namespace) -> None:
    """Execute the train subcommand.

    Builds PretrainConfig and JEPAConfig from parsed CLI arguments,
    then delegates to pretrain().

    Args:
        args: Parsed arguments from the train subparser.
    """
    encoder_config = EncoderConfig(
        vocab_size=LanguageTokenizer.VOCAB_SIZE,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_feedforward=args.d_feedforward,
        dropout=args.dropout,
    )
    predictor_config = MLPPredictorConfig(
        d_model=args.d_model,
        n_layers=args.predictor_layers,
        d_hidden=args.d_hidden,
    )
    jepa_config = JEPAConfig(
        encoder=encoder_config,
        predictor=predictor_config,
        ema_decay=args.ema_decay,
    )
    pretrain_config = PretrainConfig(
        data_dir=Path(args.data),
        output_dir=Path(args.output),
        jepa=jepa_config,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        n_blocks=args.n_blocks,
        max_seq_len=args.max_seq_len,
        log_every=args.log_every,
        save_every=args.save_every,
        vicreg=VICRegConfig(lambda_var=args.lambda_var, lambda_cov=args.lambda_cov),
        grad_clip=args.grad_clip,
        device=args.device,
    )

    pretrain(pretrain_config)
