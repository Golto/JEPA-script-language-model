"""CLI command: run inference with a fine-tuned task model.

Subcommands:
    next-token  -- show summary stats, top-k candidates at the last position,
                   optional per-position table, optional greedy generation
    classify    -- predict a structural property of a program and show
                   class probabilities

Usage:
    uv run main.py predict next-token --model PATH [PROGRAM] [-f FILE]
                                      [--top-k K] [--generate N] [--verbose]
    uv run main.py predict classify   --model PATH [PROGRAM] [-f FILE]
"""

import argparse
import math
import sys
from typing import TYPE_CHECKING

import torch

from src.cli._shared import (
    load_next_token_predictor,
    load_program_classifier,
    resolve_sources,
)
from src.tokenizer import LanguageTokenizer

if TYPE_CHECKING:
    from src.tasks.classification.model import ProgramClassifier
    from src.tasks.next_token.model import NextTokenPredictor


# ----------------------------------------------------------------
# Parser registration
# ----------------------------------------------------------------

def add_predict_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the predict command and its task subcommands.

    Args:
        subparsers: The subparsers action from the top-level argument parser.
    """
    parser = subparsers.add_parser(
        'predict',
        help='Run inference with a fine-tuned task model.',
        description='Run inference with a fine-tuned next-token or classification model.',
    )
    task_subparsers = parser.add_subparsers(dest='task', required=True)
    _add_next_token_predict_parser(task_subparsers)
    _add_classify_predict_parser(task_subparsers)


def run_predict(args: argparse.Namespace) -> None:
    """Dispatch the predict command to the appropriate task handler.

    Args:
        args: Parsed arguments from the predict subparser.
    """
    if args.task == 'next-token':
        _run_next_token_predict(args)
    elif args.task == 'classify':
        _run_classify_predict(args)


# ----------------------------------------------------------------
# next-token subcommand
# ----------------------------------------------------------------

def _add_next_token_predict_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the next-token inference subcommand.

    Args:
        subparsers: The subparsers action from the predict parser.
    """
    parser = subparsers.add_parser(
        'next-token',
        help='Run next-token prediction on a single program.',
        description=(
            'Tokenize a program, run the fine-tuned NextTokenPredictor, '
            'and print accuracy, perplexity, top-k candidates at the last '
            'position, and an optional greedy continuation.'
        ),
    )

    parser.add_argument(
        '--model', '-m',
        required=True,
        metavar='PATH',
        help=(
            'Path to next_token_predictor_final.pt. '
            'model_config.json must be in the same directory.'
        ),
    )
    parser.add_argument(
        'source',
        nargs='?',
        default=None,
        metavar='PROGRAM',
        help='Program text inline on the command line. Omit if using -f.',
    )
    parser.add_argument(
        '-f', '--file',
        default=None,
        metavar='FILE',
        help='Read the program from this file instead of the command line.',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        metavar='K',
        help='Number of top candidates to show at the last position (default: 5).',
    )
    parser.add_argument(
        '--generate',
        type=int,
        default=0,
        metavar='N',
        help='Greedily generate N additional tokens after the input (default: 0).',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show a per-position prediction table for every token in the sequence.',
    )


def _run_next_token_predict(args: argparse.Namespace) -> None:
    """Execute next-token prediction inference and print results.

    Args:
        args: Parsed arguments from the next-token subparser.
    """
    inline = [args.source] if args.source else []
    files = [args.file] if args.file else []
    sources = resolve_sources(inline, files)
    if len(sources) != 1:
        sys.exit("Error: provide exactly one program -- either a string or -f FILE.")

    model, tokenizer = load_next_token_predictor(args.model)

    # Encode with BOS but without EOS so --generate can extend naturally.
    token_ids = [LanguageTokenizer.BOS_ID] + tokenizer.encode(
        sources[0].strip(), add_special_tokens=False
    )
    token_ids_tensor = torch.tensor([token_ids], dtype=torch.long)
    padding_mask = torch.zeros(1, len(token_ids), dtype=torch.bool)

    with torch.no_grad():
        logits = model(token_ids_tensor, padding_mask)  # (1, seq_len, vocab_size)

    logits = logits[0]  # (seq_len, vocab_size)
    probs = torch.softmax(logits, dim=-1)

    _print_summary(token_ids, probs, tokenizer)

    if args.verbose:
        _print_position_table(token_ids, logits, tokenizer)

    _print_top_k(token_ids, probs, tokenizer, top_k=args.top_k)

    if args.generate > 0:
        _print_generation(token_ids, model, tokenizer, n_tokens=args.generate)


# ----------------------------------------------------------------
# classify subcommand
# ----------------------------------------------------------------

def _add_classify_predict_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the classify inference subcommand.

    Args:
        subparsers: The subparsers action from the predict parser.
    """
    parser = subparsers.add_parser(
        'classify',
        help='Predict a structural property of a program.',
        description=(
            'Tokenize a program, run the fine-tuned ProgramClassifier, '
            'and print the predicted class with all class probabilities.'
        ),
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        metavar='PATH',
        help=(
            'Path to classifier_final.pt. '
            'model_config.json and label_vocab.json must be in the same directory.'
        ),
    )
    parser.add_argument(
        'source',
        nargs='?',
        default=None,
        metavar='PROGRAM',
        help='Program text inline on the command line. Omit if using -f.',
    )
    parser.add_argument(
        '-f', '--file',
        default=None,
        metavar='FILE',
        help='Read the program from this file instead of the command line.',
    )


def _run_classify_predict(args: argparse.Namespace) -> None:
    """Execute classification inference and print results.

    Args:
        args: Parsed arguments from the classify subparser.
    """
    inline = [args.source] if args.source else []
    files = [args.file] if args.file else []
    sources = resolve_sources(inline, files)
    if len(sources) != 1:
        sys.exit("Error: provide exactly one program -- either a string or -f FILE.")

    model, tokenizer, label_vocab = load_program_classifier(args.model)
    id_to_label = {index: label for label, index in label_vocab.items()}

    token_ids = tokenizer.encode(sources[0].strip(), add_special_tokens=True)
    token_ids_tensor = torch.tensor([token_ids], dtype=torch.long)
    padding_mask = torch.zeros(1, len(token_ids), dtype=torch.bool)

    with torch.no_grad():
        logits = model(token_ids_tensor, padding_mask)  # (1, n_classes)

    probs = torch.softmax(logits[0], dim=-1)
    predicted_index = probs.argmax().item()
    predicted_label = id_to_label[predicted_index]

    print(f"Tokens:    {len(token_ids)}")
    print(f"Predicted: {predicted_label!r}  ({probs[predicted_index].item() * 100:.1f}%)")
    print()
    print("All classes:")
    for index, prob in sorted(enumerate(probs.tolist()), key=lambda x: -x[1]):
        label = id_to_label[index]
        marker = ' <--' if index == predicted_index else ''
        print(f"  {label!r:<20} {prob * 100:6.2f}%{marker}")


# ----------------------------------------------------------------
# Display helpers
# ----------------------------------------------------------------

def _token_label(token_id: int, tokenizer: LanguageTokenizer) -> str:
    """Return a human-readable string for a single token id.

    Args:
        token_id: Vocabulary index to decode.
        tokenizer: Tokenizer matching the model vocabulary.

    Returns:
        The token string, including special tokens like <|bos|>.
    """
    decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    return decoded[0] if decoded else f'<{token_id}>'


def _print_summary(
    token_ids: list[int],
    probs: torch.Tensor,
    tokenizer: LanguageTokenizer,
) -> None:
    """Print token count, accuracy, and perplexity for the full sequence.

    Args:
        token_ids: Encoded token ids including BOS.
        probs: Softmax probabilities, shape (seq_len, vocab_size).
        tokenizer: Tokenizer for decoding.
    """
    active = len(token_ids) - 1  # position i predicts i+1
    if active <= 0:
        print("Sequence too short for prediction (need at least 2 tokens).")
        return

    correct = 0
    total_log_prob = 0.0

    for position in range(active):
        target_id = token_ids[position + 1]
        predicted_id = probs[position].argmax().item()
        if predicted_id == target_id:
            correct += 1
        total_log_prob += math.log(max(probs[position, target_id].item(), 1e-10))

    accuracy = correct / active
    perplexity = math.exp(-total_log_prob / active)

    print(f"Tokens:     {len(token_ids)}")
    print(f"Accuracy:   {accuracy:.1%}  ({correct}/{active} correct)")
    print(f"Perplexity: {perplexity:.2f}")
    print()


def _print_position_table(
    token_ids: list[int],
    logits: torch.Tensor,
    tokenizer: LanguageTokenizer,
) -> None:
    """Print a table of input token, predicted next token, actual next token, and match.

    Args:
        token_ids: Encoded token ids including BOS.
        logits: Raw logits, shape (seq_len, vocab_size).
        tokenizer: Tokenizer for decoding.
    """
    column_width = 16
    header = (
        f"{'Pos':<5} {'Input':<{column_width}} "
        f"{'Predicted':<{column_width}} {'Actual':<{column_width}} Match"
    )
    print(header)
    print('-' * len(header))

    for position in range(len(token_ids) - 1):
        input_label = _token_label(token_ids[position], tokenizer)
        predicted_id = logits[position].argmax().item()
        predicted_label = _token_label(predicted_id, tokenizer)
        actual_label = _token_label(token_ids[position + 1], tokenizer)
        match_marker = 'v' if predicted_id == token_ids[position + 1] else 'x'
        print(
            f"{position:<5} {input_label:<{column_width}} "
            f"{predicted_label:<{column_width}} {actual_label:<{column_width}} {match_marker}"
        )
    print()


def _print_top_k(
    token_ids: list[int],
    probs: torch.Tensor,
    tokenizer: LanguageTokenizer,
    top_k: int,
) -> None:
    """Print the top-k predicted next tokens at the last position in the sequence.

    Args:
        token_ids: Encoded token ids including BOS.
        probs: Softmax probabilities, shape (seq_len, vocab_size).
        tokenizer: Tokenizer for decoding.
        top_k: Number of candidates to display.
    """
    last_position = len(token_ids) - 1
    last_label = _token_label(token_ids[last_position], tokenizer)

    vocab_size = probs.shape[-1]
    top_probs, top_ids = probs[last_position].topk(min(top_k, vocab_size))

    print(f"Top-{top_k} predictions after position {last_position} ({last_label!r}):")
    for rank, (token_id, prob) in enumerate(zip(top_ids.tolist(), top_probs.tolist()), 1):
        label = _token_label(token_id, tokenizer)
        print(f"  {rank}. {label!r:<22} {prob * 100:6.2f}%")
    print()


def _print_generation(
    token_ids: list[int],
    model: 'NextTokenPredictor',
    tokenizer: LanguageTokenizer,
    n_tokens: int,
) -> None:
    """Greedily generate up to n_tokens beyond the input and print the result.

    Generation stops early if EOS is produced. The full sequence (input +
    generated tokens) is reconstructed and printed as readable source code.

    Args:
        token_ids: Encoded token ids (BOS + input, without EOS).
        model: The NextTokenPredictor in eval mode.
        tokenizer: Tokenizer for decoding and source reconstruction.
        n_tokens: Maximum number of tokens to generate.
    """
    generated_ids: list[int] = list(token_ids)

    for _ in range(n_tokens):
        ids_tensor = torch.tensor([generated_ids], dtype=torch.long)
        pad_mask = torch.zeros(1, len(generated_ids), dtype=torch.bool)

        with torch.no_grad():
            step_logits = model(ids_tensor, pad_mask)

        next_id = int(step_logits[0, -1].argmax().item())
        generated_ids.append(next_id)

        if next_id == LanguageTokenizer.EOS_ID:
            break

    n_generated = len(generated_ids) - len(token_ids)
    all_tokens = tokenizer.decode(generated_ids, skip_special_tokens=True)
    source = tokenizer.tokens_to_source(all_tokens)

    print(f"Greedy continuation ({n_generated} token(s) generated):")
    print(source)
    print()
