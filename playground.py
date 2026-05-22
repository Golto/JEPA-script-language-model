"""Interactive playground for JEPA next-token prediction.

Launches a local Gradio web UI with two tabs:

    Generate  -- write a partial program and let the model complete it.
                 Shows the full reconstructed source and a top-k breakdown
                 for each generated step.

    Evaluate  -- paste a full program to see per-position accuracy,
                 perplexity, and the top prediction at every token.

Usage:
    uv run playground.py --model PATH [--port PORT] [--share]

Args:
    --model  Path to next_token_predictor_final.pt. model_config.json must
             be in the same directory (written automatically by finetune).
    --port   Local port to serve on (default: 7860).
    --share  Expose a public Gradio tunnel URL (for remote machines).
"""

import argparse
import math
import sys

import torch

from src.cli._shared import load_next_token_predictor
from src.tokenizer import LanguageTokenizer


# ----------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='JEPA Next-Token Prediction Playground',
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        metavar='PATH',
        help='Path to next_token_predictor_final.pt.',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        metavar='PORT',
        help='Local port to serve on (default: 7860).',
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public Gradio tunnel URL.',
    )
    return parser.parse_args()


# ----------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------

def _load_model(model_path: str) -> tuple:
    """Load the predictor and tokenizer, exiting with a clear message on failure."""
    try:
        model, tokenizer = load_next_token_predictor(model_path)
        return model, tokenizer
    except SystemExit as error:
        print(error, file=sys.stderr)
        sys.exit(1)


# ----------------------------------------------------------------
# Inference helpers
# ----------------------------------------------------------------

def _token_label(token_id: int, tokenizer: LanguageTokenizer) -> str:
    """Return a human-readable string for a single token id."""
    decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    return decoded[0] if decoded else f'<{token_id}>'


def _encode_partial(source: str, tokenizer: LanguageTokenizer) -> list[int]:
    """Encode a partial program with BOS but without EOS."""
    return [LanguageTokenizer.BOS_ID] + tokenizer.encode(
        source.strip(), add_special_tokens=False
    )


# ----------------------------------------------------------------
# Generate tab
# ----------------------------------------------------------------

def generate_completion(
    source: str,
    n_tokens: int,
    top_k: int,
    model: object,
    tokenizer: LanguageTokenizer,
) -> tuple[str, str, list[list]]:
    """Greedily generate up to n_tokens beyond the input.

    Args:
        source: Partial program text.
        n_tokens: Maximum tokens to generate.
        top_k: Number of top candidates to show per generated step.
        model: Loaded NextTokenPredictor.
        tokenizer: Matching tokenizer.

    Returns:
        Tuple of (full_source, continuation_only, top_k_table_rows).
    """
    if not source.strip():
        return '', '', []

    token_ids = _encode_partial(source, tokenizer)
    generated_ids: list[int] = list(token_ids)
    table_rows: list[list] = []

    for step in range(int(n_tokens)):
        ids_tensor = torch.tensor([generated_ids], dtype=torch.long)
        pad_mask = torch.zeros(1, len(generated_ids), dtype=torch.bool)

        with torch.no_grad():
            logits = model(ids_tensor, pad_mask)

        probs = torch.softmax(logits[0, -1], dim=-1)
        top_probs, top_ids = probs.topk(min(int(top_k), probs.shape[-1]))

        for rank, (token_id, prob) in enumerate(
            zip(top_ids.tolist(), top_probs.tolist()), 1
        ):
            table_rows.append([
                step + 1,
                rank,
                _token_label(token_id, tokenizer),
                f"{prob * 100:.2f}%",
            ])

        next_id = int(top_ids[0].item())
        generated_ids.append(next_id)

        if next_id == LanguageTokenizer.EOS_ID:
            break

    generated_only = generated_ids[len(token_ids):]
    gen_tokens = tokenizer.decode(generated_only, skip_special_tokens=True)
    continuation = tokenizer.tokens_to_source(gen_tokens)

    all_tokens = tokenizer.decode(generated_ids, skip_special_tokens=True)
    full_source = tokenizer.tokens_to_source(all_tokens)

    return full_source, continuation, table_rows


# ----------------------------------------------------------------
# Evaluate tab
# ----------------------------------------------------------------

def evaluate_program(
    source: str,
    model: object,
    tokenizer: LanguageTokenizer,
) -> tuple[str, list[list]]:
    """Evaluate next-token prediction accuracy over a full program.

    Args:
        source: Full program text.
        model: Loaded NextTokenPredictor.
        tokenizer: Matching tokenizer.

    Returns:
        Tuple of (summary_string, per_position_table_rows).
    """
    if not source.strip():
        return '', []

    token_ids = _encode_partial(source, tokenizer)
    ids_tensor = torch.tensor([token_ids], dtype=torch.long)
    pad_mask = torch.zeros(1, len(token_ids), dtype=torch.bool)

    with torch.no_grad():
        logits = model(ids_tensor, pad_mask)

    probs = torch.softmax(logits[0], dim=-1)
    active = len(token_ids) - 1

    if active <= 0:
        return 'Sequence too short.', []

    correct = 0
    total_log_prob = 0.0
    table_rows: list[list] = []

    for position in range(active):
        target_id = token_ids[position + 1]
        predicted_id = int(logits[0, position].argmax().item())
        prob_of_target = probs[position, target_id].item()

        if predicted_id == target_id:
            correct += 1
        total_log_prob += math.log(max(prob_of_target, 1e-10))

        input_label = _token_label(token_ids[position], tokenizer)
        predicted_label = _token_label(predicted_id, tokenizer)
        actual_label = _token_label(target_id, tokenizer)
        pred_prob = probs[position, predicted_id].item()
        match = 'v' if predicted_id == target_id else 'x'

        table_rows.append([
            position,
            input_label,
            f"{predicted_label}  ({pred_prob * 100:.1f}%)",
            actual_label,
            match,
        ])

    accuracy = correct / active
    perplexity = math.exp(-total_log_prob / active)

    summary = (
        f"Tokens: {len(token_ids)}   "
        f"Accuracy: {accuracy:.1%}  ({correct}/{active})   "
        f"Perplexity: {perplexity:.2f}"
    )

    return summary, table_rows


# ----------------------------------------------------------------
# UI
# ----------------------------------------------------------------

_EXAMPLE_SIMPLE = "input r0\ninput r1\noutput r0 + r1"

_EXAMPLE_LOOP = "input r0\nr1 = 1\nwhile r0 > 1 do"

_EXAMPLE_COND = (
    "input r0\n"
    "input r1\n"
    "if r1 == 0 then\n"
    "    output false\n"
    "    output 0\n"
    "else\n"
    "    output true\n"
    "    output r0 / r1\n"
    "endif"
)


def _build_ui(model: object, tokenizer: LanguageTokenizer) -> object:
    """Build and return the Gradio Blocks UI.

    Args:
        model: Loaded NextTokenPredictor in eval mode.
        tokenizer: Matching tokenizer.

    Returns:
        A Gradio Blocks demo object ready to launch.
    """
    import gradio as gr

    with gr.Blocks(title='JEPA Playground', theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            '# JEPA Playground\n'
            'Next-token prediction with a pre-trained JEPA encoder.\n\n'
            '**Generate** -- complete a partial program.  '
            '**Evaluate** -- inspect per-position predictions for a full program.'
        )

        with gr.Tabs():

            # ----------------------------------------------------------------
            # Generate tab
            # ----------------------------------------------------------------
            with gr.Tab('Generate'):
                with gr.Row():
                    with gr.Column(scale=2):
                        gen_source = gr.Textbox(
                            label='Partial program',
                            lines=10,
                            placeholder=_EXAMPLE_LOOP,
                            value=_EXAMPLE_LOOP,
                        )
                        gr.Examples(
                            examples=[
                                [_EXAMPLE_SIMPLE],
                                [_EXAMPLE_LOOP],
                            ],
                            inputs=[gen_source],
                            label='Examples',
                        )
                    with gr.Column(scale=1):
                        gen_n_tokens = gr.Slider(
                            minimum=1, maximum=60, value=12, step=1,
                            label='Tokens to generate',
                        )
                        gen_top_k = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label='Top-k shown per step',
                        )
                        gen_btn = gr.Button('Generate', variant='primary', size='lg')

                gen_full = gr.Textbox(
                    label='Full program (input + generated)', lines=10, interactive=False
                )
                gen_continuation = gr.Textbox(
                    label='Generated continuation only', lines=4, interactive=False
                )
                gen_table = gr.DataFrame(
                    headers=['Step', 'Rank', 'Token', 'Probability'],
                    label='Top-k candidates at each generated step',
                )

                gen_btn.click(
                    fn=lambda src, n, k: generate_completion(src, n, k, model, tokenizer),
                    inputs=[gen_source, gen_n_tokens, gen_top_k],
                    outputs=[gen_full, gen_continuation, gen_table],
                )

            # ----------------------------------------------------------------
            # Evaluate tab
            # ----------------------------------------------------------------
            with gr.Tab('Evaluate'):
                with gr.Row():
                    with gr.Column(scale=2):
                        eval_source = gr.Textbox(
                            label='Full program',
                            lines=10,
                            placeholder=_EXAMPLE_COND,
                            value=_EXAMPLE_SIMPLE,
                        )
                        gr.Examples(
                            examples=[
                                [_EXAMPLE_SIMPLE],
                                [_EXAMPLE_COND],
                            ],
                            inputs=[eval_source],
                            label='Examples',
                        )
                    with gr.Column(scale=1):
                        eval_btn = gr.Button('Evaluate', variant='primary', size='lg')

                eval_summary = gr.Textbox(
                    label='Summary', interactive=False, lines=1
                )
                eval_table = gr.DataFrame(
                    headers=['Pos', 'Input token', 'Predicted (prob)', 'Actual', 'Match'],
                    label='Per-position predictions',
                )

                eval_btn.click(
                    fn=lambda src: evaluate_program(src, model, tokenizer),
                    inputs=[eval_source],
                    outputs=[eval_summary, eval_table],
                )

    return demo


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------

def main() -> None:
    """Parse arguments, load the model, and launch the Gradio UI."""
    args = _parse_args()
    print(f"Loading model from {args.model} ...")
    model, tokenizer = _load_model(args.model)
    print("Model loaded. Starting playground ...")

    demo = _build_ui(model, tokenizer)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
