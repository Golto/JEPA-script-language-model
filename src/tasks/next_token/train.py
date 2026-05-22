"""Fine-tuning loop for next-token prediction on top of a JEPA encoder.

Entry point: finetune_next_token(config). Loads the pre-trained context
encoder, fine-tunes the full model (or LM head only if freeze_encoder=True),
and saves checkpoints, final weights, and diagnostic plots.

Saved files:
    checkpoints/checkpoint_epoch_NNN.pt   model + optimizer state
    next_token_predictor_final.pt         full model state dict
    lm_head_final.pt                      LM head only (lightweight artifact)
    training_curves.png                   loss, perplexity, accuracy
"""

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import load_all_snippets
from src.data.lm_dataset import LMDataset, collate_lm_samples
from src.tokenizer import LanguageTokenizer

from .model import NextTokenPredictor


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

@dataclass
class NTPFinetuneConfig:
    """Full configuration for one next-token prediction fine-tuning run.

    Args:
        data_dir: Directory containing snippet_* data files.
        output_dir: Where to write checkpoints and plots.
        pretrained_encoder: Path to context_encoder_final.pt from JEPA training.
            model_config.json must be in the same directory.
        n_epochs: Number of full passes over the training set.
        batch_size: Programs per gradient step.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        max_seq_len: Maximum token sequence length. Longer programs are truncated.
        val_ratio: Fraction of programs held out for validation (0.0 disables).
        freeze_encoder: If True, only the LM head is trained (linear probing).
        grad_clip: Maximum gradient norm. 0.0 disables clipping.
        log_every: Steps between console log lines.
        save_every: Epochs between checkpoint saves.
        seed: Random seed for the train/val split.
        device: Torch device string ('cpu', 'cuda', ...).
    """

    data_dir: Path
    output_dir: Path
    pretrained_encoder: Path
    n_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    max_seq_len: int = 256
    val_ratio: float = 0.1
    freeze_encoder: bool = False
    grad_clip: float = 1.0
    log_every: int = 10
    save_every: int = 5
    seed: int = 42
    device: str = 'cpu'


# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------

def finetune_next_token(config: NTPFinetuneConfig) -> NextTokenPredictor:
    """Fine-tune next-token prediction on top of a pre-trained JEPA encoder.

    Args:
        config: Full fine-tuning configuration.

    Returns:
        The fine-tuned NextTokenPredictor.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = config.output_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    device = torch.device(config.device)

    # ----------------------------------------------------------------
    # Data split
    # ----------------------------------------------------------------
    tokenizer = LanguageTokenizer()
    snippets = load_all_snippets(config.data_dir)
    all_programs = [p for snippet in snippets for p in snippet.programs]

    rng = random.Random(config.seed)
    programs_shuffled = list(all_programs)
    rng.shuffle(programs_shuffled)

    n_val = int(len(programs_shuffled) * config.val_ratio)
    val_programs = programs_shuffled[:n_val]
    train_programs = programs_shuffled[n_val:]

    train_dataset = LMDataset(train_programs, tokenizer, max_seq_len=config.max_seq_len)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_lm_samples,
    )

    has_val = len(val_programs) > 0
    if has_val:
        val_dataset = LMDataset(val_programs, tokenizer, max_seq_len=config.max_seq_len)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_lm_samples,
        )

    print(f"Dataset: {len(train_dataset)} train / {len(val_programs)} val programs")

    # ----------------------------------------------------------------
    # Model and optimizer
    # ----------------------------------------------------------------
    model = NextTokenPredictor.from_pretrained(
        config.pretrained_encoder,
        freeze_encoder=config.freeze_encoder,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    mode = 'frozen encoder' if config.freeze_encoder else 'full fine-tuning'
    print(f"Model: {trainable:,} / {total:,} trainable parameters ({mode})")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    history_train_loss: list[float] = []
    history_val_loss: list[float] = []
    history_val_ppl: list[float] = []
    history_val_acc: list[float] = []
    history_epochs: list[int] = []

    global_step = 0

    for epoch in range(1, config.n_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch in train_dataloader:
            token_ids = batch.token_ids.to(device)
            padding_mask = batch.padding_mask.to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(token_ids, padding_mask)
            loss.backward()

            if config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()

            global_step += 1
            epoch_loss_sum += loss.item()
            epoch_steps += 1

            if global_step % config.log_every == 0:
                print(
                    f"epoch {epoch:3d}  step {global_step:5d}"
                    f"  loss {loss.item():.4f}"
                    f"  ppl {math.exp(loss.item()):.2f}"
                )

        avg_train_loss = epoch_loss_sum / max(epoch_steps, 1)

        val_loss, val_acc = _evaluate(model, val_dataloader, device) if has_val else (float('nan'), float('nan'))
        val_ppl = math.exp(val_loss) if not math.isnan(val_loss) else float('nan')

        history_train_loss.append(avg_train_loss)
        history_val_loss.append(val_loss)
        history_val_ppl.append(val_ppl)
        history_val_acc.append(val_acc)
        history_epochs.append(epoch)

        print(
            f"  -- epoch {epoch}"
            f"  train_loss {avg_train_loss:.4f}"
            + (f"  val_loss {val_loss:.4f}  val_ppl {val_ppl:.2f}  val_acc {val_acc:.4f}" if has_val else "")
        )

        if epoch % config.save_every == 0:
            _save_checkpoint(model, optimizer, epoch, global_step, checkpoints_dir)

    # ----------------------------------------------------------------
    # Final save and plots
    # ----------------------------------------------------------------
    _save_final_models(model, config.output_dir, config.pretrained_encoder)
    _save_plots(history_epochs, history_train_loss, history_val_loss, history_val_ppl, history_val_acc, config.output_dir)
    print(f"Output saved to {config.output_dir}")

    return model


# ----------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------

@torch.no_grad()
def _evaluate(
    model: NextTokenPredictor,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Compute validation loss and token accuracy.

    Args:
        model: The NextTokenPredictor in eval mode.
        dataloader: Validation dataloader.
        device: Device to run inference on.

    Returns:
        Tuple of (average cross-entropy loss, token accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_active = 0

    for batch in dataloader:
        token_ids = batch.token_ids.to(device)
        padding_mask = batch.padding_mask.to(device)

        logits = model(token_ids, padding_mask)
        vocab_size = logits.shape[-1]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = token_ids[:, 1:].contiguous()

        flat_logits = shift_logits.view(-1, vocab_size)
        flat_targets = shift_targets.view(-1)

        total_loss += F.cross_entropy(flat_logits, flat_targets, ignore_index=LanguageTokenizer.PAD_ID).item()

        is_active = flat_targets != LanguageTokenizer.PAD_ID
        predictions = flat_logits.argmax(dim=-1)
        total_correct += ((predictions == flat_targets) & is_active).sum().item()
        total_active += is_active.sum().item()

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = total_correct / max(total_active, 1)

    return avg_loss, accuracy


# ----------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------

def _save_checkpoint(
    model: NextTokenPredictor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    checkpoints_dir: Path,
) -> None:
    """Save a mid-training checkpoint.

    Args:
        model: The model being trained.
        optimizer: The optimizer whose state is saved for resuming.
        epoch: Current epoch number (used in the filename).
        global_step: Total optimizer steps taken so far.
        checkpoints_dir: Directory where the checkpoint file is written.
    """
    path = checkpoints_dir / f'checkpoint_epoch_{epoch:03d}.pt'
    torch.save(
        {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        path,
    )
    print(f"  -- checkpoint saved: {path.name}")


def _save_final_models(
    model: NextTokenPredictor,
    output_dir: Path,
    pretrained_encoder: Path,
) -> None:
    """Save final model artifacts.

    Writes three files:
        next_token_predictor_final.pt  -- full model (encoder + LM head)
        lm_head_final.pt               -- LM head only (lightweight artifact)
        model_config.json              -- encoder architecture config copied from the
                                         JEPA output dir so this directory is
                                         self-contained for inference

    Args:
        model: The fine-tuned NextTokenPredictor.
        output_dir: Directory where the files are written.
        pretrained_encoder: Path to the JEPA context_encoder_final.pt used during
            fine-tuning. model_config.json is copied from the same directory.
    """
    import shutil

    torch.save(model.state_dict(), output_dir / 'next_token_predictor_final.pt')
    torch.save(model.lm_head.state_dict(), output_dir / 'lm_head_final.pt')

    src_config = pretrained_encoder.with_name('model_config.json')
    if src_config.exists():
        shutil.copy(src_config, output_dir / 'model_config.json')

    print(f"  -- final models saved to {output_dir}")


# ----------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------

def _save_plots(
    epochs: list[int],
    train_loss: list[float],
    val_loss: list[float],
    val_ppl: list[float],
    val_acc: list[float],
    output_dir: Path,
) -> None:
    """Save training curve plots to output_dir.

    Args:
        epochs: Epoch indices for the x-axis.
        train_loss: Training cross-entropy loss per epoch.
        val_loss: Validation cross-entropy loss per epoch.
        val_ppl: Validation perplexity per epoch.
        val_acc: Validation token accuracy per epoch.
        output_dir: Directory where the PNG file is written.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    has_val = not all(math.isnan(v) for v in val_loss)

    fig, (axis_loss, axis_ppl, axis_acc) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axis_loss.plot(epochs, train_loss, label='train', linewidth=1.5)
    if has_val:
        axis_loss.plot(epochs, val_loss, label='val', linewidth=1.5, linestyle='--')
        axis_loss.legend(fontsize=9)
    axis_loss.set_ylabel('Cross-Entropy Loss')
    axis_loss.set_title('Next-Token Prediction Fine-tuning')
    axis_loss.grid(True, alpha=0.3)

    if has_val:
        axis_ppl.plot(epochs, val_ppl, color='steelblue', linewidth=1.5)
    axis_ppl.set_ylabel('Val Perplexity')
    axis_ppl.grid(True, alpha=0.3)

    if has_val:
        axis_acc.plot(epochs, val_acc, color='seagreen', linewidth=1.5)
    axis_acc.set_xlabel('Epoch')
    axis_acc.set_ylabel('Val Token Accuracy')
    axis_acc.set_ylim(0.0, 1.0)
    axis_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close(fig)
