"""JEPA pre-training loop.

Entry point: pretrain(config). Loads snippets, trains the JEPA model,
monitors for representation collapse, and saves diagnostic plots at the end.

Collapse is tracked via two signals logged every log_every steps:
    - embedding_std:   should stay well above 0.
    - mean_cosine_sim: should stay well below 1.
A warning is printed if either crosses its collapse threshold.

Saved files:
    checkpoints/checkpoint_epoch_NNN.pt  -- model + optimizer state every save_every epochs
    jepa_final.pt                        -- full model state dict at end of training
    context_encoder_final.pt             -- context encoder only (kept after pre-training)
    training_loss.png
    collapse_metrics.png
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import JEPADataset, collate_jepa_samples, load_all_snippets
from src.model import JEPAConfig, JEPAModel
from src.tokenizer import LanguageTokenizer

from .metrics import compute_collapse_metrics
from .regularization import VICRegConfig, compute_vicreg_loss


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

@dataclass
class PretrainConfig:
    """Full configuration for one pre-training run.

    Args:
        data_dir: Path to the directory containing snippet_* files.
        output_dir: Where to write checkpoints and diagnostic plots.
        jepa: Architecture configuration for the JEPA model.
        n_epochs: Number of full passes over the dataset.
        batch_size: Number of programs per gradient step.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        max_seq_len: Maximum token sequence length. Longer programs are truncated.
        log_every: Number of steps between console log lines.
        save_every: Number of epochs between checkpoint saves.
        n_blocks: Structural blocks masked per sample. Increase to harden the task.
        vicreg: VICReg regularization weights. Controls the anti-collapse terms
            applied to z_context. Set lambda_var=0 and lambda_cov=0 to disable.
        grad_clip: Maximum gradient norm for clipping. 0.0 disables clipping.
        device: Torch device string ('cpu', 'cuda', 'cuda:0', ...).
        collapse_std_threshold: embedding_std below this value triggers a
            collapse warning.
        collapse_cosine_threshold: mean_cosine_sim above this value triggers a
            collapse warning.
    """

    data_dir: Path
    output_dir: Path
    jepa: JEPAConfig
    n_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    max_seq_len: int = 256
    log_every: int = 10
    save_every: int = 10
    n_blocks: int = 1
    vicreg: VICRegConfig = None  # type: ignore[assignment]  -- set in __post_init__
    grad_clip: float = 1.0
    device: str = 'cpu'
    collapse_std_threshold: float = 0.05
    collapse_cosine_threshold: float = 0.95

    def __post_init__(self) -> None:
        # Default to a fresh VICRegConfig rather than using a mutable default arg.
        if self.vicreg is None:
            self.vicreg = VICRegConfig()


# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------

def pretrain(config: PretrainConfig) -> JEPAModel:
    """Run JEPA pre-training and return the trained model.

    Saves two plots to config.output_dir:
        training_loss.png      -- MSE loss over training steps
        collapse_metrics.png   -- embedding_std and mean_cosine_sim over steps

    Args:
        config: Full pre-training configuration.

    Returns:
        The trained JEPAModel (context encoder weights are the useful output).
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = config.output_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    device = torch.device(config.device)

    # ----------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------
    tokenizer = LanguageTokenizer()
    snippets = load_all_snippets(config.data_dir)
    programs = [program for snippet in snippets for program in snippet.programs]
    dataset = JEPADataset(programs, tokenizer, max_seq_len=config.max_seq_len, n_blocks=config.n_blocks)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_jepa_samples,
        drop_last=False,
    )
    print(f"Dataset: {len(dataset)} programs across {len(snippets)} snippet files")

    # ----------------------------------------------------------------
    # Model and optimizer
    # ----------------------------------------------------------------
    model = JEPAModel(config.jepa).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {trainable_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    history_loss: list[float] = []
    history_reg: list[float] = []
    history_std: list[float] = []
    history_cosine: list[float] = []
    history_steps: list[int] = []

    global_step = 0

    for epoch in range(1, config.n_epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch in dataloader:
            token_ids = batch.token_ids.to(device)
            structural_mask = batch.structural_mask.to(device)
            padding_mask = batch.padding_mask.to(device)

            # Skip batches where no sample has any masked token.
            if not structural_mask.any():
                continue

            optimizer.zero_grad()
            output = model(token_ids, structural_mask, padding_mask)

            masked_z_hat = output.z_hat[structural_mask]
            masked_z_target = output.z_target[structural_mask]
            mse_loss = F.mse_loss(masked_z_hat, masked_z_target)

            reg = compute_vicreg_loss(output.z_context, config.vicreg, padding_mask)
            loss = mse_loss + reg.total

            loss.backward()
            if config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            model.update_target_encoder()

            global_step += 1
            epoch_loss_sum += mse_loss.item()
            epoch_steps += 1

            if global_step % config.log_every == 0:
                metrics = compute_collapse_metrics(output.z_target.detach(), padding_mask)
                history_loss.append(mse_loss.item())
                history_reg.append(reg.total.item())
                history_std.append(metrics['embedding_std'])
                history_cosine.append(metrics['mean_cosine_sim'])
                history_steps.append(global_step)

                collapse_warning = ''
                if metrics['embedding_std'] < config.collapse_std_threshold:
                    collapse_warning += '  [WARNING: std collapse]'
                if metrics['mean_cosine_sim'] > config.collapse_cosine_threshold:
                    collapse_warning += '  [WARNING: cosine collapse]'

                print(
                    f"epoch {epoch:3d}  step {global_step:5d}"
                    f"  mse {mse_loss.item():.4f}"
                    f"  reg {reg.total.item():.4f}"
                    f"  emb_std {metrics['embedding_std']:.4f}"
                    f"  cos_sim {metrics['mean_cosine_sim']:.4f}"
                    + collapse_warning
                )

        if epoch_steps > 0:
            avg_loss = epoch_loss_sum / epoch_steps
            print(f"  -- epoch {epoch} avg loss: {avg_loss:.4f}")

        if epoch % config.save_every == 0:
            _save_checkpoint(model, optimizer, epoch, global_step, checkpoints_dir)

    # ----------------------------------------------------------------
    # Final models
    # ----------------------------------------------------------------
    _save_final_models(model, config.jepa, config.output_dir)

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    _save_plots(history_steps, history_loss, history_reg, history_std, history_cosine, config.output_dir)
    print(f"Plots saved to {config.output_dir}")

    return model


# ----------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------

def _save_checkpoint(
    model: JEPAModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    checkpoints_dir: Path,
) -> None:
    """Save a mid-training checkpoint to resume or inspect later.

    Saves model and optimizer state so training can be resumed from this point.

    Args:
        model: The JEPAModel being trained.
        optimizer: The optimizer whose state is also saved.
        epoch: Current epoch number (used in the filename).
        global_step: Total number of optimizer steps taken so far.
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


def _save_final_models(model: JEPAModel, jepa_config: JEPAConfig, output_dir: Path) -> None:
    """Save the final model artifacts after training completes.

    Writes three files:
        jepa_final.pt            -- full JEPAModel state dict (all components)
        context_encoder_final.pt -- context encoder only (the artifact kept
                                    after pre-training; load into a fresh
                                    ContextEncoder for downstream use)
        model_config.json        -- EncoderConfig serialized as JSON so the
                                    CLI can reconstruct the architecture without
                                    any extra flags

    Args:
        model: The trained JEPAModel.
        jepa_config: The JEPAConfig used to build the model (encoder config is extracted).
        output_dir: Directory where the files are written.
    """
    torch.save(model.state_dict(), output_dir / 'jepa_final.pt')
    torch.save(model.context_encoder.state_dict(), output_dir / 'context_encoder_final.pt')

    config_dict = dataclasses.asdict(jepa_config.encoder)
    with (output_dir / 'model_config.json').open('w', encoding='utf-8') as config_file:
        json.dump(config_dict, config_file, indent=2)

    print(f"  -- final models saved to {output_dir}")


# ----------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------

def _save_plots(
    steps: list[int],
    loss: list[float],
    reg: list[float],
    embedding_std: list[float],
    cosine_sim: list[float],
    output_dir: Path,
) -> None:
    """Save training loss and collapse metric plots to output_dir.

    Args:
        steps: Global step indices at which metrics were recorded.
        loss: MSE loss values.
        reg: VICReg regularization loss values (weighted total).
        embedding_std: Per-dimension embedding std values.
        cosine_sim: Mean pairwise cosine similarity values.
        output_dir: Directory where PNG files are written.
    """
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend, safe for scripts
    import matplotlib.pyplot as plt

    # Loss curve: MSE + reg on same axis, two lines
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(steps, loss, linewidth=1.5, label='MSE loss')
    axis.plot(steps, reg,  linewidth=1.5, label='VICReg loss', linestyle='--')
    axis.set_xlabel('Step')
    axis.set_ylabel('Loss')
    axis.set_title('JEPA Pre-training Loss')
    axis.legend(fontsize=9)
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'training_loss.png', dpi=150)
    plt.close(fig)

    # Collapse metrics
    fig, (axis_std, axis_cos) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axis_std.plot(steps, embedding_std, color='steelblue', linewidth=1.5)
    axis_std.set_ylabel('Embedding Std')
    axis_std.set_title('Collapse Monitoring')
    axis_std.grid(True, alpha=0.3)

    axis_cos.plot(steps, cosine_sim, color='tomato', linewidth=1.5)
    axis_cos.set_xlabel('Step')
    axis_cos.set_ylabel('Mean Cosine Sim')
    axis_cos.set_ylim(-1.0, 1.0)
    axis_cos.axhline(0.95, color='tomato', linestyle='--', alpha=0.5, label='collapse threshold')
    axis_cos.legend(fontsize=8)
    axis_cos.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'collapse_metrics.png', dpi=150)
    plt.close(fig)
