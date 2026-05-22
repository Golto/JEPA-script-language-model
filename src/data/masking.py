"""Structural mask generation for JEPA pre-training.

Masks target syntactically meaningful regions rather than random spans,
forcing the predictor to reason about program structure:
    - Body of if/then blocks (between 'then' and 'else'/'endif')
    - Body of else blocks (between 'else' and 'endif')
    - Body of while/do loops (between 'do' and 'endwhile')

Random token-level masking is deliberately avoided.
"""

import random

import torch


# ----------------------------------------------------------------
# Block detection
# ----------------------------------------------------------------

_BLOCK_OPENERS: frozenset[str] = frozenset({'then', 'do'})
_BLOCK_CLOSERS: frozenset[str] = frozenset({'endif', 'endwhile'})
_BLOCK_SPLITTERS: frozenset[str] = frozenset({'else'})
_SPECIAL_TOKEN_PREFIX = '<'


def find_maskable_blocks(tokens: list[str]) -> list[tuple[int, int]]:
    """Find (start, end) inclusive index ranges for maskable structural blocks.

    Works at any nesting depth. Nested blocks are reported independently,
    so both an outer if-body and an inner while-body are candidates.

    Args:
        tokens: List of token strings for a single sequence (may include
            special tokens like BOS/EOS, which are automatically skipped).

    Returns:
        A list of (start_index, end_index) pairs (inclusive). Only non-empty
        ranges (start <= end) are included.
    """
    blocks: list[tuple[int, int]] = []
    stack: list[int] = []  # pending block start indices

    for index, token in enumerate(tokens):
        if token.startswith(_SPECIAL_TOKEN_PREFIX):
            continue
        if token in _BLOCK_OPENERS:
            stack.append(index + 1)
        elif token in _BLOCK_SPLITTERS:
            if stack:
                block_start = stack.pop()
                if block_start <= index - 1:
                    blocks.append((block_start, index - 1))
                stack.append(index + 1)
        elif token in _BLOCK_CLOSERS:
            if stack:
                block_start = stack.pop()
                if block_start <= index - 1:
                    blocks.append((block_start, index - 1))

    return blocks


# ----------------------------------------------------------------
# Mask sampling
# ----------------------------------------------------------------

def sample_structural_mask(
    tokens: list[str],
    rng: random.Random | None = None,
) -> torch.Tensor:
    """Sample a structural boolean mask for a token sequence.

    Randomly selects one maskable structural block and marks all its token
    positions as True. If the sequence contains no structural blocks (e.g.
    a one-liner), the returned mask is all False.

    Args:
        tokens: List of token strings for a single sequence.
        rng: Random number generator. Uses a default instance if None.

    Returns:
        Boolean tensor of shape (len(tokens),). True = masked position.
    """
    if rng is None:
        rng = random.Random()

    mask = torch.zeros(len(tokens), dtype=torch.bool)
    blocks = find_maskable_blocks(tokens)

    if not blocks:
        return mask

    start, end = rng.choice(blocks)
    mask[start:end + 1] = True
    return mask
