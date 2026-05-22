"""PyTorch dataset and collation for JEPA pre-training.

Each sample is a tokenized program paired with a structural mask identifying
which positions the context encoder should predict. Sequences are padded to
a uniform length within each batch.
"""

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.tokenizer import LanguageTokenizer

from .masking import sample_structural_mask
from .snippet import Program


@dataclass
class JEPASample:
    """A single (unpadded) tokenized program ready for JEPA training."""

    token_ids: torch.Tensor       # (seq_len,) int64 -- includes BOS and EOS
    structural_mask: torch.Tensor # (seq_len,) bool  -- True = masked position


@dataclass
class JEPABatch:
    """A padded batch of JEPA samples ready for model input."""

    token_ids: torch.Tensor       # (batch, seq_len) int64
    structural_mask: torch.Tensor # (batch, seq_len) bool
    padding_mask: torch.Tensor    # (batch, seq_len) bool -- True = PAD position


class JEPADataset(Dataset):
    """Dataset of tokenized programs with lazily sampled structural masks.

    The structural mask is re-sampled on every call to __getitem__, acting
    as a form of data augmentation across epochs. Programs longer than
    max_seq_len are truncated (keeping BOS and truncating before EOS).
    Programs with fewer than min_tokens tokens are excluded at construction.

    Args:
        programs: List of Program objects to draw samples from.
        tokenizer: Tokenizer used to encode program source code.
        max_seq_len: Maximum sequence length. Longer programs are truncated.
        min_tokens: Minimum number of tokens (after encoding) to include a
            program. Filters out trivially short sequences.
        n_blocks: Number of structural blocks to mask per sample. Increasing
            this value makes the pre-training task harder and reduces the risk
            of representation collapse.
    """

    def __init__(
        self,
        programs: list[Program],
        tokenizer: LanguageTokenizer,
        max_seq_len: int = 256,
        min_tokens: int = 5,
        n_blocks: int = 1,
    ):
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._n_blocks = n_blocks
        self._rng = random.Random()

        # Encode all programs up-front and filter by minimum length.
        self._samples: list[torch.Tensor] = []
        for program in programs:
            token_ids = tokenizer.encode(program.source, add_special_tokens=True)
            token_ids = token_ids[:max_seq_len]
            if len(token_ids) >= min_tokens:
                self._samples.append(torch.tensor(token_ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> JEPASample:
        token_ids = self._samples[index]
        tokens = self._tokenizer.decode(token_ids.tolist(), skip_special_tokens=False)
        structural_mask = sample_structural_mask(tokens, n_blocks=self._n_blocks, rng=self._rng)
        return JEPASample(token_ids=token_ids, structural_mask=structural_mask)


def collate_jepa_samples(samples: list[JEPASample]) -> JEPABatch:
    """Pad a list of JEPASamples to the length of the longest sequence.

    Padding uses PAD_ID for token_ids and True for padding_mask. The
    structural_mask is padded with False (padding positions are never masked).

    Args:
        samples: A list of JEPASample instances from JEPADataset.__getitem__.

    Returns:
        A JEPABatch with all tensors padded to the same sequence length.
    """
    pad_id = LanguageTokenizer.PAD_ID
    max_len = max(sample.token_ids.shape[0] for sample in samples)

    batched_token_ids: list[torch.Tensor] = []
    batched_structural_mask: list[torch.Tensor] = []
    batched_padding_mask: list[torch.Tensor] = []

    for sample in samples:
        seq_len = sample.token_ids.shape[0]
        padding_needed = max_len - seq_len

        if padding_needed > 0:
            pad_tokens = torch.full((padding_needed,), pad_id, dtype=torch.long)
            pad_false = torch.zeros(padding_needed, dtype=torch.bool)
            pad_true = torch.ones(padding_needed, dtype=torch.bool)
            batched_token_ids.append(torch.cat([sample.token_ids, pad_tokens]))
            batched_structural_mask.append(torch.cat([sample.structural_mask, pad_false]))
            batched_padding_mask.append(torch.cat([torch.zeros(seq_len, dtype=torch.bool), pad_true]))
        else:
            batched_token_ids.append(sample.token_ids)
            batched_structural_mask.append(sample.structural_mask)
            batched_padding_mask.append(torch.zeros(seq_len, dtype=torch.bool))

    return JEPABatch(
        token_ids=torch.stack(batched_token_ids),
        structural_mask=torch.stack(batched_structural_mask),
        padding_mask=torch.stack(batched_padding_mask),
    )
