"""PyTorch dataset and collation for language model fine-tuning.

Each sample is a tokenized program. No structural masking is applied --
the model learns to predict each token from the preceding context.
"""

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.tokenizer import LanguageTokenizer

from .snippet import Program


@dataclass
class LMSample:
    """A single tokenized program for language model training."""

    token_ids: torch.Tensor  # (seq_len,) int64 -- includes BOS and EOS


@dataclass
class LMBatch:
    """A padded batch of LM samples ready for model input."""

    token_ids: torch.Tensor     # (batch, seq_len) int64
    padding_mask: torch.Tensor  # (batch, seq_len) bool -- True = PAD position


class LMDataset(Dataset):
    """Dataset of tokenized programs for next-token prediction fine-tuning.

    Programs longer than max_seq_len are truncated. Programs with fewer
    than min_tokens tokens are excluded at construction time.

    Args:
        programs: List of Program objects to draw samples from.
        tokenizer: Tokenizer used to encode program source code.
        max_seq_len: Maximum sequence length. Longer programs are truncated.
        min_tokens: Minimum token count to include a program. Filters out
            trivially short sequences that carry little training signal.
    """

    def __init__(
        self,
        programs: list[Program],
        tokenizer: LanguageTokenizer,
        max_seq_len: int = 256,
        min_tokens: int = 5,
    ):
        self._samples: list[torch.Tensor] = []

        for program in programs:
            token_ids = tokenizer.encode(program.source, add_special_tokens=True)
            token_ids = token_ids[:max_seq_len]
            if len(token_ids) >= min_tokens:
                self._samples.append(torch.tensor(token_ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> LMSample:
        return LMSample(token_ids=self._samples[index])


def collate_lm_samples(samples: list[LMSample]) -> LMBatch:
    """Pad a list of LMSamples to the length of the longest sequence.

    Padding uses PAD_ID for token_ids. The padding_mask is True at padded
    positions so the model can ignore them during attention.

    Args:
        samples: A list of LMSample instances from LMDataset.__getitem__.

    Returns:
        An LMBatch with all tensors padded to the same sequence length.
    """
    pad_id = LanguageTokenizer.PAD_ID
    max_len = max(sample.token_ids.shape[0] for sample in samples)

    batched_token_ids: list[torch.Tensor] = []
    batched_padding_mask: list[torch.Tensor] = []

    for sample in samples:
        seq_len = sample.token_ids.shape[0]
        padding_needed = max_len - seq_len

        if padding_needed > 0:
            pad_tokens = torch.full((padding_needed,), pad_id, dtype=torch.long)
            batched_token_ids.append(torch.cat([sample.token_ids, pad_tokens]))
            batched_padding_mask.append(
                torch.cat([
                    torch.zeros(seq_len, dtype=torch.bool),
                    torch.ones(padding_needed, dtype=torch.bool),
                ])
            )
        else:
            batched_token_ids.append(sample.token_ids)
            batched_padding_mask.append(torch.zeros(seq_len, dtype=torch.bool))

    return LMBatch(
        token_ids=torch.stack(batched_token_ids),
        padding_mask=torch.stack(batched_padding_mask),
    )
