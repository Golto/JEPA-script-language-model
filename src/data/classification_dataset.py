"""PyTorch dataset and collation for program classification fine-tuning.

Each sample is a tokenized program paired with an integer class label derived
from program metadata or source code structure (e.g. has_loop, return_type).
"""

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from src.data.snippet import Program
from src.tasks.classification.labels import LabelType, build_label_vocab, extract_label_key
from src.tokenizer import LanguageTokenizer


@dataclass
class ClassificationSample:
    """A single tokenized program with its integer class label."""

    token_ids: torch.Tensor  # (seq_len,) int64 -- includes BOS and EOS
    label: int


@dataclass
class ClassificationBatch:
    """A padded batch of classification samples ready for model input."""

    token_ids: torch.Tensor     # (batch, seq_len) int64
    padding_mask: torch.Tensor  # (batch, seq_len) bool -- True = PAD position
    labels: torch.Tensor        # (batch,) int64


class ClassificationDataset(Dataset):
    """Dataset of labeled programs for sequence classification fine-tuning.

    Labels are derived automatically from program metadata or source code
    according to the chosen LabelType. The label vocabulary maps label strings
    to integer indices and is built from the provided programs unless an
    external vocabulary is supplied (e.g. for a val set that must share the
    same mapping as the train set).

    Args:
        programs: List of Program objects.
        tokenizer: Tokenizer used to encode program source code.
        label_type: Which program property to use as the classification target.
        max_seq_len: Maximum sequence length. Longer programs are truncated.
        min_tokens: Minimum token count to include a sample.
        label_vocab: Optional pre-built label vocabulary. If None, one is
            built from the provided programs.
    """

    def __init__(
        self,
        programs: list[Program],
        tokenizer: LanguageTokenizer,
        label_type: LabelType,
        max_seq_len: int = 256,
        min_tokens: int = 5,
        label_vocab: dict[str, int] | None = None,
    ):
        self._label_vocab = label_vocab or build_label_vocab(programs, label_type)
        self._samples: list[ClassificationSample] = []

        for program in programs:
            token_ids = tokenizer.encode(program.source, add_special_tokens=True)
            token_ids = token_ids[:max_seq_len]
            if len(token_ids) < min_tokens:
                continue

            label_key = extract_label_key(program, label_type)
            if label_key not in self._label_vocab:
                continue  # skip labels unseen in the train vocab

            self._samples.append(ClassificationSample(
                token_ids=torch.tensor(token_ids, dtype=torch.long),
                label=self._label_vocab[label_key],
            ))

    @property
    def n_classes(self) -> int:
        """Number of distinct classes in the label vocabulary."""
        return len(self._label_vocab)

    @property
    def label_vocab(self) -> dict[str, int]:
        """Mapping from label string to integer class index."""
        return dict(self._label_vocab)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> ClassificationSample:
        return self._samples[index]


def collate_classification_samples(
    samples: list[ClassificationSample],
) -> ClassificationBatch:
    """Pad a list of ClassificationSamples to the length of the longest sequence.

    Padding uses PAD_ID for token_ids. The padding_mask is True at padded
    positions so the model can ignore them during attention.

    Args:
        samples: A list of ClassificationSample instances from ClassificationDataset.

    Returns:
        A ClassificationBatch with all tensors padded to the same sequence length.
    """
    pad_id = LanguageTokenizer.PAD_ID
    max_len = max(sample.token_ids.shape[0] for sample in samples)

    batched_ids: list[torch.Tensor] = []
    batched_masks: list[torch.Tensor] = []
    batched_labels: list[int] = []

    for sample in samples:
        seq_len = sample.token_ids.shape[0]
        padding_needed = max_len - seq_len

        if padding_needed > 0:
            pad_tokens = torch.full((padding_needed,), pad_id, dtype=torch.long)
            batched_ids.append(torch.cat([sample.token_ids, pad_tokens]))
            batched_masks.append(torch.cat([
                torch.zeros(seq_len, dtype=torch.bool),
                torch.ones(padding_needed, dtype=torch.bool),
            ]))
        else:
            batched_ids.append(sample.token_ids)
            batched_masks.append(torch.zeros(seq_len, dtype=torch.bool))

        batched_labels.append(sample.label)

    return ClassificationBatch(
        token_ids=torch.stack(batched_ids),
        padding_mask=torch.stack(batched_masks),
        labels=torch.tensor(batched_labels, dtype=torch.long),
    )
