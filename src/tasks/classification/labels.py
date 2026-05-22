"""Label types and extraction for program classification.

Each LabelType defines how to derive a class label from a Program.
Binary label types produce '0' or '1'. Multiclass types (e.g. RETURN_TYPE)
produce the natural string value from the program spec.

The label vocabulary maps those strings to integer class indices and is
built from the training programs so the mapping is stable across runs.
"""

from enum import Enum

from src.data.snippet import Program


class LabelType(Enum):
    """Supported program classification targets.

    Binary (n_classes=2):
        HAS_LOOP          -- True if the source contains a while loop.
        HAS_CONDITIONAL   -- True if the source contains an if statement.
        HAS_INPUT         -- True if the source reads from an input register.

    Multiclass:
        RETURN_TYPE       -- The return type string from the program spec
                             (e.g. 'int', 'float', 'bool').
    """

    HAS_LOOP = 'has_loop'
    HAS_CONDITIONAL = 'has_conditional'
    HAS_INPUT = 'has_input'
    RETURN_TYPE = 'return_type'

    @property
    def is_binary(self) -> bool:
        """True if this label type produces binary (0/1) labels."""
        return self in {LabelType.HAS_LOOP, LabelType.HAS_CONDITIONAL, LabelType.HAS_INPUT}


def extract_label_key(program: Program, label_type: LabelType) -> str:
    """Extract a raw string key from a program for the given label type.

    Binary label types return '0' or '1'. Multiclass types return the
    natural string representation from the program spec.

    Args:
        program: The program to extract a label from.
        label_type: Which program property to use as the label.

    Returns:
        A string key suitable for label vocab mapping.

    Raises:
        ValueError: If label_type is not a recognized LabelType member.
    """
    if label_type == LabelType.HAS_LOOP:
        return '1' if 'while' in program.source else '0'
    if label_type == LabelType.HAS_CONDITIONAL:
        return '1' if 'if' in program.source else '0'
    if label_type == LabelType.HAS_INPUT:
        return '1' if 'input' in program.source else '0'
    if label_type == LabelType.RETURN_TYPE:
        return program.return_type
    raise ValueError(f"Unknown label type: {label_type}")


def build_label_vocab(programs: list[Program], label_type: LabelType) -> dict[str, int]:
    """Build a deterministic label string to integer mapping from a list of programs.

    Labels are sorted alphabetically for reproducibility across runs.

    Args:
        programs: Programs from which to collect distinct label keys.
        label_type: Which label type defines the vocabulary.

    Returns:
        A dict mapping each distinct label string to a unique integer index.
    """
    keys = sorted({extract_label_key(program, label_type) for program in programs})
    return {key: index for index, key in enumerate(keys)}
