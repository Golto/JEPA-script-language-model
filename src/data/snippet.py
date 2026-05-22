"""Loading and structuring of snippet data files.

Each snippet file contains one or more programs. Each program begins with a
spec comment in the format:

    // name: description [param: type, ...] -> return_type

followed by the source code of that program.
"""

import re
from dataclasses import dataclass
from pathlib import Path


_SPEC_PATTERN = re.compile(r'^// (\w+): (.+?) \[(.+?)\] -> (.+)$')


@dataclass
class ParamSpec:
    """A single typed parameter from a program spec."""

    name: str
    type_name: str


@dataclass
class Program:
    """A single named program extracted from a snippet file."""

    name: str
    description: str
    params: list[ParamSpec]
    return_type: str
    source: str


@dataclass
class Snippet:
    """A collection of programs loaded from a single snippet file."""

    path: Path
    programs: list[Program]


# ----------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------

def _parse_spec_line(line: str) -> tuple[str, str, list[ParamSpec], str]:
    """Parse a spec comment line into its components.

    Expected format: // name: description [param: type, ...] -> return_type

    Args:
        line: A spec comment line from a snippet file.

    Returns:
        A tuple of (name, description, params, return_type).

    Raises:
        ValueError: If the line does not match the expected format.
    """
    match = _SPEC_PATTERN.match(line)
    if not match:
        raise ValueError(f"Invalid spec line format: '{line}'")

    name, description, params_raw, return_type = match.groups()

    params: list[ParamSpec] = []
    for param_token in params_raw.split(', '):
        param_name, param_type = param_token.split(': ', 1)
        params.append(ParamSpec(name=param_name.strip(), type_name=param_type.strip()))

    return name, description, params, return_type.strip()


# ----------------------------------------------------------------
# Loading
# ----------------------------------------------------------------

def load_snippet(path: Path) -> Snippet:
    """Load and parse a single snippet file into structured programs.

    Args:
        path: Path to the snippet file.

    Returns:
        A Snippet containing all programs parsed from the file.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If a spec line cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Snippet file not found: {path}")

    lines = path.read_text(encoding='utf-8').splitlines()

    programs: list[Program] = []
    current_spec: tuple[str, str, list[ParamSpec], str] | None = None
    current_source_lines: list[str] = []

    for line in lines:
        if line.startswith('//'):
            if current_spec is not None:
                name, description, params, return_type = current_spec
                programs.append(Program(
                    name=name,
                    description=description,
                    params=params,
                    return_type=return_type,
                    source='\n'.join(current_source_lines).strip(),
                ))
            current_spec = _parse_spec_line(line)
            current_source_lines = []
        else:
            current_source_lines.append(line)

    if current_spec is not None:
        name, description, params, return_type = current_spec
        programs.append(Program(
            name=name,
            description=description,
            params=params,
            return_type=return_type,
            source='\n'.join(current_source_lines).strip(),
        ))

    return Snippet(path=path, programs=programs)


def load_all_snippets(directory: Path) -> list[Snippet]:
    """Load all snippet files from a directory, sorted by file name.

    Args:
        directory: Path to the directory containing snippet files.

    Returns:
        A list of Snippet objects, one per file, sorted by name.

    Raises:
        FileNotFoundError: If directory does not exist.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Snippet directory not found: {directory}")

    snippet_paths = sorted(directory.glob('snippet_*'))
    return [load_snippet(path) for path in snippet_paths]
