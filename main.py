"""Entry point for the JEPA CLI.

Run with uv:
    uv run main.py embed   --model PATH [PROGRAM] [-f FILE] [--save PATH]
    uv run main.py compare --model PATH [PROGRAM_A] [PROGRAM_B] [-f FILE]...
    uv run main.py train   --data DIR [options]

Run uv run main.py --help or uv run main.py <command> --help for full details.
"""

from src.cli import main

if __name__ == '__main__':
    main()
