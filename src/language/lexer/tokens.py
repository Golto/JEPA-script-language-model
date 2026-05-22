from typing import Tuple, Optional
from enum import Enum
from dataclasses import dataclass

# ─── Types de tokens ──────────────────────────────────────────────────────────

class TokenType(Enum):
    # Data types
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    BOOLEAN = "BOOLEAN"
    # Register
    REGISTER = "REGISTER"
    # Operators
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    POWER = "POWER"
    MOD = "MOD"
    # Logical operators
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    # Comparison
    EQ = "EQ"
    NEQ = "NEQ"
    LT = "LT"
    GT = "GT"
    LTE = "LTE"
    GTE = "GTE"
    # Assignment
    ASSIGN = "ASSIGN"
    # Delimiters
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    SEMICOLON = "SEMICOLON"
    # Keywords
    IF = "IF"
    THEN = "THEN"
    ELSE = "ELSE"
    ENDIF = "ENDIF"
    WHILE = "WHILE"
    DO = "DO"
    ENDWHILE = "ENDWHILE"
    BREAK = "BREAK"
    CONTINUE = "CONTINUE"
    # I/O
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    # Special
    EOF = "EOF"
    NEWLINE = "NEWLINE"

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.value}>"


# ─── Token ────────────────────────────────────────────────────────────────────

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

    def __post_init__(self):
        if not isinstance(self.type, TokenType):
            raise ValueError("Token type must be a TokenType enum member")
    
    @property
    def position(self) -> Tuple[int, int]:
        return (self.line, self.column)
