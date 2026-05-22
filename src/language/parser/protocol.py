from typing import Protocol, Optional
from src.language.lexer import Token, TokenType

class IParser(Protocol):
    current_token: Optional[Token]
    loop_depth: int
    BINARY_PRECEDENCE: dict
    BINARY_TOKEN_TO_ENUM: dict
    UNARY_TOKEN_TO_ENUM: dict

    def advance(self) -> None: ...
    def peek(self, offset: int = 1) -> Optional[Token]: ...
    def expect(self, token_type: TokenType) -> Token: ...
    def skip_newlines(self) -> None: ...