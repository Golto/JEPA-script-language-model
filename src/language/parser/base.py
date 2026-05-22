from src.language.lexer import TokenType, Token
from .ASTNodes import *
from typing import List, Optional

# ─── Exception ────────────────────────────────────────────────────────────────

class LanguageSyntaxicalError(Exception):
    """Erreur de syntaxe détectée pendant l'analyse syntaxique"""
    def __init__(self, message: str, token=None):
        self.message = message
        self.token = token
        if token:
            super().__init__(f"Erreur de syntaxe à la ligne {token.line}, colonne {token.column}: {message}")
        else:
            super().__init__(f"Erreur de syntaxe: {message}")


# ─── Analyseur syntaxique ─────────────────────────────────────────────────────

class BaseParser:

    BINARY_PRECEDENCE = {
        TokenType.OR: 1,
        TokenType.AND: 2,
        TokenType.EQ: 3, TokenType.NEQ: 3,
        TokenType.LT: 4, TokenType.GT: 4, TokenType.LTE: 4, TokenType.GTE: 4,
        TokenType.ADD: 5, TokenType.SUB: 5,
        TokenType.MUL: 6, TokenType.DIV: 6, TokenType.MOD: 6,
    }

    BINARY_TOKEN_TO_ENUM = {
        TokenType.ADD: BinaryOperatorType.ADD,
        TokenType.SUB: BinaryOperatorType.SUB,
        TokenType.MUL: BinaryOperatorType.MUL,
        TokenType.DIV: BinaryOperatorType.DIV,
        TokenType.MOD: BinaryOperatorType.MOD,
        TokenType.AND: BinaryOperatorType.AND,
        TokenType.OR: BinaryOperatorType.OR,
        TokenType.EQ: BinaryOperatorType.EQ,
        TokenType.NEQ: BinaryOperatorType.NEQ,
        TokenType.LT: BinaryOperatorType.LT,
        TokenType.GT: BinaryOperatorType.GT,
        TokenType.LTE: BinaryOperatorType.LTE,
        TokenType.GTE: BinaryOperatorType.GTE,
    }

    UNARY_TOKEN_TO_ENUM = {
        TokenType.ADD: UnaryOperatorType.POSITIVE,
        TokenType.SUB: UnaryOperatorType.NEGATIVE,
        TokenType.NOT: UnaryOperatorType.NOT,
    }

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        self.current_token = tokens[0] if tokens else None
        self.loop_depth = 0 # loop context (for break/continue)


    # ─── Méthodes communes ────────────────────────────────────────────────────

    def advance(self):
        """Avance au token suivant"""
        self.position += 1
        if self.position < len(self.tokens):
            self.current_token = self.tokens[self.position]
        else:
            self.current_token = None


    def peek(self, offset: int = 1) -> Optional[Token]:
        """Regarde le token à une position future"""
        peek_position = self.position + offset
        return self.tokens[peek_position] if peek_position < len(self.tokens) else None
    

    def expect(self, token_type) -> Token:
        """Vérifie que le token courant est du type attendu"""
        if not self.current_token or self.current_token.type != token_type:
            raise LanguageSyntaxicalError(
                f"Attendu {token_type.value}, obtenu {self.current_token.type.value if self.current_token else 'EOF'}",
                self.current_token
            )
        token = self.current_token
        self.advance()
        return token
    

    def skip_newlines(self):
        """Ignore les retours à la ligne"""
        while self.current_token and self.current_token.type == TokenType.NEWLINE:
            self.advance()