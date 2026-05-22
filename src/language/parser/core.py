from typing import List
from src.language.lexer import TokenType, Token
from .base import BaseParser
from .ASTNodes import *

from .expression import _ExpressionParser
from .statement import _StatementParser

# ─── Parser principal ─────────────────────────────────────────────────────────

class Parser(BaseParser):

    def __init__(self, tokens: List[Token]):
        super().__init__(tokens)
        self.expr = _ExpressionParser(self)
        self.stmt = _StatementParser(self)

    def parse(self) -> ProgramNode:
        statements = []
        self.skip_newlines()
        while self.current_token and self.current_token.type != TokenType.EOF:
            statements.append(self.stmt.parse())
            self.skip_newlines()
        return ProgramNode(statements=statements, position=(1, 1))
    
    # ─── Méthodes Helpers ─────────────────────────────────────────────────────
    
    def __repr__(self):
        return f"<Parser position={self.position} current_token={self.current_token}>"