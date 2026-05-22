from src.language.lexer import TokenType
from .base import LanguageSyntaxicalError
from .ASTNodes import *

from .protocol import IParser


# ─── ExpressionParser ─────────────────────────────────────────────────────────

class _ExpressionParser:
    """Gère le parsing des expressions via Pratt / precedence climbing."""

    def __init__(self, parser: IParser):
        self.parser = parser


    def parse(self, min_prec: int = 0) -> ASTNode:
        left = self._parse_unary()

        while True:
            token = self.parser.current_token
            if token is None or token.type not in self.parser.BINARY_PRECEDENCE:
                break

            prec = self.parser.BINARY_PRECEDENCE[token.type]
            if prec <= min_prec:
                break

            op = self.parser.BINARY_TOKEN_TO_ENUM[token.type]
            op_pos = token.position
            self.parser.advance()
            right = self.parse(prec)  # left-associative : prec (pas prec-1)
            left = BinaryOpNode(left=left, operator=op, right=right, position=op_pos)
        
        return left
    

    def _parse_unary(self) -> ASTNode:
        token = self.parser.current_token
        
        if token and token.type in self.parser.UNARY_TOKEN_TO_ENUM:
            op = self.parser.UNARY_TOKEN_TO_ENUM[token.type]
            self.parser.advance()
            operand = self._parse_unary()  # récursif pour `-(-r0)`
            return UnaryOpNode(operator=op, operand=operand, position=token.position)
        
        return self._parse_primary()


    def _parse_primary(self) -> ASTNode:
        token = self.parser.current_token
        if token is None:
            raise LanguageSyntaxicalError("Expression attendue, obtenu EOF")

        # Parenthèses groupées
        if token.type == TokenType.LPAREN:
            self.parser.advance()
            node = self.parse(0)
            self.parser.expect(TokenType.RPAREN)
            return node

        # Littéraux numériques
        if token.type == TokenType.INTEGER:
            self.parser.advance()
            return NumberNode.from_token(token, value=int(token.value), type=NumberType.INTEGER)

        if token.type == TokenType.FLOAT:
            self.parser.advance()
            return NumberNode.from_token(token, value=float(token.value), type=NumberType.FLOAT)

        # Booléens
        if token.type == TokenType.BOOLEAN:
            self.parser.advance()
            return BooleanNode.from_token(token, value=(token.value == 'true'))

        # Registres
        if token.type == TokenType.REGISTER:
            self.parser.advance()
            return RegisterNode.from_token(token, value=token.value)

        raise LanguageSyntaxicalError(
            f"Token inattendu dans une expression : '{token.value}'", token
        )