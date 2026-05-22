from typing import List
from src.language.lexer import TokenType
from .base import LanguageSyntaxicalError
from .ASTNodes import *

from .protocol import IParser
from .expression import _ExpressionParser

# ─── StatementParser ──────────────────────────────────────────────────────────

class _StatementParser:
    """Gère le parsing des instructions."""

    def __init__(self, parser: IParser):
        self.parser = parser
        self.expression_parser: _ExpressionParser = self.parser.expr

    def parse_block(self, *terminators: TokenType) -> List[ASTNode]:
        """Parse une suite d'instructions jusqu'à l'un des terminateurs."""
        stmts = []
        self.parser.skip_newlines()
        while self.parser.current_token and self.parser.current_token.type not in terminators:
            stmts.append(self.parse())
            self.parser.skip_newlines()
        return stmts


    def parse(self) -> ASTNode:
        token = self.parser.current_token
        if token is None:
            raise LanguageSyntaxicalError("Instruction attendue, obtenu EOF")

        if token.type == TokenType.IF:
            return self._parse_if()
        if token.type == TokenType.WHILE:
            return self._parse_while()
        if token.type == TokenType.BREAK:
            return self._parse_break()
        if token.type == TokenType.CONTINUE:
            return self._parse_continue()
        if token.type == TokenType.INPUT:
            return self._parse_input()
        if token.type == TokenType.OUTPUT:
            return self._parse_output()

        # Assignation : REGISTER ASSIGN expr
        if token.type == TokenType.REGISTER:
            next_token = self.parser.peek()
            if next_token and next_token.type == TokenType.ASSIGN:
                return self._parse_assignment()

        # Expression seule (r0, 1+2, true, etc.)
        node = self.expression_parser.parse(0)
        self._expect_end_of_statement()
        return node
    

    # ── Instructions ──────────────────────────────────────────────────────────

    def _parse_assignment(self) -> AssignmentNode:
        token = self.parser.current_token
        target = RegisterNode.from_token(token, value=token.value)
        self.parser.advance() # consomme le registre
        self.parser.expect(TokenType.ASSIGN)
        value = self.expression_parser.parse(0)
        self._expect_end_of_statement()
        return AssignmentNode.from_token(token, target=target, value=value)


    def _parse_if(self) -> IfNode:
        token = self.parser.expect(TokenType.IF)
        condition = self.expression_parser.parse(0)
        self.parser.expect(TokenType.THEN)
        self.parser.skip_newlines()

        then_block = self.parse_block(TokenType.ELSE, TokenType.ENDIF)
        else_block = None

        if self.parser.current_token and self.parser.current_token.type == TokenType.ELSE:
            self.parser.advance()
            self.parser.skip_newlines()
            else_block = self.parse_block(TokenType.ENDIF)

        self.parser.expect(TokenType.ENDIF)
        self._expect_end_of_statement()
        return IfNode.from_token(token, condition=condition, then_block=then_block, else_block=else_block)


    def _parse_while(self) -> WhileNode:
        token = self.parser.expect(TokenType.WHILE)
        condition = self.expression_parser.parse(0)
        self.parser.expect(TokenType.DO)
        self.parser.skip_newlines()

        self.parser.loop_depth += 1
        body = self.parse_block(TokenType.ENDWHILE)
        self.parser.loop_depth -= 1

        self.parser.expect(TokenType.ENDWHILE)
        self._expect_end_of_statement()
        return WhileNode.from_token(token, condition=condition, body=body)


    def _parse_break(self) -> BreakNode:
        token = self.parser.current_token
        if self.parser.loop_depth == 0:
            raise LanguageSyntaxicalError("'break' en dehors d'une boucle", token)
        self.parser.advance()
        self._expect_end_of_statement()
        return BreakNode.from_token(token)
    

    def _parse_continue(self) -> ContinueNode:
        token = self.parser.current_token
        if self.parser.loop_depth == 0:
            raise LanguageSyntaxicalError("'continue' en dehors d'une boucle", token)
        self.parser.advance()
        self._expect_end_of_statement()
        return ContinueNode.from_token(token)
    

    def _parse_input(self) -> InputNode:
        token = self.parser.expect(TokenType.INPUT)
        register_token = self.parser.expect(TokenType.REGISTER)
        target = RegisterNode.from_token(register_token, value=register_token.value)
        self._expect_end_of_statement()
        return InputNode.from_token(token, target=target)


    def _parse_output(self) -> OutputNode:
        token = self.parser.expect(TokenType.OUTPUT)
        value = self.expression_parser.parse(0)
        self._expect_end_of_statement()
        return OutputNode.from_token(token, value=value)


    # ── Helpers ───────────────────────────────────────────────────────────────

    def _expect_end_of_statement(self):
        """Consomme un NEWLINE ou SEMICOLON de fin d'instruction, ou accepte EOF."""
        token = self.parser.current_token
        if token is None or token.type == TokenType.EOF:
            return
        if token.type in (TokenType.NEWLINE, TokenType.SEMICOLON):
            self.parser.advance()
            return
        # Fin de bloc implicite (endif, endwhile, else), on ne consomme pas
        if token.type in (TokenType.ENDIF, TokenType.ENDWHILE, TokenType.ELSE):
            return
        raise LanguageSyntaxicalError(
            f"Fin d'instruction attendue, obtenu '{token.value}'", token
        )