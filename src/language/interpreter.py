from typing import Optional, List

from .lexer import Lexer
from .lexer.tokens import Token
from .parser import Parser
from .parser.ASTNodes import ProgramNode
from .evaluator import Evaluator, InputFn, OutputFn
from .evaluator.environment import Environment, ValidRegisterType


class Interpreter:

    def __init__(
        self,
        input_fn:  Optional[InputFn]  = None,
        output_fn: Optional[OutputFn] = None,
    ):
        self._env        = Environment()
        self._input_fn   = input_fn
        self._output_fn  = output_fn
        self._evaluator  = Evaluator(
            env       = self._env,
            input_fn  = input_fn,
            output_fn = output_fn,
        )

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def tokenize(self, source: str) -> List[Token]:
        return Lexer(source).tokenize()

    def parse(self, source: str) -> ProgramNode:
        tokens = self.tokenize(source)
        return Parser(tokens).parse()

    def execute(
        self,
        source:    str,
        input_fn:  Optional[InputFn]  = None,
        output_fn: Optional[OutputFn] = None,
    ) -> Optional[ValidRegisterType]:
        """
        Exécute un programme source.
        Les input_fn/output_fn passés ici surchargent ceux du constructeur
        pour cet appel uniquement.
        """
        ast = self.parse(source)

        if input_fn is not None or output_fn is not None:
            evaluator = Evaluator(
                env       = self._env,
                input_fn  = input_fn  or self._input_fn,
                output_fn = output_fn or self._output_fn,
            )
        else:
            evaluator = self._evaluator

        return evaluator.visit(ast)

    # ── Property ──────────────────────────────────────────────────────────────

    @property
    def env(self) -> Environment:
        return self._env