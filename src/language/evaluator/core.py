from typing import Optional, Callable, Dict, Self

from src.language.parser.ASTNodes import *

from .exception import LanguageExecutionError
from .environment import ValidRegisterType, Environment

from .base import _BaseEvaluator
from .expression import _ExpressionEvaluator
from .statement import _StatementEvaluator

# ─── Evaluator ────────────────────────────────────────────────────────────────


InputFn  = Callable[[str], ValidRegisterType]
OutputFn = Callable[[ValidRegisterType], None]


class Evaluator:

    def __init__(
        self,
        env: Optional[Environment] = None,
        input_fn: Optional[InputFn] = None,
        output_fn: Optional[OutputFn] = None,
    ):
        self.env = env or Environment()
        self.input_fn = input_fn or self._default_input
        self.output_fn = output_fn or self._default_output

        self._expr = _ExpressionEvaluator(self)
        self._stmt = _StatementEvaluator(self)

        # Dispatch table : routing vers le bon sous-évaluateur
        self._dispatch: Dict[type, Union[Self, _BaseEvaluator]] = {
            ProgramNode:    self,
            NumberNode:     self._expr,
            BooleanNode:    self._expr,
            RegisterNode:   self._expr,
            UnaryOpNode:    self._expr,
            BinaryOpNode:   self._expr,
            AssignmentNode: self._stmt,
            IfNode:         self._stmt,
            WhileNode:      self._stmt,
            BreakNode:      self._stmt,
            ContinueNode:   self._stmt,
            InputNode:      self._stmt,
            OutputNode:     self._stmt,
        }
    
    
    # ── Visit ─────────────────────────────────────────────────────────────────

    def visit(self, node: ASTNode) -> Optional[ValidRegisterType]:
        handler = self._dispatch.get(type(node))
        if handler is None:
            raise LanguageExecutionError(f"Node non supporté : {type(node).__name__}")
        method_name = f"visit_{type(node).__name__}"
        return getattr(handler, method_name)(node)

    def visit_ProgramNode(self, node: ProgramNode) -> Optional[ValidRegisterType]:
        result = None
        for statement in node.statements:
            result = self.visit(statement)
        return result
    

    # ── I/O par défaut ────────────────────────────────────────────────────────

    @staticmethod
    def _default_input(register_name: str) -> ValidRegisterType:
        raw = input(f"input {register_name} > ")
        try:
            if '.' in raw:
                return float(raw)
            if raw in ('true', 'false'):
                return raw == 'true'
            return int(raw)
        except ValueError:
            raise LanguageExecutionError(f"Valeur d'entrée invalide : '{raw}'")

    @staticmethod
    def _default_output(value: ValidRegisterType) -> None:
        print(value)