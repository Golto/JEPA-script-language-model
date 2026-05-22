from src.language.parser.ASTNodes import *

from .exception import LanguageExecutionError
from .base import _BaseEvaluator


MAX_LOOP_ITERATIONS = 2048


# ─── Signaux de contrôle de flux (non exposés) ────────────────────────────────

class _BreakSignal(Exception):
    pass

class _ContinueSignal(Exception):
    pass


# ─── StatementEvaluator ───────────────────────────────────────────────────────

class _StatementEvaluator(_BaseEvaluator):

    def visit_AssignmentNode(self, node: AssignmentNode) -> None:
        value = self.evaluator.visit(node.value)
        self.evaluator.env.set(node.target.value, value)


    def visit_IfNode(self, node: IfNode) -> None:
        condition = self.evaluator.visit(node.condition)
        if not isinstance(condition, bool):
            raise LanguageExecutionError(
                f"La condition d'un 'if' doit être booléenne, obtenu {type(condition).__name__}", node
            )
        if condition:
            for stmt in node.then_block:
                self.evaluator.visit(stmt)
        elif node.else_block is not None:
            for stmt in node.else_block:
                self.evaluator.visit(stmt)


    def visit_WhileNode(self, node: WhileNode) -> None:
        iterations = 0
        while True:
            if iterations >= MAX_LOOP_ITERATIONS:
                raise LanguageExecutionError(
                    f"Boucle infinie détectée : limite de {MAX_LOOP_ITERATIONS} itérations atteinte", node
                )
            condition = self.evaluator.visit(node.condition)
            if not isinstance(condition, bool):
                raise LanguageExecutionError(
                    f"La condition d'un 'while' doit être booléenne, obtenu {type(condition).__name__}", node
                )
            if not condition:
                break
            try:
                for stmt in node.body:
                    self.evaluator.visit(stmt)
            except _BreakSignal:
                break
            except _ContinueSignal:
                continue
            finally:
                iterations += 1
    

    def visit_BreakNode(self, node: BreakNode) -> None:
        raise _BreakSignal()
    

    def visit_ContinueNode(self, node: ContinueNode) -> None:
        raise _ContinueSignal()
    

    def visit_InputNode(self, node: InputNode) -> None:
        value = self.evaluator.input_fn(node.target.value)
        self.evaluator.env.set(node.target.value, value)


    def visit_OutputNode(self, node: OutputNode) -> None:
        value = self.evaluator.visit(node.value)
        self.evaluator.output_fn(value)