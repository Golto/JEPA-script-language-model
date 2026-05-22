from typing import Union

from src.language.parser.ASTNodes import *

from .environment import ValidRegisterType
from .exception import LanguageExecutionError
from .base import _BaseEvaluator


MAX_INT_VALUE       = 2 ** 63
MAX_FLOAT_VALUE     = 1e308


def _check_value_bounds(value: ValidRegisterType, node) -> None:
    if isinstance(value, int) and abs(value) > MAX_INT_VALUE:
        raise LanguageExecutionError(
            f"Dépassement entier : valeur {value} hors des bornes "
            f"[-{MAX_INT_VALUE}, {MAX_INT_VALUE}]", node
        )
    if isinstance(value, float) and (abs(value) > MAX_FLOAT_VALUE or value != value): # NaN != Nan (true)
        raise LanguageExecutionError(
            f"Dépassement flottant : valeur hors des bornes", node
        )
    
# ─── ExpressionEvaluator ──────────────────────────────────────────────────────

class _ExpressionEvaluator(_BaseEvaluator):
    
    def visit_NumberNode(self, node: NumberNode) -> Union[int, float]:
        return node.value


    def visit_BooleanNode(self, node: BooleanNode) -> bool:
        return node.value
    

    def visit_RegisterNode(self, node: RegisterNode) -> ValidRegisterType:
        return self.evaluator.env.get(node.value)
    

    def visit_UnaryOpNode(self, node: UnaryOpNode) -> ValidRegisterType:
        operand = self.evaluator.visit(node.operand)
        match node.operator:
            case UnaryOperatorType.POSITIVE:
                return +operand
            case UnaryOperatorType.NEGATIVE:
                return -operand
            case UnaryOperatorType.NOT:
                if not isinstance(operand, bool):
                    raise LanguageExecutionError(
                        f"'not' attend un booléen, obtenu {type(operand).__name__}", node
                    )
                return not operand
    

    def visit_BinaryOpNode(self, node: BinaryOpNode) -> ValidRegisterType:
        left  = self.evaluator.visit(node.left)
        right = self.evaluator.visit(node.right)
        op    = node.operator
        try:
            match op:
                case BinaryOperatorType.ADD: result = left + right
                case BinaryOperatorType.SUB: result = left - right
                case BinaryOperatorType.MUL: result = left * right
                case BinaryOperatorType.DIV:
                    if right == 0:
                        raise LanguageExecutionError("Division par zéro", node)
                    result = left / right
                case BinaryOperatorType.MOD:
                    if right == 0:
                        raise LanguageExecutionError("Modulo par zéro", node)
                    result = left % right
                case BinaryOperatorType.AND: return left and right
                case BinaryOperatorType.OR:  return left or right
                case BinaryOperatorType.EQ:  return left == right
                case BinaryOperatorType.NEQ: return left != right
                case BinaryOperatorType.LT:  return left < right
                case BinaryOperatorType.GT:  return left > right
                case BinaryOperatorType.LTE: return left <= right
                case BinaryOperatorType.GTE: return left >= right
        except TypeError as exc:
            raise LanguageExecutionError(
                f"Opération '{op.value}' invalide entre "
                f"{type(left).__name__} et {type(right).__name__}", node
            ) from exc

        _check_value_bounds(result, node)
        return result