from typing import Protocol, Optional, Callable
from .environment import Environment, ValidRegisterType
from src.language.parser.ASTNodes import ASTNode


class IEvaluator(Protocol):
    env: Environment
    input_fn: Callable[[str], ValidRegisterType]
    output_fn: Callable[[ValidRegisterType], None]

    def visit(self, node: ASTNode) -> Optional[ValidRegisterType]: ...