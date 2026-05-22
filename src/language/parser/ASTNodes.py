
from typing import List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC
from src.language.lexer import Token


# ─── ASTNode : Classe abstraite ───────────────────────────────────────────────

@dataclass
class ASTNode(ABC):
    """Classe de base pour tous les nœuds de l'arbre syntaxique"""
    position: Optional[tuple[int, int]] = field(default=None, kw_only=True)  # (line, column)
    
    @classmethod
    def from_token(cls, token: Token, **kwargs):
        """Factory method pour créer un node avec position du token"""
        return cls(**kwargs, position=token.position)
    
    @property
    def line(self) -> Optional[int]:
        """Ligne de la position du node"""
        return self.position[0] if self.position else None
    
    @property
    def column(self) -> Optional[int]:
        """Colonne de la position du node"""
        return self.position[1] if self.position else None


class TypeEnum(Enum):
    """Clear representation of types"""
    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"


# ─── Structure du programme ───────────────────────────────────────────────────

@dataclass
class ProgramNode(ASTNode):
    """Nœud racine du programme"""
    statements: List[ASTNode]


# ─── Types de données ─────────────────────────────────────────────────────────

class NumberType(TypeEnum):
    INTEGER = "integer"
    FLOAT = "float"

@dataclass
class NumberNode(ASTNode):
    """Nœud pour les nombres (entiers, flottants, complexes)"""
    value: Union[int, float]
    type: NumberType


@dataclass
class BooleanNode(ASTNode):
    """Nœud pour les booléens"""
    value: bool


# ─── Registres ────────────────────────────────────────────────────────────────

@dataclass
class RegisterNode(ASTNode):
    """Nœud pour les registres"""
    value: str


@dataclass
class AssignmentNode(ASTNode):
    """Nœud pour l'assignation"""
    target: RegisterNode
    value: ASTNode


# ─── Opérateurs ────────────────────────────────────────────────────────────────

class BinaryOperatorType(TypeEnum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    AND = "and"
    OR = "or"
    EQ = "=="
    NEQ = "!="
    LT = "<"
    GT = ">"
    LTE = "<="
    GTE = ">="

class UnaryOperatorType(TypeEnum):
    POSITIVE = "+"
    NEGATIVE = "-"
    NOT = "not"


@dataclass
class BinaryOpNode(ASTNode):
    """Nœud pour les opérations binaires (+, -, *, /, etc.)"""
    left: ASTNode
    operator: BinaryOperatorType
    right: ASTNode


@dataclass
class UnaryOpNode(ASTNode):
    """Nœud pour les opérations unaires (-, +, not)"""
    operator: UnaryOperatorType
    operand: ASTNode


# ─── Contrôle ──────────────────────────────────────────────────────────────────

@dataclass
class IfNode(ASTNode):
    """Nœud pour les conditions if/else"""
    condition: ASTNode
    then_block: List[ASTNode]
    else_block: Optional[List[ASTNode]]


@dataclass
class WhileNode(ASTNode):
    """Nœud pour les boucles while"""
    condition: ASTNode
    body: List[ASTNode]


@dataclass
class BreakNode(ASTNode):
    """Nœud pour l'instruction break"""


@dataclass
class ContinueNode(ASTNode):
    """Nœud pour l'instruction continue"""


# ─── I/O ──────────────────────────────────────────────────────────────────────

@dataclass
class InputNode(ASTNode):
    """Nœud pour l'instruction input"""
    target: RegisterNode

@dataclass
class OutputNode(ASTNode):
    """Nœud pour l'instruction output"""
    value: ASTNode