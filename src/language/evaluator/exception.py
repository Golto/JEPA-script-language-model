from typing import Optional
from src.language.parser.ASTNodes import ASTNode

class LanguageExecutionError(Exception):
    def __init__(self, message: str, node: Optional[ASTNode] = None):
        if node and node.position:
            line, col = node.position
            super().__init__(f"Erreur à la ligne {line}, colonne {col}: {message}")
        else:
            super().__init__(message)