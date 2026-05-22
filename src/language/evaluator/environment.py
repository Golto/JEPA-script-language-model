from typing import Optional, Union
from .exception import LanguageExecutionError

# NOTE Type de valeurs valides dans un registre
ValidRegisterType = Union[int, float, bool]

REGISTER_COUNT = 16
VALID_REGISTERS = {f"r{i}" for i in range(REGISTER_COUNT)}


# ─── Exception ────────────────────────────────────────────────────────────────

class EnvironmentError(LanguageExecutionError):
    """Erreur liée à l'environnement d'exécution"""


class UndefinedRegisterError(EnvironmentError):
    """Lecture d'un registre non initialisé"""
    def __init__(self, name: str):
        super().__init__(f"Registre '{name}' lu avant d'être initialisé")
        self.name = name


class InvalidRegisterError(EnvironmentError):
    """Accès à un registre inexistant"""
    def __init__(self, name: str):
        super().__init__(f"'{name}' n'est pas un registre valide (r0–r15)")
        self.name = name


# ─── Environment ──────────────────────────────────────────────────────────────

class Environment:
    """
    Environnement d'exécution : gère les 16 registres r0–r15.
    Les registres sont initialisés à None (non définis).
    """

    def __init__(self):
        self._registers: dict[str, Optional[ValidRegisterType]] = {
            f"r{i}": None for i in range(REGISTER_COUNT)
        }

    # ── Accès ─────────────────────────────────────────────────────────────────

    def get(self, name: str) -> ValidRegisterType:
        self._validate(name)
        value = self._registers[name]
        if value is None:
            raise UndefinedRegisterError(name)
        return value

    def set(self, name: str, value: ValidRegisterType) -> None:
        self._validate(name)
        self._registers[name] = value

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _validate(self, name: str) -> None:
        if name not in VALID_REGISTERS:
            raise InvalidRegisterError(name)

    def reset(self) -> None:
        """Remet tous les registres à None"""
        for key in self._registers:
            self._registers[key] = None

    def snapshot(self) -> dict[str, Optional[ValidRegisterType]]:
        """Retourne une copie de l'état courant des registres"""
        return dict(self._registers)

    def __repr__(self) -> str:
        defined = {k: v for k, v in self._registers.items() if v is not None}
        return f"<Environment {defined}>"