"""Fuzzy inference engines."""

from .mamdani import Mamdani
from .tsk import TSK
from .tsukamoto import Tsukamoto

__all__ = ["Mamdani", "TSK", "Tsukamoto"]
