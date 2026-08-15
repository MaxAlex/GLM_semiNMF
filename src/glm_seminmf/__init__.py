"""NB-GLM semi-NMF: signed loadings, non-negative usages, NB likelihood."""

from .model import NBGLMSemiNMF
from .simulate import SimulatedData, simulate_nb_seminmf

__all__ = ["NBGLMSemiNMF", "SimulatedData", "simulate_nb_seminmf"]
__version__ = "0.1.0"
