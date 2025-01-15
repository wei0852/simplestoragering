from .components import Element
import numpy as np


class HBend(Element):

    def __init__(self, name: str = None, length: float = 0, theta: float = 0, theta_in: float = 0, theta_out: float = 0,
                 k1: float = 0, gap: float = 0, fint_in: float = 0.5, fint_out: float = 0.5,
                 n_slices: int = 10, k2: float = 0, k3: float = 0, Ax: float = 10, Ay: float = 10) -> None:...
    def driving_terms(self, delta) -> np.ndarray:...




