import numpy as np


class Element():

    name: str
    length: float
    h: float
    k1: float
    k2: float
    k3: float
    Ax: float
    Ay: float
    n_slices: int

    s: float
    closed_orbit: np.ndarray
    betax: float
    alphax: float
    gammax: float
    psix: float
    betay: float
    alphay: float
    gammay: float
    psiy: float
    etax: float
    etaxp: float
    etay: float
    etayp: float

    @property
    def matrix(self):...
    
    @property
    def nux(self):...

    @property
    def nuy(self):...

    def slice(self, n_slices: int) -> list[Element]:...

    def copy(self) -> Element: ...

    def linear_optics(self) -> tuple[np.ndarray, np.ndarray]: ...

    def off_momentum_optics(self, delta) -> np.ndarray: ...


class Mark(Element):
    data: np.ndarray
    record: bool
    def __init__(self, name: str) -> None:...
