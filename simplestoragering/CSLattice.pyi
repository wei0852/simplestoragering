from .components import Element, Mark
from .DrivingTerms import DrivingTerms
import numpy as np

class CSLattice(object):
    nux: float
    nuy: float
    xi_x: float
    xi_y: float
    natural_xi_x: float
    natural_xi_y: float
    I1: float
    I2: float
    I3: float
    I4: float
    I5: float
    Jx: float
    Jy: float
    Js: float
    sigma_e: float
    emittance: float
    U0: float
    f_c: float
    tau0: float
    tau_s: float
    tau_x: float
    tau_y: float
    alpha: float  # momentum compaction factor
    etap: float  # phase slip factor

    delta: float
    length: float
    n_periods: int
    elements: list[Element]
    mark: dict[str, list[Mark]]
    angle: float
    abs_angle: float

    initial_twiss: np.ndarray  # betax, alphax, betay, alphay, etax, etaxp, etay, etayp

    def __init__(self, ele_list: list[Element], n_periods: int = 1, delta: float = 0) -> None:...

    def linear_optics(self, periodicity=True, line_mode=False) -> None:...

    def set_initial_twiss(self, betax, alphax, betay, alphay, etax, etaxp, etay, etayp) -> None:...

    def off_momentum_optics(self, delta=0.0) -> None:...

    def slice_elements(self, drift_length=10.0, bend_length=10.0, quad_length=10.0, sext_length=10.0, oct_length=10.0) -> list[Element]:...
    
    def driving_terms(self, n_periods=None, verbose=True) -> DrivingTerms:...

    def chromatic_driving_terms(self, n_periods=None, verbose=True) -> dict:...

    def track_chromaticity(self, order=2, verbose=True, delta=None, matrix_precision=1e-9, resdl_limit=1e-12) -> dict: ...

