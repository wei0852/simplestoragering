from .CSLattice import CSLattice
import numpy as np


def symplectic_track(particle, lattice, n_turns: int, record = True) -> np.ndarray:...

def radiation_track(particle, lattice, n_turns: int, record = True) -> np.ndarray:...


def track_4d_closed_orbit(lattice: CSLattice, delta, verbose=True, matrix_precision=1e-9, resdl_limit=1e-12) -> dict:...

