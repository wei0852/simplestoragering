# -*- coding: utf-8 -*-
"""
simple storage ring, compiled by cython.
first set_ref_energy
then define different components:
    Drift(name: str = None, length: float = 0.0, Ax: float = 10, Ay: float = 10)
    HBend(name: str = None, length: float = 0, theta: float = 0, theta_in: float = 0, theta_out: float = 0, k1: float = 0, n_slices: int = 1, Ax: float = 10, Ay: float = 10)
    Quadrupole(name: str = None, length: float = 0, k1: float = 0, n_slices: int = 4, Ax: float = 10, Ay: float = 10) 
    Sextupole(name: str = None, length: float = 0, k2: float = 0, n_slices: int = 4, Ax: float = 10, Ay: float = 10)
    Octupole(name: str = None, length: float = 0, k3: float = 0, n_slices: int = 1, Ax: float = 10, Ay: float = 10)
    Mark(name)
finally use CSLattice class to generate storage ring lattice.
Courant-Snyder method for uncoupled motion, solving twiss parameters of vertical and horizontal direction.

Functions:
    symplectic_track(particle, lattice, n_turns: int, record = True)
    output_opa_file(lattice: CSLattice, file_name=None)
    output_elegant_file(lattice: CSLattice, filename=None, new_version=True)
    chromaticity_correction(lattice: CSLattice, sextupole_name_list: list, target: list=None, initial_k2=None, update_sext=True, printout=True)
    
and some functions to visualize lattice data quickly.
    plot_layout_in_ax(ele_list: list, ax: Axes, ratio=0.03)
    plot_resonance_line_in_ax(ax: Axes, order: int = 3, refnux: float=None, refnuy: float=None)
    plot_lattice(ele_list, parameter: str or list[str], with_layout=True)
or use get_col(ele_list, parameter) to get a column of the parameter along the element list.
"""


from .globalvars import set_ref_energy
from .components import Mark
from .Drift import Drift
from .HBend import HBend
from .Quadrupole import Quadrupole
from .Sextupole import Sextupole
from .Octupole import Octupole
from .CSLattice import CSLattice
from .DrivingTerms import DrivingTerms
from .plotlib import plot_lattice, plot_layout_in_ax, plot_with_background, get_col, plot_resonance_line_in_ax, plot_RDTs_along_ring
from .functions import output_opa_file, output_elegant_file, chromaticity_correction
from .line_matrix import line_matrix
from .track import symplectic_track, track_4d_closed_orbit
from .exceptions import ParticleLost
from .DynamicAperture import XYGrid, NLine
