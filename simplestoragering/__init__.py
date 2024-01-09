# -*- coding: utf-8 -*-
"""
simple storage ring
define different components:
    Drift, HBend, Quadrupole, Sextupole, RFCavity, Mark
Courant-Snyder method for uncoupled motion, solving twiss parameters of vertical and horizontal direction.
"""


from .globalvars import set_ref_energy, RefParticle
from .components import Mark
from .Drift import Drift
from .HBend import HBend
from .Quadrupole import Quadrupole
from .Sextupole import Sextupole
from .Octupole import Octupole
from .RFCavity import RFCavity
from .CSLattice import CSLattice
from .plotlib import plot_lattice, plot_layout_in_ax, plot_with_background, get_col, plot_resonance_line_in_ax
from .functions import compute_transfer_matrix_by_tracking, output_opa_file, chromaticity_correction, track_4d_closed_orbit, output_elegant_file
from .exceptions import ParticleLost, Unstable
