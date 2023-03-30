# -*- coding: utf-8 -*-
"""
simple storage ring
define different components:
    Drift, HBend, Quadrupole, Sextupole, RFCavity, Mark
two method to solve lattice:
    Courant-Snyder method for uncoupled motion, solving twiss parameters of vertical and horizontal direction.
    Slim method developed by A. Chao, calculating equilibrium beam matrix by transfer matrix.
"""


from simplestoragering.globalvars import set_ref_energy, RefParticle
from simplestoragering.components import Mark
from simplestoragering.Drift import Drift
from simplestoragering.HBend import HBend
from simplestoragering.Quadrupole import Quadrupole
from simplestoragering.Sextupole import Sextupole
from simplestoragering.Octupole import Octupole
from simplestoragering.RFCavity import RFCavity
from simplestoragering.CSLattice import CSLattice
from simplestoragering.slimlattice import SlimRing, compute_twiss_of_slim_method
from simplestoragering.plotlib import plot_lattice, plot_layout_in_ax, plot_with_background, get_col, plot_resonance_line_in_ax
from simplestoragering.functions import compute_transfer_matrix_by_tracking, output_opa_file, chromaticity_correction, track_4d_closed_orbit, output_elegant_file
