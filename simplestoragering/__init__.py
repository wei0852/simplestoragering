# -*- coding: utf-8 -*-
"""
simple storage ring
define different components:
    Drift, HBend, Quadrupole, Sextupole, RFCavity
two method to solve lattice:
    Courant-Snyder method for uncoupled motion, solving twiss parameters of vertical and horizontal direction.
    Slim method developed by A. Chao, calculating equilibrium beam matrix by transfer matrix.
"""


from simplestoragering.particles import RefParticle, calculate_beta, Beam7
from simplestoragering.components import Mark
from simplestoragering.Drift import Drift
from simplestoragering.HBend import HBend
from simplestoragering.Quadrupole import Quadrupole
from simplestoragering.Sextupole import Sextupole
from simplestoragering.RFCavity import RFCavity
from simplestoragering.CSLattice import CSLattice
from simplestoragering.slimlattice import SlimRing, compute_twiss_of_slim_method
from simplestoragering.plotlattice import plot_lattice, plot_layout_in_ax, plot_with_background, get_col
from simplestoragering.functions import compute_transfer_matrix_by_tracking, output_opa_file, chromaticity_correction
