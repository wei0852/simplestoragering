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
from simplestoragering.drift import Drift
from simplestoragering.hbend import HBend
from simplestoragering.quadrupole import Quadrupole
from simplestoragering.sextupole import Sextupole
from simplestoragering.rfcavity import RFCavity
from simplestoragering.cslattice import CSLattice
from simplestoragering.slimlattice import SlimRing, compute_twiss_of_slim_method
from simplestoragering.plotlattice import plot_lattice, plot_layout_in_ax, plot_with_background, get_col
from simplestoragering.functions import compute_transfer_matrix_by_tracking, output_opa_file
