# -*- coding: utf-8 -*-
"""
simple storage ring
define different components:
    Drift, HBend, Quadrupole, Sextupole, RFCavity
two method to solve lattice:
    Courant-Snyder method for uncoupled motion, solving twiss parameters of vertical and horizontal direction.
    Slim method developed by A. Chao, calculating equilibrium beam matrix by transfer matrix.
"""

import simplestoragering.particles
import simplestoragering.components
import simplestoragering.cslattice
import simplestoragering.slimlattice
import simplestoragering.drift
import simplestoragering.hbend
import simplestoragering.quadrupole
import simplestoragering.sextupole
import simplestoragering.rfcavity
import simplestoragering.plotlattice
import simplestoragering.functions

RefParticle = simplestoragering.particles.RefParticle
Drift = simplestoragering.drift.Drift
HBend = simplestoragering.hbend.HBend
Quadrupole = simplestoragering.quadrupole.Quadrupole
Sextupole = simplestoragering.sextupole.Sextupole
RFCavity = simplestoragering.rfcavity.RFCavity
CSLattice = cslattice.CSLattice
SlimRing = slimlattice.SlimRing
plot_lattice = plotlattice.plot_lattice
plot_with_background = plotlattice.plot_with_background
get_col = plotlattice.get_col

# other functions

compute_twiss_of_slim_method = simplestoragering.slimlattice.compute_twiss_of_slim_method
calculate_beta = simplestoragering.particles.calculate_beta
compute_transfer_matrix_by_tracking = simplestoragering.functions.compute_transfer_matrix_by_tracking
