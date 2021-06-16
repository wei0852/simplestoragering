"""simple storage ring
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

Particle = simplestoragering.particles.RefParticle
Drift = simplestoragering.drift.Drift
HBend = simplestoragering.hbend.HBend
Quadrupole = simplestoragering.quadrupole.Quadrupole
Sextupole = simplestoragering.sextupole.Sextupole
RFCavity = simplestoragering.rfcavity.RFCavity
CSLattice = cslattice.CSLattice
SlimRing = slimlattice.SlimRing

