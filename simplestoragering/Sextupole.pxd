# -*- coding: utf-8 -*-
# cython: language_level=3
from .components cimport Element, assin_twiss
from .globalvars cimport Cr, refgamma, refbeta, pi
from .Drift cimport drift_matrix
from .exceptions import ParticleLost
from .c_functions cimport next_twiss
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, cos, sin


cdef class Sextupole(Element):
    """sextupole"""

    cdef int symplectic_track(self, double[6] particle)
    cdef int radiation_track(self, double[6] particle)
    cpdef copy(self)

    cpdef linear_optics(self)

    cpdef driving_terms(self, delta)
