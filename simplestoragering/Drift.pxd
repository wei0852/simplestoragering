# -*- coding: utf-8 -*-
# cython: language_level=3
from .components cimport Element, assin_twiss
from .exceptions import ParticleLost
from .globalvars cimport refgamma, refbeta, refenergy
from .c_functions cimport next_twiss
import numpy as np
cimport numpy as np

cdef class Drift(Element):
    cpdef copy(self)

    cdef int symplectic_track(self, double[6] particle)

    cpdef linear_optics(self)

cdef drift_matrix(double[6][6] matrix, double length)
