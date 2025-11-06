# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: profile=False
from .components cimport Element, assin_twiss
from .globalvars cimport Cr, refgamma, refbeta
from .Drift cimport drift_matrix
from .exceptions import ParticleLost
from .c_functions cimport next_twiss, track_matrix
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, cos, sin


cdef class RFCavity(Element):
    """ """
    cdef public bint enable
    cdef public double voltage
    cdef public double f_rf
    cdef public double harmonic_number
    cdef public double phase

    cdef int symplectic_track(self, double[6] particle)

    cpdef copy(self)
