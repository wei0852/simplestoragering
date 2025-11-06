# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: profile=False
from .components cimport Element, assin_twiss
from .globalvars cimport Cr, refgamma, refbeta, refenergy
from .Drift cimport drift_matrix
from .exceptions import ParticleLost
from .c_functions cimport next_twiss, track_matrix
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, cos, sin


cdef extern from "RFCavityPass.c":
    void RFCavityPass(double *r_in, double le, double nv, double freq, double h, double lag, double philag,
                  int nturn, double T0)

cdef class RFCavity(Element):
    """RFCavity(name: str = None, voltage_in_MeV: float = 0, frequency: float = 0, harmonic_number: float = 0, phase: float = 0)"""

    def __init__(self, name: str = None, voltage_in_MeV: float = 0, frequency: float = 0, harmonic_number: float = 0, phase: float = 0):
        self.name = name
        self.voltage = voltage_in_MeV
        self.f_rf = frequency
        self.harmonic_number = harmonic_number
        self.phase = phase
        self.enable = 1

    @property
    def matrix(self):  #TODO
        return np.identity(6)

    cdef int symplectic_track(self, double[6] particle):
        if not self.enable:
            return 0
        cdef double[6] r
        cdef double nv
        cdef double T0
        r[0] = particle[0]
        r[1] = particle[1]
        r[2] = particle[2]
        r[3] = particle[3]
        r[5] = -particle[4]
        r[4] = particle[5]
        nv = self.voltage / refenergy
        T0 = 1.0/self.f_rf  #      /* Does not matter since nturns == 0 */

        RFCavityPass(<double *> &r, 0.0, nv, self.f_rf, self.harmonic_number, 0.0, self.phase, 0, T0)

        particle[0] = r[0]
        particle[1] = r[1]
        particle[2] = r[2]
        particle[3] = r[3]
        particle[4] = -r[5]
        particle[5] = r[4]
        return 0

    cpdef copy(self):
        return RFCavity(self.name, self.voltage, self.f_rf, self.harmonic_number, self.phase)
