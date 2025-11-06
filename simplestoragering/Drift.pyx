# -*- coding: utf-8 -*-
# cython: language_level=3
from .components cimport Element, assin_twiss
from .globalvars cimport refgamma, refbeta, refenergy
from .c_functions cimport next_twiss, track_matrix
import numpy as np
cimport numpy as np


cdef extern from "<math.h>":
    double sqrt(double x)

cdef extern from "<math.h>":
    double pow(double x, double y)
    
cdef extern from "<math.h>":
    double fabs(double x)

cdef class Drift(Element):
    """Drift(name: str = None, length: float = 0.0, Ax: float = 10, Ay: float = 10)"""

    def __init__(self, name: str = None, length: float = 0.0, Ax: float = 10, Ay: float = 10):
        self.name = name
        self.length = length
        self.Ax = Ax
        self.Ay = Ay

    def slice(self, n_slices: int, delta=0.0) -> list:
        """slice component to element list, return ele_list"""
        cdef double[12] twiss0=[self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]        
        cdef double[6][6] matrix
        cdef double[6] closed_orbit
        cdef double[12] twiss1
        cdef double length = self.length / n_slices
        cdef double current_s = self.s
        ele_list = []
        closed_orbit = self.closed_orbit
        for i in range(n_slices):
            ele = Drift(self.name, length, self.Ax, self.Ay)
            ele.s = current_s
            assin_twiss(ele, twiss0)
            ele.closed_orbit = closed_orbit
            track_matrix(ele, matrix)
            ele.symplectic_track(closed_orbit)
            next_twiss(matrix, twiss0, twiss1)
            for i in range(12):
                twiss0[i] = twiss1[i]
            ele_list.append(ele)
            current_s = current_s + ele.length
        return ele_list

    cpdef linear_optics(self):
        cdef double[12] twiss=[self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]        
        cdef double[6][6] matrix
        cdef double[12] twiss1
        drift_matrix(matrix, self.length)
        next_twiss(matrix, twiss, twiss1)
        return np.zeros(7), np.array(twiss1)

    @property
    def matrix(self):
        cdef double[6][6] matrix
        drift_matrix(matrix, self.length)
        return np.array(matrix)

    cdef int symplectic_track(self, double[6] particle):
        cdef double x0, y0, z0, ds, d1_square, d1
        x0 = particle[0]
        y0 = particle[2]
        z0 = particle[4]
        ds = self.length
        d1_square = 1 - particle[1] ** 2 - particle[3] ** 2 + 2 * particle[5] + particle[5] ** 2
        if d1_square <= 0:
            return -1
        d1 = sqrt(d1_square)
        particle[0] = x0 + ds * particle[1] / d1
        particle[2] = y0 + ds * particle[3] / d1
        if (particle[0] / self.Ax) ** 2 + (particle[2] / self.Ay) ** 2 > 1:
            return -1
        particle[4] = z0 + ds * (1 - (1 + particle[5]) / d1)
        return 0
    
    cdef int radiation_track(self, double[6] particle):
        return self.symplectic_track(particle)

    cpdef copy(self):
        return Drift(self.name, self.length, self.Ax, self.Ay)


cdef drift_matrix(double[6][6] matrix, double length):
    for i in range(6):
        for j in range(6):
            if i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0
    matrix[0][1] = length
    matrix[2][3] = length
    matrix[4][5] = length / pow(refgamma, 2)
