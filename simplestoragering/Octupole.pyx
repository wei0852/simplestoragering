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

cdef extern from "BndStrMPoleSymplectic4Pass.c":
    void BndStrMPoleSymplectic4Pass(double *r, double le, double irho, double *A, double *B,
        int max_order, int num_int_steps,
        double entrance_angle, double exit_angle,
        double fint1, double fint2, double gap)

cdef class Octupole(Element):
    """Octupole(name: str = None, length: float = 0, k3: float = 0, n_slices: int = 1, Ax: float = 10, Ay: float = 10)"""

    def __init__(self, name: str = None, length: float = 0, k3: float = 0, n_slices: int = 1, Ax: float = 10, Ay: float = 10):
        self.name = name
        self.length = length
        self.k3 = k3
        self.n_slices = n_slices
        self.Ax = Ax
        self.Ay = Ay

    def slice(self, n_slices: int) -> list:
        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]
        cdef double[12] twiss1
        cdef double[6][6] matrix
        cdef double[6] closed_orbit
        cdef double current_s, length
        ele_list = []
        current_s = self.s
        length = self.length / n_slices
        closed_orbit = self.closed_orbit
        kick_slices = max(int(self.n_slices / n_slices), 1)
        for i in range(n_slices):
            ele = Octupole(self.name, length, self.k3, kick_slices, self.Ax, self.Ay)
            assin_twiss(ele, twiss0)
            ele.s = current_s
            ele.closed_orbit = closed_orbit
            # drift_matrix(matrix, length)
            track_matrix(ele, matrix)
            ele.symplectic_track(closed_orbit)
            next_twiss(matrix, twiss0, twiss1)
            for i in range(12):
                twiss0[i] = twiss1[i]
            ele_list.append(ele)
            current_s = current_s + ele.length
        return ele_list

    @property
    def matrix(self):
        cdef double[6][6] matrix
        drift_matrix(matrix, self.length)
        return np.array(matrix)

    cdef int symplectic_track(self, double[6] particle):
        cdef double[6] r
        cdef double[4] A = [0.0, 0.0, 0.0, 0.0]
        cdef double[4] B = [0.0, 0.0, 0.0, self.k3 / 6]
        r[0] = particle[0]
        r[1] = particle[1]
        r[2] = particle[2]
        r[3] = particle[3]
        r[5] = -particle[4]
        r[4] = particle[5]

        BndStrMPoleSymplectic4Pass(<double *> &r, self.length, 0.0, <double *> &A, <double *> &B, 3, self.n_slices, 0.0, 0.0, 0.0, 0.0, 0.0)

        particle[0] = r[0]
        particle[1] = r[1]
        particle[2] = r[2]
        particle[3] = r[3]
        particle[4] = -r[5]
        particle[5] = r[4]
        if not (particle[0] / self.Ax) ** 2 + (particle[2] / self.Ay) ** 2 < 1:
            return -1
        return 0

    cpdef copy(self):
        return Octupole(self.name, self.length, self.k3, self.n_slices, self.Ax, self.Ay)

    cpdef linear_optics(self):
        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]
        cdef double[6][6] matrix
        cdef double[12] twiss1
        drift_matrix(matrix, self.length)
        next_twiss(matrix, twiss0, twiss1)
        return np.zeros(7), np.array(twiss1)

    cpdef driving_terms(self, delta):
        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]
        cdef double[6][6] matrix
        cdef double[6] closed_orbit
        cdef double[12] twiss1
        cdef double length, b4l, b3l
        cdef complex h21000, h30000, h10110, h10020, h10200, jj
        cdef complex h21000j, h30000j, h10110j, h10020j, h10200j
        cdef complex  h22000, h11110, h00220, h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400
        jj = complex(0, 1)
        h21000 = h30000 = h10110 = h10020 = h10200 = 0
        h22000 = h11110 = h00220 = h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
        n_slices = 4
        length = self.length / n_slices
        sub_slices = max(int(self.n_slices / n_slices), 1)
        b4l = length * self.k3 / 6
        b3l_0 = length * self.k2 / 2
        ele_slice = Octupole(self.name, length, self.k3, sub_slices, self.Ax, self.Ay)
        ele_slice.length = length
        ele_slice.n_slices = sub_slices
        closed_orbit = self.closed_orbit
        for i in range(4):
            assin_twiss(ele_slice, twiss0)
            ele_slice.closed_orbit = closed_orbit
            xco = closed_orbit[0]
            track_matrix(ele_slice, matrix)
            ele_slice.symplectic_track(closed_orbit)
            next_twiss(matrix, twiss0, twiss1)
            b3l = b4l * 3 * xco + b3l_0
            betax = (twiss0[0] + twiss1[0]) / 2
            betay = (twiss0[3] + twiss1[3]) / 2
            psix = (twiss0[10] + twiss1[10]) / 2
            psiy = (twiss0[11] + twiss1[11]) / 2

            if b3l != 0:
                h21000j = -b3l * betax ** 1.5 * (cos(psix) + jj * sin(psix)) / 8
                h30000j = -b3l * betax ** 1.5 * (cos(3 * psix) + jj * sin(3 * psix)) / 24
                h10110j = b3l * betax ** 0.5 * betay * (cos(psix) + jj * sin(psix)) / 4
                h10020j = b3l * betax ** 0.5 * betay * (cos(psix - 2 * psiy) + jj * sin(psix - 2 * psiy)) / 8
                h10200j = b3l * betax ** 0.5 * betay * (cos(psix + 2 * psiy) + jj * sin(psix + 2 * psiy)) / 8
    
                h12000j = h21000j.conjugate()
                h01110j = h10110j.conjugate()
                h01200j = h10020j.conjugate()
                h12000 = h21000.conjugate()
                h01110 = h10110.conjugate()
                h01200 = h10020.conjugate()
    
                h22000 = h22000 + jj * ((h21000 * h12000j - h12000 * h21000j) * 3 
                      + (h30000 * h30000j.conjugate() - h30000.conjugate() * h30000j) * 9)
    
                h11110 = h11110 + jj * ((h21000 * h01110j - h01110 * h21000j) * 2 
                      - (h12000 * h10110j - h10110 * h12000j) * 2 
                      - (h10020 * h01200j - h01200 * h10020j) * 4 
                      + (h10200 * h10200j.conjugate() - h10200.conjugate() * h10200j) * 4)
    
                h00220 = h00220 + jj * ((h10020 * h01200j - h01200 * h10020j) 
                      + (h10200 * h10200j.conjugate() - h10200.conjugate() * h10200j) 
                      + (h10110 * h01110j - h01110 * h10110j))
    
                h31000 = h31000 + jj * (h30000 * h12000j - h12000 * h30000j) * 6
    
                h40000 = h40000 + jj * (h30000 * h21000j - h21000 * h30000j) * 3
    
                h20110 = h20110 + jj * ((h30000 * h01110j - h01110 * h30000j) * 3 
                      - (h21000 * h10110j - h10110 * h21000j) 
                      + (h10200 * h10020j - h10020 * h10200j) * 4)
    
                h11200 = h11200 + jj * ((h10200 * h12000j - h12000 * h10200j) * 2 
                      + (h21000 * h01200j - h01200 * h21000j) * 2 
                      + (h10200 * h01110j - h01110 * h10200j) * 2 
                      - (h10110 * h01200j - h01200 * h10110j) * 2)
    
                h20020 = h20020 + jj * (-(h21000 * h10020j - h10020 * h21000j) 
                      + (h30000 * h10200j.conjugate() - h10200.conjugate() * h30000j) * 3 
                      + (h10110 * h10020j - h10020 * h10110j) * 2)
    
                h20200 = h20200 + jj * ((h30000 * h01200j - h01200 * h30000j) * 3 
                      + (h10200 * h21000j - h21000 * h10200j) 
                      - (h10110 * h10200j - h10200 * h10110j) * 2)
    
                h00310 = h00310 + jj * ((h10200 * h01110j - h01110 * h10200j) 
                      + (h10110 * h01200j - h01200 * h10110j))
    
                h00400 = h00400 + jj * (h10200 * h01200j - h01200 * h10200j)
                h21000 += h21000j
                h30000 += h30000j
                h10110 += h10110j
                h10020 += h10020j
                h10200 += h10200j

            h22000 = h22000 - 3 * b4l * betax ** 2 / 32
            h11110 = h11110 + 3 * b4l * betax * betay / 8
            h00220 = h00220 - 3 * b4l * betay ** 2 / 32
            
            h31000 = h31000 - b4l * betax ** 2 * (cos(2 * psix) + jj * sin(2 * psix)) / 16
            h40000 = h40000 - b4l * betax ** 2 * (cos(4 * psix) + jj * sin(4 * psix)) / 64
            h20110 = h20110 + 3 * b4l * betax * betay * (cos(2 * psix) + jj * sin(2 * psix)) / 16
            h11200 = h11200 + 3 * b4l * betax * betay * (cos(2 * psiy) + jj * sin(2 * psiy)) / 16
            h20020 = h20020 + 3 * b4l * betax * betay * (cos(2 * psix - 2 * psiy) + jj * sin(2 * psix - 2 * psiy)) / 32
            h20200 = h20200 + 3 * b4l * betax * betay * (cos(2 * psix + 2 * psiy) + jj * sin(2 * psix + 2 * psiy)) / 32
            h00310 = h00310 - b4l * betay ** 2 * (cos(2 * psiy) + jj * sin(2 * psiy)) / 16
            h00400 = h00400 - b4l * betay ** 2 * (cos(4 * psiy) + jj * sin(4 * psiy)) / 64
            for i in range(12):
                twiss0[i] = twiss1[i]
        return np.array([h21000, h30000, h10110, h10020, h10200,
                         h22000, h11110, h00220, h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400])
