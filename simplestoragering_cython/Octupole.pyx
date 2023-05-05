# -*- coding: utf-8 -*-
# cython: language_level=3
from .components cimport Element, assin_twiss
from .globalvars cimport Cr, refgamma, refbeta
from .Drift cimport drift_matrix
from .exceptions import ParticleLost
from .c_functions cimport next_twiss
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, cos, sin


cdef class Octupole(Element):
    """Octupole(name: str = None, length: float = 0, k3: float = 0, n_slices: int = 1, Ax: float = 10, Ay: float = 10)"""

    def __init__(self, name: str = None, length: float = 0, k3: float = 0, n_slices: int = 1, Ax: float = 10, Ay: float = 10):
        self.name = name
        self.length = length
        self.k3 = k3
        self.n_slices = n_slices if n_slices != 0 else 1
        self.Ax = Ax
        self.Ay = Ay

    def slice(self, n_slices: int) -> list:
        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]
        cdef double[12] twiss1
        cdef double[6][6] matrix
        cdef double current_s, length
        ele_list = []
        current_s = self.s
        length = self.length / n_slices
        for i in range(n_slices):
            ele = Octupole(self.name, length, self.k3, 1, self.Ax, self.Ay)
            assin_twiss(ele, twiss0)
            ele.s = current_s
            drift_matrix(matrix, length)
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
        cdef double x0, px0, y0, py0, ct0, dp0, beta0, ds, k2, d1, x1, y1, ct1, px1, py1, x2, y2, ct2
        x0 = particle[0]
        px0 = particle[1]
        y0 = particle[2]
        py0 = particle[3]
        ct0 = particle[4]
        dp0 = particle[5]
        beta0 = refbeta

        ds = self.length / self.n_slices
        k3 = self.k3
        for i in range(self.n_slices):
            d1 = sqrt(1 - px0 * px0 - py0 * py0 + 2 * dp0 / beta0 + dp0 * dp0)

            x1 = x0 + ds * px0 / d1 / 2
            y1 = y0 + ds * py0 / d1 / 2
            ct1 = ct0 + ds * (1 - (1 + beta0 * dp0) / d1) / beta0 / 2

            px1 = px0 - (x1 ** 3 / 3 - x1 * y1 ** 2) * k3 * ds / 2
            py1 = py0 - (y1 ** 3 / 3 - y1 * x1 ** 2) * k3 * ds / 2
            d1_square = 1 - px1 * px1 - py1 * py1 + 2 * dp0 / beta0 + dp0 * dp0
            if d1_square <=0:
                return -1
            d1 = sqrt(d1_square)
            x2 = x1 + ds * px1 / d1 / 2
            y2 = y1 + ds * py1 / d1 / 2
            ct2 = ct1 + ds * (1 - (1 + beta0 * dp0) / d1) / beta0 / 2
            x0 = x2
            px0 = px1
            y0 = y2
            py0 = py1
            ct0 = ct2
        particle[0] = x2
        particle[1] = px1
        particle[2] = y2
        particle[3] = py1
        particle[4] = ct2
        particle[5] = dp0
        if (particle[0] / self.Ax) ** 2 + (particle[2] / self.Ay) ** 2 > 1:
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

    cpdef driving_terms(self):
        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]
        cdef double[6][6] matrix
        cdef double[12] twiss1
        cdef double length, b4l, h22000, h11110, h00220
        cdef complex  h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400, jj
        jj = complex(0, 1)
        h22000 = h11110 = h00220 = 0
        h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
        length = self.length / self.n_slices  # can be very small.
        b4l = length * self.k3 / 6
        for i in range(self.n_slices):
            drift_matrix(matrix, length)
            next_twiss(matrix, twiss0, twiss1)
            betax = (twiss0[0] + twiss1[0]) / 2
            betay = (twiss0[3] + twiss1[3]) / 2
            psix = (twiss0[10] + twiss1[10]) / 2
            psiy = (twiss0[11] + twiss1[11]) / 2

            h22000 += betax ** 2
            h11110 += betax * betay
            h00220 += betay ** 2

            h31000 += betax ** 2 * (cos(2 * psix) + jj * sin(2 * psix))
            h40000 += betax ** 2 * (cos(4 * psix) + jj * sin(4 * psix))
            h20110 += betax * betay * (cos(2 * psix) + jj * sin(2 * psix))
            h11200 += betax * betay * (cos(2 * psiy) + jj * sin(2 * psiy))
            h20020 += betax * betay * (cos(2 * psix - 2 * psiy) + jj * sin(2 * psix - 2 * psiy))
            h20200 += betax * betay * (cos(2 * psix + 2 * psiy) + jj * sin(2 * psix + 2 * psiy))
            h00310 += betay ** 2 * (cos(2 * psiy) + jj * sin(2 * psiy))
            h00400 += betay ** 2 * (cos(4 * psiy) + jj * sin(4 * psiy))
            for i in range(12):
                twiss0[i] = twiss1[i]
        h31000 =  -b4l * h31000 / 16
        h40000 =  -b4l * h40000 / 64
        h20110 =  3 * b4l * h20110 / 16
        h11200 =  3 * b4l * h11200 / 16
        h20020 =  3 * b4l * h20020 / 32
        h20200 =  3 * b4l * h20200 / 32
        h00310 =  -b4l * h00310 / 16
        h00400 =  -b4l * h00400 / 64
        h22000 = -3 * b4l * h22000 / 32
        h11110 =  3 * b4l * h11110 / 8
        h00220 = -3 * b4l * h00220 / 32
        return np.array([h22000, h11110, h00220, h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400])

