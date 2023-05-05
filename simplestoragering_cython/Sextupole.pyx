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


cdef class Sextupole(Element):
    """sextupole(name: str = None, length: float = 0, k2: float = 0, n_slices: int = 4, Ax: float = 10, Ay: float = 10)
    """

    def __init__(self, name: str = None, length: float = 0, k2: float = 0, n_slices: int = 4, Ax: float = 10, Ay: float = 10):
        self.name = name
        self.length = length
        self.k2 = k2
        self.n_slices = n_slices if n_slices != 0 else 4
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
        for i in range(n_slices - 1):
            ele = Sextupole(self.name, length, self.k2, 1, self.Ax, self.Ay)
            assin_twiss(ele, twiss0)
            ele.s = current_s
            drift_matrix(matrix, length)
            next_twiss(matrix, twiss0, twiss1)
            for i in range(12):
                twiss0[i] = twiss1[i]
            ele_list.append(ele)
            current_s = current_s + ele.length
        length = self.length + self.s - current_s
        ele = Sextupole(self.name, length, self.k2, 1, self.Ax, self.Ay)
        ele.s = current_s
        assin_twiss(ele, twiss0)
        ele_list.append(ele)
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
        k2 = self.k2
        for i in range(self.n_slices):
            d1 = sqrt(1 - px0 * px0 - py0 * py0 + 2 * dp0 / beta0 + dp0 * dp0)

            x1 = x0 + ds * px0 / d1 / 2
            y1 = y0 + ds * py0 / d1 / 2
            ct1 = ct0 + ds * (1 - (1 + beta0 * dp0) / d1) / beta0 / 2

            px1 = px0 - (x1 * x1 - y1 * y1) * k2 * ds / 2
            py1 = py0 + x1 * y1 * k2 * ds
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
        return Sextupole(self.name, self.length, self.k2, self.n_slices, self.Ax, self.Ay)

    cpdef linear_optics(self):
        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]
        cdef double[7] integrals = [0, 0, 0, 0, 0, 0, 0]
        cdef double[12] twiss1
        cdef double[6][6] matrix
        cdef double current_s, length
        current_s = 0
        length = 0.01
        while current_s < self.length - length:
            drift_matrix(matrix, length)
            next_twiss(matrix, twiss0, twiss1)
            betax = (twiss0[0] + twiss1[0]) / 2
            betay = (twiss0[3] + twiss1[3]) / 2
            etax = (twiss0[6] + twiss1[6]) / 2
            integrals[5] += etax * self.k2 * length * betax / 4 / pi
            integrals[6] += - etax * self.k2 * length * betay / 4 / pi
            current_s = current_s + length
            for i in range(12):
                twiss0[i] = twiss1[i]
        length = self.length - current_s
        drift_matrix(matrix, length)
        next_twiss(matrix, twiss0, twiss1)
        betax = (twiss0[0] + twiss1[0]) / 2
        betay = (twiss0[3] + twiss1[3]) / 2
        etax = (twiss0[6] + twiss1[6]) / 2
        integrals[5] += etax * self.k2 * length * betax / 4 / pi
        integrals[6] += - etax * self.k2 * length * betay / 4 / pi
        return np.array(integrals), np.array(twiss1)

    cpdef driving_terms(self):
        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]
        cdef double[12] twiss1
        cdef double[6][6] matrix
        cdef double length, b3l
        cdef complex h20001, h00201, h10002, h21000, h30000, h10110, h10020, h10200, jj
        cdef complex h21000j, h30000j, h10110j, h10020j, h10200j
        cdef complex h22000, h11110, h00220, h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400
        jj = complex(0, 1)
        h20001 = h00201 = h10002 = h21000 = h30000 = h10110 = h10020 = h10200 = 0
        h22000 = h11110 = h00220 = h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
        length = self.length / self.n_slices  # can be very small.
        b3l = length * self.k2 / 2
        for i in range(self.n_slices):
            drift_matrix(matrix, length)
            next_twiss(matrix, twiss0, twiss1)
            betax = (twiss0[0] + twiss1[0]) / 2
            betay = (twiss0[3] + twiss1[3]) / 2
            etax = (twiss0[6] + twiss1[6]) / 2
            psix = (twiss0[10] + twiss1[10]) / 2
            psiy = (twiss0[11] + twiss1[11]) / 2

            h20001 += betax * etax * (cos(2 * psix) + jj * sin(2 * psix))
            h00201 += betay * etax * (cos(2 * psiy) + jj * sin(2 * psiy))
            h10002 += betax ** 0.5 * etax ** 2 * (cos(psix) + jj * sin(psix))

            h21000j = betax ** 1.5 * (cos(psix) + jj * sin(psix))
            h30000j = betax ** 1.5 * (cos(3 * psix) + jj * sin(3 * psix))
            h10110j = betax ** 0.5 * betay * (cos(psix) + jj * sin(psix))
            h10020j = betax ** 0.5 * betay * (cos(psix - 2 * psiy) + jj * sin(psix - 2 * psiy))
            h10200j = betax ** 0.5 * betay * (cos(psix + 2 * psiy) + jj * sin(psix + 2 * psiy))

            h12000j = h21000j.conjugate()
            h01110j = h10110j.conjugate()
            h01200j = h10020j.conjugate()
            h22000 += ((h21000 * h12000j - h21000.conjugate() * h21000j) * 3
                        +(h30000 * h30000j.conjugate() - h30000.conjugate() * h30000j))
            h11110 += (-(h21000 * h01110j - h10110.conjugate() * h21000j)
                        +(h21000.conjugate() * h10110j - h10110 * h12000j)
                        -(h10020 * h01200j - h10020.conjugate() * h10020j)
                        +(h10200 * h10200j.conjugate() - h10200.conjugate() * h10200j))
            h00220 += ((h10020 * h01200j - h10020.conjugate() * h10020j)
                        +(h10200 * h10200j.conjugate() - h10200.conjugate() * h10200j)
                        +(h10110 * h01110j - h10110.conjugate() * h10110j) * 4)
            h31000 += (h30000 * h12000j - h21000.conjugate() * h30000j)
            h40000 += (h30000 * h21000j - h21000 * h30000j)
            h20110 += (-(h30000 * h01110j - h10110.conjugate() * h30000j)
                        +(h21000 * h10110j - h10110 * h21000j)
                            +(h10200 * h10020j - h10020 * h10200j) * 2)
            h11200 += (-(h10200 * h12000j - h21000.conjugate() * h10200j)
                            -(h21000 * h01200j - h10020.conjugate() * h21000j)
                            +(h10200 * h01110j - h10110.conjugate() * h10200j) * 2
                            -(h10110 * h01200j - h10020.conjugate() * h10110j) * 2)
            h20020 += ((h21000 * h10020j - h10020 * h21000j)
                        -(h30000 * h10200j.conjugate() - h10200.conjugate() * h30000j)
                        +(h10110 * h10020j - h10020 * h10110j) * 4)
            h20200 += (-(h30000 * h01200j - h10020.conjugate() * h30000j)
                        -(h10200 * h21000j - h21000 * h10200j)
                        -(h10110 * h10200j - h10200 * h10110j) * 4)
            h00310 += ((h10200 * h01110j - h10110.conjugate() * h10200j)
                        +(h10110 * h01200j - h10020.conjugate() * h10110j))
            h00400 += (h10200 * h01200j - h10020.conjugate() * h10200j)
            h21000 += h21000j
            h30000 += h30000j
            h10110 += h10110j
            h10020 += h10020j
            h10200 += h10200j
            for i in range(12):
                twiss0[i] = twiss1[i]
        h20001 = -h20001 * b3l/ 4
        h00201 = h00201 * b3l / 4
        h10002 = -h10002 * b3l / 2
        h21000 = - h21000 * b3l / 8
        h30000 = - h30000 * b3l / 24
        h10110 = h10110 * b3l / 4
        h10020 = h10020 * b3l / 8
        h10200 = h10200 * b3l / 8
        h22000 = jj * b3l ** 2  * h22000 / 64
        h11110 = jj * b3l ** 2  * h11110 / 16
        h00220 = jj * b3l ** 2  * h00220 / 64
        h31000 = jj * b3l ** 2  * h31000 / 32
        h40000 = jj * b3l ** 2  * h40000 / 64
        h20110 = jj * b3l ** 2  * h20110 / 32
        h11200 = jj * b3l ** 2  * h11200 / 32
        h20020 = jj * b3l ** 2  * h20020 / 64
        h20200 = jj * b3l ** 2  * h20200 / 64
        h00310 = jj * b3l ** 2  * h00310 / 32
        h00400 = jj * b3l ** 2  * h00400 / 64
        return np.array([h21000, h30000, h10110, h10020, h10200, h20001, h00201, h10002,
                         h22000, h11110, h00220, h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400])
