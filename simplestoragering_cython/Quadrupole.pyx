# -*- coding: utf-8 -*-
# cython: language_level=3
from .components cimport Element, assin_twiss
from .globalvars cimport Cr, refgamma, refbeta, pi
from .exceptions import ParticleLost
from .c_functions cimport next_twiss
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, cos, sin, sinh, cosh

cdef class Quadrupole(Element):
    """Quadrupole(name: str = None, length: float = 0, k1: float = 0, n_slices: int = 4, Ax: float = 10, Ay: float = 10)
    
    normal.
    """

    def __init__(self, name: str = None, length: float = 0, k1: float = 0, n_slices: int = 4, Ax: float = 10, Ay: float = 10):
        self.name = name
        self.length = length
        self.k1 = k1
        self.n_slices = n_slices
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
            ele = Quadrupole(self.name, length, self.k1, Ax=self.Ax, Ay=self.Ay)
            ele.s = current_s
            assin_twiss(ele, twiss0)
            quad_matrix(matrix, length, self.k1)
            next_twiss(matrix, twiss0, twiss1)
            for i in range(12):
                twiss0[i] = twiss1[i]
            ele_list.append(ele)
            current_s = current_s + ele.length
        length = self.length + self.s - current_s
        ele = Quadrupole(self.name, length, self.k1, Ax=self.Ax, Ay=self.Ay)
        ele.s = current_s
        assin_twiss(ele, twiss0)
        ele_list.append(ele)
        return ele_list

    @property
    def matrix(self):
        cdef double[6][6] matrix
        quad_matrix(matrix, self.length, self.k1)
        return np.array(matrix)

    cdef int symplectic_track(self, double[6] particle):
        x0 = particle[0]
        px0 = particle[1]
        y0 = particle[2]
        py0 = particle[3]
        ct0 = particle[4]
        dp0 = particle[5]
        beta0 = refbeta

        ds = self.length
        k1 = self.k1
        d1_square = 1 + 2 * dp0 / beta0 + dp0 * dp0
        if d1_square <= 0:
            return -1
        d1 = sqrt(d1_square)

        w_2 = k1 / d1
        if w_2 >0:
            w = sqrt(w_2)
            xs = sin(w * ds)
            xc = cos(w * ds)
            ys = sinh(w * ds)
            yc = cosh(w * ds)
            xs2 = sin(2 * w * ds)
            ys2 = sinh(2 * w * ds)

            particle[0] = x0 * xc + px0 * xs * w / k1
            particle[1] = -k1 * x0 * xs / w + px0 * xc
            particle[2] = y0 * yc + py0 * ys * w / k1
            particle[3] = k1 * y0 * ys / w + py0 * yc

            d0 = 1 / beta0 + dp0
            d2 = -d0 / d1 / d1 / d1 / 2

            c0 = (1 / beta0 - d0 / d1) * ds
            c11 = k1 * k1 * d2 * (xs2 / w - 2 * ds) / w_2 / 4
            c12 = -k1 * d2 * xs * xs / w_2
            c22 = d2 * (xs2 / w + 2 * ds) / 4
            c33 = k1 * k1 * d2 * (ys2 / w - 2 * ds) / w_2 / 4
            c34 = k1 * d2 * ys * ys / w / w
            c44 = d2 * (ys2 / w + 2 * ds) / 4

            particle[4] = ct0 + c0 + c11 * x0 * x0 + c12 * x0 * px0 + c22 * px0 * px0 + c33 * y0 * y0 + c34 * y0 * py0 + c44 * py0 * py0

        elif w_2 < 0:
            w = sqrt(-w_2)
            xs = sinh(w * ds)
            xc = cosh(w * ds)
            ys = sin(w * ds)
            yc = cos(w * ds)
            xs2 = sinh(2 * w * ds)
            ys2 = sin(2 * w * ds)

            particle[0] = x0 * xc - px0 * xs * w / k1
            particle[1] = -k1 * x0 * xs / w + px0 * xc
            particle[2] = y0 * yc - py0 * ys * w / k1
            particle[3] = k1 * y0 * ys / w + py0 * yc

            d0 = 1 / beta0 + dp0
            d2 = -d0 / d1 / d1 / d1 / 2

            c0 = (1 / beta0 - d0 / d1) * ds
            c11 = k1 * k1 * d2 * (xs2 / w - 2 * ds) / w_2 / 4
            c12 = k1 * d2 * xs * xs / w_2
            c22 = d2 * (xs2 / w + 2 * ds) / 4
            c33 = k1 * k1 * d2 * (ys2 / w - 2 * ds) / w_2 / 4
            c34 = -k1 * d2 * ys * ys / w_2
            c44 = d2 * (ys2 / w + 2 * ds) / 4

            particle[4] = ct0 + c0 + c11 * x0 * x0 + c12 * x0 * px0 + c22 * px0 * px0 + c33 * y0 * y0 + c34 * y0 * py0 + c44 * py0 * py0
        
        else:
            d1_square = 1 - particle[1] ** 2 - particle[3] ** 2 + 2 * particle[5] / refbeta + particle[5] ** 2
            if d1_square <= 0:
                return -1
            d1 = sqrt(d1_square)
            particle[0] = x0 + ds * particle[1] / d1
            particle[2] = y0 + ds * particle[3] / d1
            particle[4] = ct0 + ds * (1 - (1 + refbeta * particle[5]) / d1) / refbeta
        if (particle[0] / self.Ax) ** 2 + (particle[2] / self.Ay) ** 2 > 1:
            return -1
        return 0

    cpdef copy(self):
        return Quadrupole(self.name, self.length, self.k1, self.n_slices, self.Ax, self.Ay)

    cpdef linear_optics(self):
        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]
        cdef double[7] integrals = [0, 0, 0, 0, 0, 0, 0]
        cdef double[12] twiss1
        cdef double[6][6] matrix
        cdef double current_s, length
        current_s = 0
        length = 0.01
        while current_s < self.length - length:
            quad_matrix(matrix, length, self.k1)
            next_twiss(matrix, twiss0, twiss1)
            betax = (twiss0[0] + twiss1[0]) / 2
            betay = (twiss0[3] + twiss1[3]) / 2
            integrals[5] += - self.k1 * length * betax / 4 / pi
            integrals[6] += self.k1 * length * betay / 4 / pi
            current_s = current_s + length
            for i in range(12):
                twiss0[i] = twiss1[i]
        length = self.length - current_s
        quad_matrix(matrix, length, self.k1)
        next_twiss(matrix, twiss0, twiss1)
        betax = (twiss0[0] + twiss1[0]) / 2
        betay = (twiss0[3] + twiss1[3]) / 2
        integrals[5] += - self.k1 * length * betax / 4 / pi
        integrals[6] += self.k1 * length * betay / 4 / pi
        return integrals, twiss1

    cpdef driving_terms(self):
        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]
        cdef double[12] twiss1
        cdef double[6][6] matrix
        cdef double current_s, length
        cdef complex  h20001, h00201, h10002, jj
        jj = complex(0, 1)
        current_s = 0
        h20001 = h00201 = h10002 = 0
        length = self.length / self.n_slices  # can be very small.
        for i in range(self.n_slices):
            quad_matrix(matrix, length, self.k1)
            next_twiss(matrix, twiss0, twiss1)
            betax = (twiss0[0] + twiss1[0]) / 2
            betay = (twiss0[3] + twiss1[3]) / 2
            etax = (twiss0[6] + twiss1[6]) / 2
            psix = (twiss0[10] + twiss1[10]) / 2
            psiy = (twiss0[11] + twiss1[11]) / 2

            h20001 += betax * (cos(2 * psix) + jj * sin(2 * psix))
            h00201 += betay * (cos(2 * psiy) + jj * sin(2 * psiy))
            h10002 += betax ** 0.5 * etax * (cos(psix) + jj * sin(psix))

            for i in range(12):
                twiss0[i] = twiss1[i]
        h20001 = h20001 * length * self.k1 / 8
        h00201 = -h00201 * length * self.k1 / 8
        h10002 = h10002 * length * self.k1 / 2
        return np.array([h20001, h00201, h10002])


cdef quad_matrix(double[6][6] matrix, double length, double k1):
    cdef double sqk, sqkl
    for i in range(6):
        for j in range(6):
            if i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0
    if k1 > 0:
        sqk = sqrt(k1)
        sqkl = sqk * length
        matrix[0][0] = cos(sqkl)
        matrix[0][1] = sin(sqkl) / sqk
        matrix[1][0] = - sqk * sin(sqkl)
        matrix[1][1] = cos(sqkl)
        matrix[2][2] = cosh(sqkl)
        matrix[2][3] = sinh(sqkl) / sqk
        matrix[3][2] = sqk * sinh(sqkl)
        matrix[3][3] = cosh(sqkl)
        matrix[4][5] = length / refgamma ** 2
    elif k1 < 0:
        sqk = sqrt(-k1)
        sqkl = sqk * length
        matrix[0][0] = cosh(sqkl)
        matrix[0][1] = sinh(sqkl) / sqk
        matrix[1][0] = sqk * sinh(sqkl)
        matrix[1][1] = cosh(sqkl)
        matrix[2][2] = cos(sqkl)
        matrix[2][3] = sin(sqkl) / sqk
        matrix[3][2] = - sqk * sin(sqkl)
        matrix[3][3] = cos(sqkl)
        matrix[4][5] = length / refgamma ** 2
    else:
        for i in range(6):
            for j in range(6):
                if i == j:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0
        matrix[0][1] = length
        matrix[2][3] = length
        matrix[4][5] = length / refgamma ** 2