# -*- coding: utf-8 -*-
# cython: language_level=3
from .components cimport Element, assin_twiss
from .globalvars cimport Cr, refgamma, refbeta, pi
from .exceptions import ParticleLost
from .c_functions cimport next_twiss, track_matrix
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, cos, sin, sinh, cosh
from libc.stdlib cimport malloc, free


cdef extern from "BndStrMPoleSymplectic4Pass.c":
    void BndStrMPoleSymplectic4Pass(double *r, double le, double irho, double *A, double *B,
        int max_order, int num_int_steps,
        double entrance_angle, double exit_angle,
        double fint1, double fint2, double gap)

cdef class Quadrupole(Element):
    """Quadrupole(name: str = None, length: float = 0, k1: float = 0, k2: float = 0, k3: float = 0, n_slices: int = 10, Ax: float = 10, Ay: float = 10)
    
    normal.
    """

    def __init__(self, name: str = None, length: float = 0, k1: float = 0, k2: float = 0, k3: float = 0, n_slices: int = 10, Ax: float = 10, Ay: float = 10):
        self.name = name
        self.length = length
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.n_slices = n_slices
        self.Ax = Ax
        self.Ay = Ay

    def slice(self, n_slices: int, delta=0.0) -> list:
        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]
        cdef double[12] twiss1
        cdef double[6][6] matrix
        cdef double[6] closed_orbit
        cdef double current_s, length
        ele_list = []
        current_s = self.s
        length = self.length / n_slices
        sub_slices = max(int(self.n_slices / n_slices), 1)
        closed_orbit = self.closed_orbit
        for i in range(n_slices):
            ele = Quadrupole(self.name, length, self.k1, self.k2, self.k3, sub_slices, self.Ax, self.Ay)
            ele.s = current_s
            assin_twiss(ele, twiss0)
            track_matrix(ele, matrix)
            ele.closed_orbit = closed_orbit
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
        quad_matrix(matrix, self.length, self.k1)
        return np.array(matrix)

    cdef int symplectic_track(self, double[6] particle):
        if self.n_slices > 1 or self.k2 != 0 or self.k3 != 0:
            return self.symplectic4pass(particle)
        # else use faster method (SAMM)
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

    cdef int symplectic4pass(self, double[6] particle):
        cdef double[6] r
        cdef double* A = NULL
        cdef double* B = NULL
        if self.k3 != 0:
            A = <double *>malloc(4 * sizeof(double))
            if not A:
                raise MemoryError("Unable to allocate memory for array.")
            B = <double *>malloc(4 * sizeof(double))
            if not B:
                raise MemoryError("Unable to allocate memory for array.")
            A[0] = 0.0
            A[1] = 0.0
            A[2] = 0.0
            A[3] = 0.0
            B[0] = 0.0
            B[1] = self.k1
            B[2] = self.k2 / 2
            B[3] = self.k3 / 6
            max_order = 3
        elif self.k2 != 0:
            A = <double *>malloc(3 * sizeof(double))
            if not A:
                raise MemoryError("Unable to allocate memory for array.")
            B = <double *>malloc(3 * sizeof(double))
            if not B:
                raise MemoryError("Unable to allocate memory for array.")
            A[0] = 0.0
            A[1] = 0.0
            A[2] = 0.0
            B[0] = 0.0
            B[1] = self.k1
            B[2] = self.k2 / 2
            max_order = 2
        else:
            A = <double *>malloc(2 * sizeof(double))
            if not A:
                raise MemoryError("Unable to allocate memory for array.")
            B = <double *>malloc(2 * sizeof(double))
            if not B:
                raise MemoryError("Unable to allocate memory for array.")
            A[0] = 0.0
            A[1] = 0.0
            B[0] = 0.0
            B[1] = self.k1
            max_order = 1
        r[0] = particle[0]
        r[1] = particle[1]
        r[2] = particle[2]
        r[3] = particle[3]
        r[5] = -particle[4]
        r[4] = particle[5]
        BndStrMPoleSymplectic4Pass(<double *> &r, self.length, 0.0, A, B, max_order, self.n_slices, 0.0, 0.0, 0.0, 0.0, 0.0)

        particle[0] = r[0]
        particle[1] = r[1]
        particle[2] = r[2]
        particle[3] = r[3]
        particle[4] = -r[5]
        particle[5] = r[4]
        free(A)
        free(B)
        if not (particle[0] / self.Ax) ** 2 + (particle[2] / self.Ay) ** 2 < 1:
            return -1
        return 0

    cpdef copy(self):
        return Quadrupole(self.name, self.length, self.k1, self.k2, self.k3, self.n_slices, self.Ax, self.Ay)

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
        b4l = length * self.k3 / 6 / (1 + delta)
        b3l_0 = length * self.k2 / 2 / (1 + delta)
        ele_slice = Quadrupole(self.name, length, self.k1, self.k2, self.k3, sub_slices, self.Ax, self.Ay)
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