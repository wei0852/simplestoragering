# -*- coding: utf-8 -*-
# cython: language_level=3
from .components cimport Element, assin_twiss
from .globalvars cimport Cr, pi, refbeta, refgamma
from .c_functions cimport next_twiss
from .exceptions import ParticleLost
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "<math.h>":
    double sin(double x)

cdef extern from "<math.h>":
    double cos(double x)

cdef extern from "<math.h>":
    double sinh(double x)

cdef extern from "<math.h>":
    double cosh(double x)

cdef extern from "<math.h>":
    double sqrt(double x)

cdef extern from "<math.h>":
    double pow(double x, double y)

cdef extern from "<math.h>":
    double tan(double x)

cdef class HBend(Element):
    """HBend(name: str = None, length: float = 0, theta: float = 0, theta_in: float = 0, theta_out: float = 0,
                 k1: float = 0, n_slices: int = 1, Ax: float = 10, Ay: float = 10)
    
    horizontal Bend.
    """

    def __init__(self, name: str = None, length: float = 0, theta: float = 0, theta_in: float = 0, theta_out: float = 0,
                 k1: float = 0, n_slices: int = 1, Ax: float = 10, Ay: float = 10):
        self.name = name
        self.length = length
        self.h = theta / self.length
        self.theta_in = theta_in
        self.theta_out = theta_out
        self.k1 = k1
        self.n_slices = n_slices
        self.Ax = Ax
        self.Ay = Ay

    @property
    def theta(self):
        return self.h * self.length

    @property
    def matrix(self):
        cdef double[6][6] matrix
        hbend_matrix(matrix, self.length, self.h, self.theta_in, self.theta_out, self.k1)
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
        k0 = self.h
        k1 = self.k1
        d1_square = 1 + 2 * dp0 / beta0 + dp0 * dp0
        if d1_square <= 0:
            return -1
        d1 = sqrt(d1_square)
        if k0 == 0 and k1 == 0:
            particle[0] = x0 + ds * particle[1] / d1
            particle[2] = y0 + ds * particle[3] / d1
            particle[4] = ct0 + ds * (1 - (1 + refbeta * particle[5]) / d1) / refbeta
            if (particle[0] / self.Ax) ** 2 + (particle[2] / self.Ay) ** 2 > 1:
                return -1
            return 0
        r10 = k0 * tan(self.theta_in)
        r32 = -k0 * tan(self.theta_in)

        px1 = px0 + r10 * x0
        py1 = py0 + r32 * y0

        # % Then, apply a map for the body of the dipole
        h = self.h
        a1 = h - k0 / d1


        wx_square = (h * k0 + k1) / d1
        if wx_square >= 0:
            xc = cos(sqrt(wx_square) * ds)
            xs = sin(sqrt(wx_square) * ds) / sqrt(wx_square)
            xs2 = sin(2 * sqrt(wx_square) * ds) / sqrt(wx_square)
        else:
            xc = cosh(sqrt(-wx_square) * ds)
            xs = sinh(sqrt(-wx_square) * ds) / sqrt(-wx_square)
            xs2 = sinh(2 * sqrt(-wx_square) * ds) / sqrt(-wx_square)

        wy_square = k1 / d1
        if wy_square > 0:
            yc = cosh(sqrt(wy_square) * ds)
            ys = sinh(sqrt(wy_square) * ds) / sqrt(wy_square)
            ys2 = sinh(2 * sqrt(wy_square) * ds) / sqrt(wy_square)
        elif wy_square == 0:
            yc = 1
            ys = ds
            ys2 = 2 * ds
        elif wy_square < 0:
            yc = cos(sqrt(-wy_square) * ds)
            ys = sin(sqrt(-wy_square) * ds) / sqrt(-wy_square)
            ys2 = sin(2 * sqrt(-wy_square) * ds) / sqrt(-wy_square)            

        particle[0] = x0 * xc + px1 * xs / d1 + a1 * (1 - xc) / wx_square
        px2 = -d1 * wx_square * x0 * xs + px1 * xc + a1 * xs * d1

        particle[2] = y0 * yc + py1 * ys / d1
        py2 = d1 * wy_square * y0 * ys + py1 * yc

        d0 = 1 / beta0 + dp0

        c0 = (1 / beta0 - d0 / d1) * ds - d0 * a1 * (h * (ds - xs) + a1 * (2 * ds-xs2) / 8)/ wx_square / d1

        c1 = -d0 * (h * xs - a1* (2 * ds-xs2) / 4)/ d1

        c2 = -d0 * (h * (1 - xc) / wx_square + a1* xs* xs / 2) / d1 / d1

        c11 = -d0 * wx_square * (2 * ds - xs2) / d1 / 8
        c12 = d0 * wx_square * xs * xs / d1 / d1 / 2
        c22 = -d0 * (2 * ds + xs2) / d1 / d1 / d1 / 8

        c33 = -d0 * wy_square * (2 * ds - ys2) / d1 / 8
        c34 = -d0 * wy_square * ys * ys / d1 / d1 / 2
        c44 = -d0 * (2 * ds + ys2) / d1 / d1 / d1 / 8

        particle[4] = ct0 + c0 + c1 * x0 + c2 * px1 + c11 * x0 * x0 + c12 * x0 * px1 + c22 * px1 * px1 + c33 * y0 * y0 + c34 * y0 * py1 + c44 * py1 * py1

        r10 = k0 * tan(self.theta_out)
        r32 = -k0 * tan(self.theta_out)

        particle[1] = px2 + r10 * particle[0]
        particle[3] = py2 + r32 * particle[2]
        if (particle[0] / self.Ax) ** 2 + (particle[2] / self.Ay) ** 2 > 1:
            return -1
        return 0

    cdef __radiation_integrals(self,double length,double[7] integrals,double[12] twiss0,double[12] twiss1):
        betax = (twiss0[0] + twiss1[0]) / 2
        alphax = (twiss0[1] + twiss1[1]) / 2
        gammax = (twiss0[2] + twiss1[2]) / 2
        betay = (twiss0[3] + twiss1[3]) / 2
        etax = (twiss0[6] + twiss1[6]) / 2
        etaxp = (twiss0[7] + twiss1[7]) / 2
        integrals[0] += length * etax * self.h
        integrals[1] += length * self.h ** 2
        integrals[2] += length * abs(self.h) ** 3
        integrals[3] += length * (self.h ** 2 + 2 * self.k1) * etax * self.h
        curl_H = gammax * etax ** 2 + 2 * alphax * etax * etaxp + betax * etaxp ** 2
        integrals[4] += length * curl_H * abs(self.h) ** 3
        integrals[5] += (- (self.k1 + self.h ** 2) * length * betax) / 4 / pi
        integrals[6] += (self.k1 * length * betay) / 4 / pi

    cpdef copy(self):
        return HBend(self.name, self.length, self.theta, self.theta_in, self.theta_out, self.k1, self.n_slices, self.Ax, self.Ay)
    
    @cython.cdivision(True)
    cpdef linear_optics(self):
        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]        
        cdef double[6][6] matrix
        cdef double[7] integrals=[0, 0, 0, 0, 0, 0, 0]
        cdef double[12] twiss1
        cdef double current_s, length
        current_s = 0
        length = 0.01
        # inlet
        integrals[3] += - self.h ** 2 * self.etax * tan(self.theta_in)
        integrals[5] += (self.h * tan(self.theta_in) * self.betax - 2 * self.k1 * tan(self.theta_in) * self.etax * self.betax) / 4 / pi
        integrals[6] += (- self.h * tan(self.theta_in) * self.betay + 2 * self.k1 * tan(self.theta_in) * self.etax * self.betay) / 4 / pi
        hbend_matrix(matrix, length, self.h, self.theta_in, 0, self.k1)
        next_twiss(matrix, twiss0, twiss1)
        self.__radiation_integrals(length, integrals, twiss0, twiss1)
        current_s = current_s + length
        for i in range(12):
            twiss0[i] = twiss1[i]
        while current_s < self.length - length:
            hbend_matrix(matrix, length, self.h, 0, 0, self.k1)
            next_twiss(matrix, twiss0, twiss1)
            self.__radiation_integrals(length, integrals, twiss0, twiss1)
            current_s = current_s + length
            for j in range(12):
                twiss0[j] = twiss1[j]
        length = self.length - current_s
        hbend_matrix(matrix, length, self.h, 0, self.theta_out, self.k1)
        next_twiss(matrix, twiss0, twiss1)
        self.__radiation_integrals(length, integrals, twiss0, twiss1)
        integrals[3] += - self.h ** 2 * twiss1[6] * tan(self.theta_out)
        integrals[5] += (self.h * tan(self.theta_out) * twiss1[0] - 2 * self.k1 * tan(self.theta_out) * twiss1[6] * twiss1[0]) / 4 / pi
        integrals[6] += (- self.h * tan(self.theta_out) * twiss1[3] + 2 * self.k1 * tan(self.theta_out) * twiss1[6] * twiss1[3]) / 4 / pi
        return np.array(integrals), np.array(twiss1)
    
    @cython.cdivision(True)
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
        hbend_matrix(matrix, 0, self.h, self.theta_in, 0, self.k1)
        next_twiss(matrix, twiss0, twiss1)
        for i in range(self.n_slices):
            hbend_matrix(matrix, length, self.h, 0, 0, self.k1)
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

    def slice(self, n_slices: int) -> list:
        """slice component to element list, return [ele_list, final_z]

        this method is rewritten because of the edge angles."""

        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]
        cdef double current_s = self.s
        cdef double[6][6] matrix
        cdef double[12] twiss1
        ele_list = []
        if n_slices == 1:
            ele = HBend(self.name, self.length, self.theta, self.theta_in, self.theta_out, self.k1, self.n_slices, self.Ax, self.Ay)
            ele.s = current_s
            assin_twiss(ele, twiss0)
            return [ele]
        length = self.length / n_slices
        ele = HBend(self.name, length, self.h * length, self.theta_in, 0, self.k1, self.n_slices, self.Ax, self.Ay)
        ele.s = current_s
        assin_twiss(ele, twiss0)
        hbend_matrix(matrix, length, self.h, self.theta_in, 0, self.k1)
        next_twiss(matrix, twiss0, twiss1)
        for j in range(12):
            twiss0[j] = twiss1[j]
        ele_list.append(ele)
        current_s = current_s + ele.length
        for i in range(n_slices - 2):
            ele = HBend(self.name, length, self.h * length, 0, 0, self.k1, self.n_slices, self.Ax, self.Ay)
            ele.s = current_s
            assin_twiss(ele, twiss0)
            hbend_matrix(matrix, length, self.h, 0, 0, self.k1)
            next_twiss(matrix, twiss0, twiss1)
            for j in range(12):
                twiss0[j] = twiss1[j]
            ele_list.append(ele)
            current_s = current_s + ele.length
        length = self.length + self.s - current_s
        ele = HBend(self.name, length, self.h * length, 0, self.theta_out, self.k1, self.n_slices, self.Ax, self.Ay)
        ele.s = current_s
        assin_twiss(ele, twiss0)
        ele_list.append(ele)
        return ele_list

    def __str__(self):
        text = str(self.name)
        text += (' ' * max(0, 6 - len(self.name)))
        text += (': ' + str(self.type))
        text += (':   s = ' + str(self.s))
        text += f',   length = {self.length: .6f}'
        theta = self.theta * 180 / pi
        text += f',   theta = {theta: .6f}'
        text += f',   theta_in = {self.theta_in * 180 / pi: .6f}'
        text += f',   theta_out = {self.theta_out * 180 / pi: .6f}'
        if self.k1 != 0:
            text += f',   k1 = {self.k1: .6f}'
        return text

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cdef hbend_matrix(double[6][6] matrix, double length,double h,double theta_in,double theta_out,double k1):
    cdef double h_beta, fx, cx, sx, dx, fy, cy, sy, dy, m56
    cdef double[3] csd
    cdef double[6][6] middle=[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    cdef double[6][6] temp
    cdef double[6][6] inlet
    cdef double[6][6] outlet
    h_beta = h / refbeta
    fx = k1 + pow(h, 2)
    calculate_csd(length, fx, csd)
    cx = csd[0]
    sx = csd[1]
    dx = csd[2]
    if fx != 0:
        m56 = length / pow(refgamma, 2) / pow(refbeta, 2) - pow(h, 2) * (length - sx) / fx
    else:
        m56 = length / pow(refgamma, 2) / pow(refbeta, 2) - pow(h, 2) * pow(length, 3) / 6
    fy = - k1
    calculate_csd(length, fy, csd)
    cy = csd[0]
    sy = csd[1]
    dy = csd[2]    
    middle[0][0] = cx
    middle[0][1] = sx
    middle[0][5] = h_beta * dx
    middle[1][0] = -fx * sx
    middle[1][1] = cx
    middle[1][5] = h_beta * sx
    middle[2][2] = cy
    middle[2][3] = sy
    middle[3][2] = -fy * sy
    middle[3][3] = cy
    middle[4][0] = h_beta * sx
    middle[4][1] = h_beta * dx
    middle[4][5] = - m56
    middle[5][5] = 1
    middle[4][4] = 1
    if theta_in != 0:
        for i in range(6):
            for j in range(6):
                if i == j:
                    inlet[i][j] = 1
                else:
                    inlet[i][j] = 0
        inlet[1][0] = tan(theta_in) * h
        inlet[3][2] = -tan(theta_in) * h
        for i in range(6):
            for j in range(6):
                temp[i][j] = 0
                for k in range(6):
                    temp[i][j] += middle[i][k] * inlet[k][j]
    else:
        for i in range(6):
            for j in range(6):
                temp[i][j] = middle[i][j]
    if theta_out != 0:
        for i in range(6):
            for j in range(6):
                if i == j:
                    outlet[i][j] = 1
                else:
                    outlet[i][j] = 0
        outlet[1][0] = tan(theta_out) * h
        outlet[3][2] = -tan(theta_out) * h
        for i in range(6):
            for j in range(6):
                matrix[i][j] = 0
                for k in range(6):
                    matrix[i][j] += outlet[i][k] * temp[k][j]
    else:
        for i in range(6):
            for j in range(6):
                matrix[i][j] = temp[i][j]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cdef int calculate_csd(double length, double fu, double[3] csd):
    cdef double sqrt_fu_z
    if fu > 0:
        sqrt_fu_z = sqrt(fu) * length
        csd[0] = cos(sqrt_fu_z)
        csd[1] = sin(sqrt_fu_z) / sqrt(fu)
        csd[2] = (1 - csd[0]) / fu
    elif fu < 0:
        sqrt_fu_z = sqrt(-fu) * length
        csd[0] = cosh(sqrt_fu_z)
        csd[1] = sinh(sqrt_fu_z) / sqrt(-fu)
        csd[2] = (1 - csd[0]) / fu
    else:
        csd[0] = 1
        csd[1] = length
        csd[2] = pow(length, 2) / 2
    return 0