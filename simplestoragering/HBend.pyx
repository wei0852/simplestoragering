# -*- coding: utf-8 -*-
# cython: language_level=3
from .components cimport Element, assin_twiss
from .globalvars cimport Cr, pi, refbeta, refgamma
from .c_functions cimport next_twiss, track_matrix
from .exceptions import ParticleLost
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, cos, sin, sinh, cosh, pow, tan
from libc.stdlib cimport malloc, free


cdef extern from "BndMPoleSymplectic4Pass.c":
    void BndMPoleSymplectic4Pass(double *r, double le, double irho, double *A, double *B,
        int max_order, int num_int_steps,
        double entrance_angle, double exit_angle,
        int FringeBendEntrance, int FringeBendExit,
        double fint1, double fint2, double gap)

cdef class HBend(Element):
    """HBend(name: str = None, length: float = 0, theta: float = 0, theta_in: float = 0, theta_out: float = 0, k1: float = 0, gap: double = 0, fint_in: float = 0.5, fint_out: float = 0.5, n_slices: int = 10, Ax: float = 10, Ay: float = 10)
    
    horizontal Bend.

    edge_method:
    /*     method 0 no fringe field
     *     method 1 legacy version Brown First Order
     *     method 2 SOLEIL close to second order of Brown
     *     method 3 THOMX
     */
    """

    def __init__(self, name: str = None, length: float = 0, theta: float = 0, theta_in: float = 0, theta_out: float = 0,
                 k1: float = 0, gap: float = 0, fint_in: float = 0.5, fint_out: float = 0.5, n_slices: int = 10, k2: float = 0, k3: float = 0, edge_method: int = 2, Ax: float = 10, Ay: float = 10):
        self.name = name
        self.length = length
        self.h = theta / self.length
        self.theta_in = theta_in
        self.theta_out = theta_out
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.gap = gap
        self.fint_in = fint_in
        self.fint_out = fint_out
        self.n_slices = n_slices
        self.edge_method = edge_method
        self.Ax = Ax
        self.Ay = Ay

    @property
    def theta(self):
        return self.h * self.length

    @property
    def matrix(self):
        cdef double[6][6] matrix
        hbend_matrix(matrix, self.length, self.h, self.theta_in, self.theta_out, self.k1, self.gap, self.fint_in, self.fint_out)
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
        k0 = self.h
        k1 = self.k1
        d1_square = 1 + 2 * dp0 / beta0 + dp0 * dp0
        if d1_square <= 0:
            return -1
        d1 = sqrt(d1_square)
        if k0 == 0 and k1 == 0:  # drift
            particle[0] = x0 + ds * particle[1] / d1
            particle[2] = y0 + ds * particle[3] / d1
            particle[4] = ct0 + ds * (1 - (1 + refbeta * particle[5]) / d1) / refbeta
            if (particle[0] / self.Ax) ** 2 + (particle[2] / self.Ay) ** 2 > 1:
                return -1
            return 0
        sin_edge = sin(self.theta_in)
        cos_edge = cos(self.theta_in)
        r10 = k0 * tan(self.theta_in)
        if self.edge_method == 1:
            r32 = -k0 * tan(self.theta_in - self.h * self.gap * self.fint_in * (1 + sin_edge * sin_edge) / cos_edge / (1 + particle[5]))
        elif self.edge_method == 2:
            r32 = -k0 * tan(self.theta_in - self.h * self.gap * self.fint_in * (1 + sin_edge * sin_edge) / cos_edge / (1 + particle[5])) /  (1 + particle[5])

        px1 = px0 + r10 * x0
        py1 = py0 + r32 * y0

        # % Then, apply a map for the body of the dipole
        h = self.h
        a1 = h - k0 / d1


        wx_square = (h * k0 + k1) / d1
        if wx_square > 0:
            xc = cos(sqrt(wx_square) * ds)
            xs = sin(sqrt(wx_square) * ds) / sqrt(wx_square)
            xs2 = sin(2 * sqrt(wx_square) * ds) / sqrt(wx_square)
        elif wx_square == 0:
            xc = 1
            xs = ds
            xs2 = 2 * ds
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

        sin_edge = sin(self.theta_out)
        cos_edge = cos(self.theta_out)
        r10 = k0 * tan(self.theta_out)
        if self.edge_method == 1:
            r32 = -k0 * tan(self.theta_out - self.h * self.gap * self.fint_out * (1 + sin_edge * sin_edge) / cos_edge / (1 + particle[5]))
        elif self.edge_method == 2:
            r32 = -k0 * tan(self.theta_out - self.h * self.gap * self.fint_out * (1 + sin_edge * sin_edge) / cos_edge / (1 + particle[5])) /  (1 + particle[5])

        particle[1] = px2 + r10 * particle[0]
        particle[3] = py2 + r32 * particle[2]
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
        BndMPoleSymplectic4Pass(<double *> &r, self.length, self.h, A, B, max_order, self.n_slices,
                                    self.theta_in, self.theta_out, self.edge_method, self.edge_method, self.fint_in, self.fint_out, self.gap)
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

    cdef __radiation_integrals(self,double length,double[7] integrals,double[12] twiss0,double[12] twiss1):
        # 0: entrance
        # 1: after entrance angle  (beta0 == beta1)
        # 2: before exit angle  (beta2 == beta3)
        # 3: exit

        betax0 = twiss0[0]
        betaz0 = twiss0[3]
        alphax0 = twiss0[1]
        alphaz0 = twiss0[4]
        eta0 = twiss0[6]
        etap0 = twiss0[7]

        ll = self.length
        rho = 1/self.h
        rho2 = rho * rho
        K = self.k1
        kx2 = K + 1.0 / rho2
        kz2 = -K
        eps1 = tan(self.theta_in) / rho
        eps2 = tan(self.theta_out) / rho
    
        alphax1 = alphax0 - betax0 * eps1
        alphaz1 = alphaz0 + betaz0 * eps1
        gammax1 = (1.0 + alphax1 * alphax1) / betax0
        gammaz1 = (1.0 + alphaz1 * alphaz1) / betaz0
        etap1 = etap0 + eta0 * eps1

        eta3 = twiss1[6]
        betax3 = twiss1[0]
        betaz3 = twiss1[3]
        etap2 = twiss1[7] - eta3 * eps2
        alphax2 = twiss1[1] + betax3 * eps2
        alphaz2 = twiss1[4] - betaz3 * eps2
    
        h0 = gammax1 * eta0 * eta0 + 2.0 * alphax1 * eta0 * etap1 + betax0 * etap1 * etap1
    
        if kx2 != 0.0:
            if kx2 > 0.0:  # Focusing
                kl = ll * sqrt(kx2)
                ss = sin(kl) / kl
                cc = cos(kl)
            else:  # Defocusing
                kl = ll * sqrt(-kx2)
                ss = sinh(kl) / kl
                cc = cosh(kl)
            betax_ave = ((gammax1 + betax0 * kx2) + (alphax2 - alphax1) / ll) / 2 / kx2
            eta_ave = (ll / rho - (etap2 - etap1)) / kx2 / ll
            bb = 2.0 * (alphax1 * eta0 + betax0 * etap1) * rho
            aa = -2.0 * (alphax1 * etap1 + gammax1 * eta0) * rho
            h_ave = h0 + (aa * (1.0 - ss) + bb * (1.0 - cc) / ll 
                         + gammax1 * (3.0 - 4.0 * ss + ss * cc) / 2.0 / kx2 
                         - alphax1 * (1.0 - cc) ** 2 / kx2 / ll 
                         + betax0 * (1.0 - ss * cc) / 2.0) / kx2 / rho2
        else:
            betax_ave = betax0 - alphax1 * ll + gammax1 * ll**2 / 3

            eta_ave = 0.5 * (eta0 + eta3) - ll * ll / 12.0 / rho
            hp0 = 2.0 * (alphax1 * eta0 + betax0 * etap1) / rho
            h2p0 = 2.0 * (-alphax1 * etap1 + betax0 / rho - gammax1 * eta0) / rho
            h_ave = (h0 + hp0 * ll / 2.0 + h2p0 * ll * ll / 6.0 
                     - alphax1 * ll ** 3/4.0 / rho2 
                     + gammax1 * ll ** 4/20.0 / rho2)
        if kz2 != 0:
            betaz_ave = ((gammaz1 + betaz0 * kz2) + (alphaz2 - alphaz1) / ll) / 2 / kz2
        else:
            betaz_ave = betaz0 - alphaz1 * ll + gammaz1 * ll**2 / 3

        integrals[0] += eta_ave * ll / rho
        integrals[1] += ll / rho2
        integrals[2] += ll / abs(rho) / rho2
        integrals[3] += eta_ave * ll * (2.0 * K + 1.0 / rho2) / rho - (eta0 * eps1 + eta3 * eps2) / rho
        integrals[4] += h_ave * ll / abs(rho) / rho2
        integrals[5] += (-betax_ave * ll * kx2) / 4 / pi
        integrals[6] += (-betaz_ave * ll * kz2) / 4 / pi

        sin_edge = sin(self.theta_in)
        cos_edge = cos(self.theta_in)
        integrals[5] += (self.h * tan(self.theta_in) * betax0) / 4 / pi
        integrals[6] += (- self.h * tan(self.theta_in - self.h * self.gap * self.fint_in * (1 + sin_edge * sin_edge) / cos_edge) * betaz0) / 4 / pi

        sin_edge = sin(self.theta_out)
        cos_edge = cos(self.theta_out)
        integrals[5] += (self.h * eps2 * rho * betax3) / 4 / pi
        integrals[6] += (- self.h * tan(self.theta_out - self.h * self.gap * self.fint_out * (1 + sin_edge * sin_edge) / cos_edge) * betaz3) / 4 / pi

    cpdef copy(self):
        return HBend(self.name, self.length, self.theta, self.theta_in, self.theta_out, self.k1, self.gap, self.fint_in, self.fint_out, self.n_slices, self.k2, self.k3, self.edge_method, self.Ax, self.Ay)
    
    @cython.cdivision(True)
    cpdef linear_optics(self):
        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]        
        cdef double[6][6] matrix
        cdef double[7] integrals=[0, 0, 0, 0, 0, 0, 0]
        cdef double[12] twiss1
        hbend_matrix(matrix, self.length, self.h, self.theta_in, self.theta_out, self.k1, self.gap, self.fint_in, self.fint_out)
        next_twiss(matrix, twiss0, twiss1)
        self.__radiation_integrals(self.length, integrals, twiss0, twiss1)
        return np.array(integrals), np.array(twiss1)

    def slice(self, n_slices: int) -> list:
        """slice component to element list, return [ele_list, final_z]

        this method is rewritten because of the edge angles."""

        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]
        cdef double current_s = self.s
        cdef double[6][6] matrix
        cdef double[6] closed_orbit
        cdef double[12] twiss1
        ele_list = []
        closed_orbit = self.closed_orbit
        if n_slices == 1:
            ele = HBend(self.name, self.length, self.theta, self.theta_in, self.theta_out, self.k1, self.gap, self.fint_in, self.fint_out, self.n_slices, self.k2, self.k3, self.edge_method, self.Ax, self.Ay)
            ele.s = current_s
            assin_twiss(ele, twiss0)
            return [ele]
        length = self.length / n_slices
        sub_slices = max(int(self.n_slices / n_slices), 1)
        ele = HBend(self.name, self.length, self.theta, self.theta_in, self.theta_out, self.k1, self.gap, self.fint_in, self.fint_out, self.n_slices, self.k2, self.k3, self.edge_method, self.Ax, self.Ay)
        ele.length = length
        ele.n_slices = sub_slices
        ele.theta_out = 0.0
        ele.fint_out = 0.0
        ele.s = current_s
        assin_twiss(ele, twiss0)
        track_matrix(ele, matrix)
        ele.closed_orbit = closed_orbit
        ele.symplectic_track(closed_orbit)        
        next_twiss(matrix, twiss0, twiss1)
        for j in range(12):
            twiss0[j] = twiss1[j]
        ele_list.append(ele)
        current_s = current_s + ele.length
        for i in range(n_slices - 2):
            ele = HBend(self.name, self.length, self.theta, self.theta_in, self.theta_out, self.k1, self.gap, self.fint_in, self.fint_out, self.n_slices, self.k2, self.k3, self.edge_method, self.Ax, self.Ay)
            ele.length = length
            ele.n_slices = sub_slices
            ele.theta_in = 0.0
            ele.fint_in = 0.0
            ele.theta_out = 0.0
            ele.fint_out = 0.0
            ele.closed_orbit = closed_orbit
            ele.s = current_s
            assin_twiss(ele, twiss0)
            track_matrix(ele, matrix)
            ele.symplectic_track(closed_orbit)      
            next_twiss(matrix, twiss0, twiss1)
            for j in range(12):
                twiss0[j] = twiss1[j]
            ele_list.append(ele)
            current_s = current_s + ele.length
        length = self.length + self.s - current_s
        ele = HBend(self.name, self.length, self.theta, self.theta_in, self.theta_out, self.k1, self.gap, self.fint_in, self.fint_out, self.n_slices, self.k2, self.k3, self.edge_method, self.Ax, self.Ay)
        ele.length = length
        ele.n_slices = sub_slices
        ele.theta_in = 0.0
        ele.fint_in = 0.0
        ele.closed_orbit = closed_orbit
        ele.s = current_s
        assin_twiss(ele, twiss0)
        ele_list.append(ele)
        return ele_list

    cpdef driving_terms(self, delta):
        cdef double[12] twiss0 = [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy]
        cdef double[6][6] matrix
        cdef double[12] twiss1
        cdef double length, b4l, b3l
        cdef complex h21000, h30000, h10110, h10020, h10200, jj
        cdef complex h21000j, h30000j, h10110j, h10020j, h10200j
        cdef complex  h22000, h11110, h00220, h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400
        jj = complex(0, 1)
        h21000 = h30000 = h10110 = h10020 = h10200 = 0
        h22000 = h11110 = h00220 = h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
        length = self.length / 4
        b4l = length * self.k3 / 6 / (1 + delta)
        b3l_0 = length * self.k2 / 2 / (1 + delta)
        if not b4l and not b3l_0:
            return np.zeros(16)
        bend_slices = self.slice(4)
        for i in range(4):
            xco = bend_slices[i].closed_orbit[0]
            b3l = b4l * 3 * xco + b3l_0
            track_matrix(bend_slices[i], matrix)
            next_twiss(matrix, twiss0, twiss1)
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
cdef hbend_matrix(double[6][6] matrix, double length,double h,double theta_in,double theta_out,double k1, double gap, double fint_in, double fint_out):
    cdef double h_beta, fx, cx, sx, dx, fy, cy, sy, dy, m56, sin_edge, cos_edge
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
        sin_edge = sin(theta_in)
        cos_edge = cos(theta_in)
        inlet[1][0] = tan(theta_in) * h
        inlet[3][2] = -tan(theta_in - h * gap * fint_in * (1 + sin_edge * sin_edge) / cos_edge) * h
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
        sin_edge = sin(theta_out)
        cos_edge = cos(theta_out)
        outlet[1][0] = tan(theta_out) * h
        outlet[3][2] = -tan(theta_out - h * gap * fint_out * (1 + sin_edge * sin_edge) / cos_edge) * h
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