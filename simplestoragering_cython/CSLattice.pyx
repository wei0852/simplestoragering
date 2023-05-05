# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: profile=True
import numpy as np
cimport numpy as np
cimport cython
from .components cimport LineEnd, Mark
from .c_functions cimport symplectic_track_ele
from .line_matrix cimport line_matrix
from .globalvars cimport pi, c, Cq, Cr, refgamma, refenergy, refbeta
from .HBend cimport HBend
from .Drift cimport Drift
from .Quadrupole cimport Quadrupole
from .Sextupole cimport Sextupole
from .Octupole cimport Octupole
from .DrivingTerms import DrivingTerms
import warnings


cdef extern from "<math.h>":
    double sin(double x)

cdef extern from "<math.h>":
    double cos(double x)

cdef extern from "<math.h>":
    double acos(double x)

cdef extern from "<math.h>":
    double sqrt(double x)

cdef extern from "<math.h>":
    double pow(double x, double y)

cdef extern from "<math.h>":
    double fabs(double x)


class CSLattice(object):
    """CSLattice(ele_list: list[Elements], n_periods: int = 1, coupling: float = 0.00)
    lattice object, solve by Courant-Snyder method.

    Attributes:
        length: float
        n_periods: int, number of periods
        angle, abs_angle: float
        elements: list of Elements
        mark: Dictionary of Mark in lattice. The key is the name of Mark, and the value is a list of Mark with the same name.
        twiss_x0, twiss_y0: np.ndarray, [beta, alpha, gamma]
        eta_x0, eta_y0: np.ndarray, [eta$, eta']. eta_y0=[0,0] because the coupled motion has not be considered yet.
        nux, nuy: Tunes
        xi_x, xi_y: Chromaticities
        natural_xi_x, natural_xi_y: Natural chromaticities
        I1, I2, I3, I4, I5: radiation integrals.
        Jx, Jy, Js: horizontal / vertical / longitudinal damping partition number
        sigma_e: natural energy spread
        emittance: natural emittance
        U0: energy loss [MeV]
        f_c: frequency
        tau_s, tau_x, tau_y: longitudinal / horizontal / vertical damping time
        alpha: Momentum compaction
        etap: phase slip factor
    
    Methods:
        set_initial_twiss(betax, alphax, betay, alphay, etax, etaxp, etay, etayp)
        linear_optics(periodicity=True, line_mode=False)
        driving_terms(printout=True)
        nonlinear_terms_change()
        driving_terms_plot_data()
        higher_order_chromaticity(printout=True, order=3, delta=1e-3, matrix_precision=1e-9, resdl_limit=1e-16)

        slice_elements(drift_maxlength=10.0, bend_maxlength=10.0, quad_maxlength=10.0, sext_maxlength=10.0)
        output_twiss(file_name: str = u'twiss_data.txt')
    """

    def __init__(self, ele_list: list, n_periods: int = 1, coupling: float = 0.00):
        self.length = 0
        self.n_periods = n_periods
        self.coup = coupling
        self.elements = []
        self.mark = {}
        self.angle = 0
        self.abs_angle = 0
        current_s = 0
        for oe in ele_list:
            ele = oe.copy()
            ele.s = current_s
            if isinstance(ele, Mark):
                if ele.name in self.mark:
                    self.mark[ele.name].append(ele)
                else:
                    self.mark[ele.name] = [ele]
            self.elements.append(ele)
            self.length = self.length + ele.length
            if isinstance(ele, HBend):
                self.angle += ele.theta
                self.abs_angle += abs(ele.theta)
            current_s = current_s + ele.length
        last_ele = LineEnd(s=self.length)
        self.elements.append(last_ele)
        self.angle = self.angle * 180 / pi * n_periods
        self.abs_angle = self.abs_angle * 180 / pi * n_periods
        self.length = self.length * n_periods
        # initialize twiss
        self.twiss_x0 = None
        self.twiss_y0 = None
        self.eta_x0 = None
        self.eta_y0 = None
        # solve twiss
        self.nux = None
        self.nuy = None
        self.nuz = None
        # integration
        self.xi_x = None
        self.xi_y = None
        self.natural_xi_x = None
        self.natural_xi_y = None
        self.I1 = None
        self.I2 = None
        self.I3 = None
        self.I4 = None
        self.I5 = None
        # global parameters
        self.Jx = None
        self.Jy = None
        self.Js = None
        self.sigma_e = None
        self.emittance = None
        self.U0 = None
        self.f_c = None
        self.tau0 = None
        self.tau_s = None
        self.tau_x = None
        self.tau_y = None
        self.alpha = None
        self.emitt_x = None
        self.emitt_y = None
        self.etap = None
        self.sigma_z = None

    def linear_optics(self, periodicity=True, line_mode=False):
        """linear_optics(self, periodicity=True, line_mode=False)
        calculate optical functions and storage ring parameters.
        periodicity: if True, the periodic solution will be the initial twiss data. Otherwise initial twiss should be set
                    by CSLattice.set_initial_twiss(betax, alphax, betay, alphay, etax, etaxp, etay, etayp)
        line_mode: if True, the storage ring parameters such as emittance and damping time are not calculated."""

        if periodicity:
            self._the_periodic_solution()
        else:
            if self.twiss_x0 is None or self.twiss_y0 is None or self.eta_x0 is None or self.eta_y0 is None:
                raise Exception('need initial twiss data. use set_initial_twiss() or linear_optics(periodicity=True)')
        self.__solve_along()
        #  global_parameters
        self.U0 = Cr * refenergy ** 4 * self.I2 / (2 * pi)
        if not line_mode:
            np.seterr(all='raise')
            self.f_c = c * refbeta / self.length
            self.Jx = 1 - self.I4 / self.I2
            self.Jy = 1
            self.Js = 2 + self.I4 / self.I2
            self.sigma_e = refgamma * np.sqrt(Cq * self.I3 / (self.Js * self.I2))
            self.emittance = Cq * refgamma * refgamma * self.I5 / (self.Jx * self.I2)
            self.tau0 = 2 * refenergy / self.U0 / self.f_c
            self.tau_s = self.tau0 / self.Js
            self.tau_x = self.tau0 / self.Jx
            self.tau_y = self.tau0 / self.Jy
            self.alpha = self.I1 * self.f_c / c  # momentum compaction factor
            self.emitt_x = self.emittance / (1 + self.coup)
            self.emitt_y = self.emittance * self.coup / (1 + self.coup)
            self.etap = self.alpha - 1 / refgamma ** 2  # phase slip factor

    def _the_periodic_solution(self):
        """compute periodic solution and initialize twiss"""

        matrix = line_matrix(self.elements)
        # x direction
        cos_mu = (matrix[0, 0] + matrix[1, 1]) / 2
        assert fabs(cos_mu) < 1, "can not find period solution, UNSTABLE!!!"
        mu = acos(cos_mu) * np.sign(matrix[0, 1])
        beta = matrix[0, 1] / sin(mu)
        alpha = (matrix[0, 0] - matrix[1, 1]) / (2 * sin(mu))
        gamma = - matrix[1, 0] / sin(mu)
        self.twiss_x0 = np.array([beta, alpha, gamma])
        # y direction
        cos_mu = (matrix[2, 2] + matrix[3, 3]) / 2
        assert fabs(cos_mu) < 1, "can not find period solution, UNSTABLE!!!"
        mu = acos(cos_mu) * np.sign(matrix[2, 3])
        beta = matrix[2, 3] / sin(mu)
        alpha = (matrix[2, 2] - matrix[3, 3]) / (2 * sin(mu))
        gamma = - matrix[3, 2] / sin(mu)
        self.twiss_y0 = np.array([beta, alpha, gamma])
        # solve eta
        sub_matrix_x = matrix[0:2, 0:2]
        matrix_etax = np.array([matrix[0, 5], matrix[1, 5]])
        self.eta_x0 = np.linalg.inv(np.identity(2) - sub_matrix_x).dot(matrix_etax)
        sub_matrix_y = matrix[2:4, 2:4]
        matrix_etay = np.array([matrix[2, 5], matrix[3, 5]])
        self.eta_y0 = np.linalg.inv(np.identity(2) - sub_matrix_y).dot(matrix_etay)

    def set_initial_twiss(self, betax, alphax, betay, alphay, etax, etaxp, etay, etayp):
        """set_initial_twiss(betax, alphax, betay, alphay, etax, etaxp, etay, etayp)
        work if run CSLattice.linear_optics() with periodicity=False.
        """
        self.twiss_x0 = np.array([betax, alphax, (1 + alphax**2) / betax])
        self.twiss_y0 = np.array([betay, alphay, (1 + alphay**2) / betay])
        self.eta_x0 = np.array([etax, etaxp])
        self.eta_y0 = np.array([etay, etayp])

    def __solve_along(self):
        [betax, alphax, gammax] = self.twiss_x0
        [betay, alphay, gammay] = self.twiss_y0
        [etax, etaxp] = self.eta_x0
        [etay, etayp] = self.eta_y0
        psix = psiy = integral1 = integral2 = integral3 = integral4 = integral5 = 0
        natural_xi_x = sextupole_part_xi_x = natural_xi_y = sextupole_part_xi_y = 0
        twiss = np.array([betax, alphax, gammax, betay, alphay, gammay, etax, etaxp, etay, etayp, psix, psiy])
        for ele in self.elements:
            ele.betax = twiss[0]
            ele.alphax = twiss[1]
            ele.gammax = twiss[2]
            ele.betay = twiss[3]
            ele.alphay = twiss[4]
            ele.gammay = twiss[5]
            ele.etax = twiss[6]
            ele.etaxp = twiss[7]
            ele.etay = twiss[8]
            ele.etayp = twiss[9]
            ele.psix = twiss[10]
            ele.psiy = twiss[11]
            [i1, i2, i3, i4, i5, xix, xiy], twiss = ele.linear_optics()
            integral1 += i1
            integral2 += i2
            integral3 += i3
            integral4 += i4
            integral5 += i5
            if isinstance(ele, Sextupole):
                sextupole_part_xi_x += xix
                sextupole_part_xi_y += xiy
            else:
                natural_xi_x += xix
                natural_xi_y += xiy
        self.I1 = integral1 * self.n_periods
        self.I2 = integral2 * self.n_periods
        self.I3 = integral3 * self.n_periods
        self.I4 = integral4 * self.n_periods
        self.I5 = integral5 * self.n_periods
        self.natural_xi_x = natural_xi_x * self.n_periods
        self.natural_xi_y = natural_xi_y * self.n_periods
        self.xi_x = (natural_xi_x + sextupole_part_xi_x) * self.n_periods
        self.xi_y = (natural_xi_y + sextupole_part_xi_y) * self.n_periods
        self.nux = self.elements[-1].nux * self.n_periods
        self.nuy = self.elements[-1].nuy * self.n_periods

    def slice_elements(self, drift_length=10.0, bend_length=10.0, quad_length=10.0, sext_length=10.0):
        """slice_elements(drift_length=10.0, bend_length=10.0, quad_length=10.0, sext_length=10.0)
        slice elements of ring, the twiss data of each slice will be calculated. To draw lattice functions smoothly.
        return list of elements."""
        ele_slices = []
        for ele in self.elements:
            if isinstance(ele, Drift):
                ele_slices += ele.slice(max(int(ele.length / drift_length), 1))
            elif isinstance(ele, HBend):
                ele_slices += ele.slice(max(int(ele.length / bend_length), 1))
            elif isinstance(ele, Quadrupole):
                ele_slices += ele.slice(max(int(ele.length / quad_length), 1))
            elif isinstance(ele, Sextupole):
                ele_slices += ele.slice(max(int(ele.length / sext_length), 1))
            else:
                ele_slices += ele.slice(1)
        return ele_slices
    
    @cython.cdivision(True)
    def adts(self, n_periods=None, printout=True):
        """adts(self, printout=True)
        compute ADTS terms. 
        Return:
            {'dQxx': , 'dQxy': , 'dQyy':}

        references:
        1. The Sextupole Scheme for the SLS: An Analytic Approach, SLS Note 09/97, Johan Bengtsson"""
        
        cdef double b3l_i, b4l, beta_xi, mu_ix, mu_iy, beta_yi
        cdef double b3l, beta_xj, beta_xij, beta_yj, mu_jx, mu_ijx, mu_jy, mu_ijy, Qxx, Qxy, Qyy, pi_nux, pi_nuy
        cdef int current_ind, sext_num, i, j
        cdef list sext_index, oct_index, ele_list
        ele_list = []
        sext_index = []
        oct_index = []
        current_ind = 0
        sext_num = 0
        for ele in self.elements:
            if isinstance(ele, Sextupole):
                ele_list += ele.slice(ele.n_slices)
                for i in range(ele.n_slices):
                    sext_index.append(current_ind)
                    current_ind += 1
            elif isinstance(ele, Octupole):
                ele_list += ele.slice(ele.n_slices)
                for i in range(ele.n_slices):
                    oct_index.append(current_ind)
                    current_ind += 1
            else:
                ele_list.append(ele)
                current_ind += 1
        Qxx = Qxy = Qyy = 0
        pi_nux = ele_list[current_ind-1].psix / 2
        pi_nuy = ele_list[current_ind-1].psiy / 2
        if sin(pi_nux) == 0 or sin(3 * pi_nux) == 0 or sin(pi_nux + 2 * pi_nuy) == 0 or sin(pi_nux - 2 * pi_nuy) == 0:
            nonlinear_terms = {'Qxx': 1e60, 'Qxy': 1e60, 'Qyy': 1e60}
            if printout:
                print('\n on resonance line.')
            return nonlinear_terms
        sext_num = len(sext_index)
        for i in range(sext_num):
            b3l_i = ele_list[sext_index[i]].k2 * ele_list[sext_index[i]].length / 2  # k2 = 2 * b3, k1 = b2
            if b3l_i != 0:
                beta_xi = (ele_list[sext_index[i]].betax + ele_list[sext_index[i] + 1].betax) / 2
                beta_yi = (ele_list[sext_index[i]].betay + ele_list[sext_index[i] + 1].betay) / 2
                mu_ix = (ele_list[sext_index[i]].psix + ele_list[sext_index[i] + 1].psix) / 2
                mu_iy = (ele_list[sext_index[i]].psiy + ele_list[sext_index[i] + 1].psiy) / 2
                Qxx += b3l_i ** 2 / (-16 * pi) * pow(beta_xi, 3) * (
                        3 * cos(0 - pi_nux) / sin(pi_nux)
                        + cos(3 * 0 - 3 * pi_nux) / sin(3 * pi_nux))
                Qxy += b3l_i ** 2 / (8 * pi) * beta_xi * beta_yi * (
                        2 * beta_xi * cos(pi_nux) / sin(pi_nux)
                        - beta_yi * cos(pi_nux + 2 * pi_nuy) / sin(pi_nux + 2 * pi_nuy)
                        + beta_yi * cos(pi_nux - 2 * pi_nuy) / sin(pi_nux - 2 * pi_nuy))
                Qyy += b3l_i ** 2 / (-16 * pi) * beta_xi * beta_yi * beta_yi * (
                        4 * cos(pi_nux) / sin(pi_nux)
                        + cos(pi_nux + 2 * pi_nuy) / sin(pi_nux + 2 * pi_nuy)
                        + cos(pi_nux - 2 * pi_nuy) / sin(pi_nux - 2 * pi_nuy))                
                for j in range(i):
                    b3l = ele_list[sext_index[j]].k2 * ele_list[sext_index[j]].length * b3l_i / 2
                    if b3l != 0:
                        beta_xj = (ele_list[sext_index[j]].betax + ele_list[sext_index[j] + 1].betax) / 2
                        beta_yj = (ele_list[sext_index[j]].betay + ele_list[sext_index[j] + 1].betay) / 2
                        mu_jx = (ele_list[sext_index[j]].psix + ele_list[sext_index[j] + 1].psix) / 2
                        mu_ijx = fabs(mu_ix - mu_jx)
                        mu_jy = (ele_list[sext_index[j]].psiy + ele_list[sext_index[j] + 1].psiy) / 2
                        mu_ijy = fabs(mu_iy - mu_jy)
                        beta_xij = beta_xj * beta_xi
                        Qxx += 2 * b3l / (-16 * pi) * pow(beta_xi * beta_xj, 1.5) * (
                                3 * cos(mu_ijx - pi_nux) / sin(pi_nux)
                                + cos(3 * mu_ijx - 3 * pi_nux) / sin(3 * pi_nux))
                        Qxy += 2 * b3l / (8 * pi) * pow(beta_xij, 0.5) * beta_yj * (
                                2 * beta_xi * cos(mu_ijx - pi_nux) / sin(pi_nux)
                                - beta_yi * cos(mu_ijx + 2 * mu_ijy - pi_nux - 2 * pi_nuy) / sin(pi_nux + 2 * pi_nuy)
                                + beta_yi * cos(mu_ijx - 2 * mu_ijy - pi_nux + 2 * pi_nuy) / sin(pi_nux - 2 * pi_nuy))
                        Qyy += 2 * b3l / (-16 * pi) * pow(beta_xij, 0.5) * beta_yj * beta_yi * (
                                4 * cos(mu_ijx - pi_nux) / sin(pi_nux)
                                + cos(mu_ijx + 2 * mu_ijy - pi_nux - 2 * pi_nuy) / sin(pi_nux + 2 * pi_nuy)
                                + cos(mu_ijx - 2 * mu_ijy - pi_nux + 2 * pi_nuy) / sin(pi_nux - 2 * pi_nuy))
        for i in oct_index:
            b4l = ele_list[i].k3 * ele_list[i].length / 6
            beta_xi = (ele_list[i].betax + ele_list[i + 1].betax) / 2
            beta_yi = (ele_list[i].betay + ele_list[i + 1].betay) / 2
            Qxx += 3 * b4l * beta_xi ** 2 / 8 / pi
            Qxy -= 3 * b4l * beta_xi * beta_yi / (4 * pi)
            Qyy += 3 * b4l * beta_yi ** 2 / 8 / pi

        n_periods = self.n_periods if n_periods is None else n_periods
        nonlinear_terms = {'dQxx': Qxx * n_periods, 'dQxy': Qxy * n_periods, 'dQyy': Qyy * n_periods}
        if printout:
            print(f'ADTS terms, {n_periods} periods:')
            for k, b4l in nonlinear_terms.items():
                print(f'    {str(k):7}: {b4l:.2f}')
        return nonlinear_terms

    def s_dependent_nonlinear_terms(self):
        """compute resonance driving terms of one period, and the starting position varies.
        Return: a dictionary, each value is a np.ndarray.
                {'s':, 'h21000': , 'h30000': , 'h10110': , 'h10020': ,
                 'h10200': , 'dQxx': , 'dQxy': , 'dQyy': ,
                 'h31000': , 'h40000': , 'h20110': , 'h11200': ,
                 'h20020': , 'h20200': , 'h00310': , 'h00400': }

        references:
        1. The Sextupole Scheme for the SLS: An Analytic Approach, SLS Note 09/97, Johan Bengtsson
        2. Explicit formulas for 2nd-order driving terms due to sextupoles and chromatic effects of quadrupoles, Chun-xi Wang
        3. Perspectives for future light source lattices incorporating yet uncommon magnets, S. C. Leemann and A. Streun"""
        
        cdef double b3l_i, beta_xi, mu_ix, mu_iy, beta_yi, periodic_psix, periodic_psiy
        cdef int current_ind
        cdef double h21000r, h21000i, h30000r, h30000i, h10110r, h10110i, h10020r, h10020i, h10200r, h10200i
        cdef double b3l, beta_xj, beta_xij, beta_yj, mu_jx, mu_ijx, mu_jy, mu_ijy, Qxx, Qxy, Qyy, pi_nux, pi_nuy, const, sign
        cdef double h31000r, h40000r, h20110r, h11200r, h20020r, h20200r, h00310r, h00400r, h31000i, h40000i, h20110i, h11200i, h20020i, h20200i, h00310i, h00400i
        cdef list ele_list, sext_index
        cdef np.ndarray[dtype=np.float64_t] s, qxx, qxy, qyy, f21000, f30000, f10110, f10020, f10200, f31000, f40000, f20110, f11200, f20020, f20200, f00310, f00400
        
        ele_list = []
        sext_index = []
        current_ind = 0
        for ele in self.elements:
            if isinstance(ele, Sextupole):
                ele_list += ele.slice(ele.n_slices)
                for i in range(ele.n_slices):
                    sext_index.append(current_ind)
                    current_ind += 1
            if isinstance(ele, Octupole):
                raise Exception('Unfinished, s_dependent_nonlinear_terms with Octupole.')
            else:
                ele_list.append(ele)
                current_ind += 1
        pi_nux = self.elements[-1].psix / 2
        pi_nuy = self.elements[-1].psiy / 2
        periodic_psix = self.elements[-1].psix
        periodic_psiy = self.elements[-1].psiy
        qxx = np.zeros(2*(len(sext_index) + 1))
        qxy = np.zeros(2*(len(sext_index) + 1))
        qyy = np.zeros(2*(len(sext_index) + 1))
        f21000 = np.zeros(2*(len(sext_index) + 1))
        f30000 = np.zeros(2*(len(sext_index) + 1))
        f10110 = np.zeros(2*(len(sext_index) + 1))
        f10020 = np.zeros(2*(len(sext_index) + 1))
        f10200 = np.zeros(2*(len(sext_index) + 1))
        f31000 = np.zeros(2*(len(sext_index) + 1))
        f40000 = np.zeros(2*(len(sext_index) + 1))
        f20110 = np.zeros(2*(len(sext_index) + 1))
        f11200 = np.zeros(2*(len(sext_index) + 1))
        f20020 = np.zeros(2*(len(sext_index) + 1))
        f20200 = np.zeros(2*(len(sext_index) + 1))
        f00310 = np.zeros(2*(len(sext_index) + 1))
        f00400 = np.zeros(2*(len(sext_index) + 1))
        s = np.zeros(2*(len(sext_index) + 1))
        current_ind = 0
        for k in sext_index:    # 起点在直线段变化时，四阶项和ADTS项只关心相对相移，三阶项角度变化，绝对值不变，所以只计算六极铁处就够了
            s[current_ind*2 + 1] = ele_list[k].s
            s[current_ind*2 + 2] = ele_list[k].s
            Qxx = Qxy = Qyy = 0
            h21000r = h21000i = h30000r = h30000i = h10110r = h10110i = h10020r = h10020i = h10200r = h10200i = 0
            h31000r = h40000r = h20110r = h11200r = h20020r = h20200r = h00310r = h00400r = h31000i = h40000i = h20110i = h11200i = h20020i = h20200i = h00310i = h00400i = 0
            psix_list = []
            psiy_list = []
            for i in range(len(ele_list) - 1):
                if i < k:
                    psix_list.append((ele_list[i].psix + ele_list[i + 1].psix) / 2 - ele_list[k].psix + periodic_psix)
                    psiy_list.append((ele_list[i].psiy + ele_list[i + 1].psiy) / 2 - ele_list[k].psiy + periodic_psiy)
                else:
                    psix_list.append((ele_list[i].psix + ele_list[i + 1].psix) / 2 - ele_list[k].psix)
                    psiy_list.append((ele_list[i].psiy + ele_list[i + 1].psiy) / 2 - ele_list[k].psiy)
            for i in sext_index:
                b3l_i = ele_list[i].k2 * ele_list[i].length / 2  # k2 = 2 * b3, k1 = b2
                if b3l_i != 0:
                    beta_xi = (ele_list[i].betax + ele_list[i + 1].betax) / 2
                    beta_yi = (ele_list[i].betay + ele_list[i + 1].betay) / 2
                    mu_ix = psix_list[i]
                    mu_iy = psiy_list[i]
                    h21000r += - b3l_i * beta_xi ** 1.5 * cos(mu_ix) / 8
                    h30000r += - b3l_i * beta_xi ** 1.5 * cos(3 * mu_ix) / 24
                    h10110r += b3l_i * beta_xi ** 0.5 * beta_yi * cos(mu_ix) / 4
                    h10020r += b3l_i * beta_xi ** 0.5 * beta_yi * cos(mu_ix - 2 * mu_iy) / 8
                    h10200r += b3l_i * beta_xi ** 0.5 * beta_yi * cos(mu_ix + 2 * mu_iy) / 8
                    
                    h21000i += - b3l_i * beta_xi ** 1.5 * sin(mu_ix) / 8
                    h30000i += - b3l_i * beta_xi ** 1.5 * sin(3 * mu_ix) / 24
                    h10110i += b3l_i * beta_xi ** 0.5 * beta_yi * sin(mu_ix) / 4
                    h10020i += b3l_i * beta_xi ** 0.5 * beta_yi * sin(mu_ix - 2 * mu_iy) / 8
                    h10200i += b3l_i * beta_xi ** 0.5 * beta_yi * sin(mu_ix + 2 * mu_iy) / 8
                    for j in sext_index:
                        b3l_j = ele_list[j].k2 * ele_list[j].length / 2
                        b3l = b3l_j * b3l_i
                        if b3l != 0:
                            beta_xj = (ele_list[j].betax + ele_list[j + 1].betax) / 2
                            beta_yj = (ele_list[j].betay + ele_list[j + 1].betay) / 2
                            mu_jx = psix_list[j]
                            mu_ijx = fabs(mu_ix - mu_jx)
                            mu_jy = psiy_list[j]
                            mu_ijy = fabs(mu_iy - mu_jy)
                            beta_xij = beta_xj * beta_xi
                            mu_ij_x2y = mu_ijx + 2 * mu_ijy
                            mu_ij_x_2y = mu_ijx - 2 * mu_ijy
                            Qxx += b3l / (-16 * pi) * pow(beta_xi * beta_xj, 1.5) * (
                                    3 * cos(mu_ijx - pi_nux) / sin(pi_nux)
                                    + cos(3 * mu_ijx - 3 * pi_nux) / sin(3 * pi_nux))
                            Qxy += b3l / (8 * pi) * pow(beta_xij, 0.5) * beta_yj * (
                                    2 * beta_xi * cos(mu_ijx - pi_nux) / sin(pi_nux)
                                    - beta_yi * cos(mu_ijx + 2 * mu_ijy - pi_nux - 2 * pi_nuy) / sin(pi_nux + 2 * pi_nuy)
                                    + beta_yi * cos(mu_ijx - 2 * mu_ijy - pi_nux + 2 * pi_nuy) / sin(pi_nux - 2 * pi_nuy))
                            Qyy += b3l / (-16 * pi) * pow(beta_xij, 0.5) * beta_yj * beta_yi * (
                                    4 * cos(mu_ijx - pi_nux) / sin(pi_nux)
                                    + cos(mu_ijx + 2 * mu_ijy - pi_nux - 2 * pi_nuy) / sin(pi_nux + 2 * pi_nuy)
                                    + cos(mu_ijx - 2 * mu_ijy - pi_nux + 2 * pi_nuy) / sin(pi_nux - 2 * pi_nuy))
                            if (mu_ix - mu_jx > 0):
                                sign = -1
                            elif (mu_ix - mu_jx < 0):
                                sign = 1
                            else:
                                continue
                            const = sign * b3l
                            h31000r += const * beta_xij ** 1.5 * (-sin(3 * mu_ix - mu_jx)) / 32
                            h40000r += const * beta_xij ** 1.5 * (-sin(3 * mu_ix + mu_jx)) / 64
                            h20110r += const * beta_xij ** 0.5 * beta_yi * (
                                    beta_xj * ((-sin(3 * mu_jx - mu_ix)) - (-sin(mu_ix + mu_jx))) +
                                    2 * beta_yj * (-sin(mu_ix + mu_jx + 2 * mu_iy - 2 * mu_jy))) / 32
                            h11200r += const * beta_xij ** 0.5 * beta_yi * (
                                    beta_xj * ((-sin(-mu_ix + mu_jx + 2 * mu_iy)) - (-sin(mu_ix - mu_jx + 2 * mu_iy))) +
                                    2 * beta_yj * ((-sin(mu_ix - mu_jx + 2 * mu_iy)) + (-sin(- mu_ix + mu_jx + 2 * mu_iy)))) / 32
                            h20020r += const * beta_xij ** 0.5 * beta_yi * (
                                    beta_xj * (-sin(-mu_ix + 3 * mu_jx - 2 * mu_iy)) -
                                    (beta_xj + 4 * beta_yj) * (-sin(mu_ix + mu_jx - 2 * mu_iy))) / 64
                            h20200r += const * beta_xij ** 0.5 * beta_yi * (
                                    beta_xj * (-sin(-mu_ix + 3 * mu_jx + 2 * mu_iy))
                                    - (beta_xj - 4 * beta_yj) * ((-sin(mu_ix + mu_jx + 2 * mu_iy)))) / 64
                            h00310r += const * beta_xij ** 0.5 * beta_yi * beta_yj * (
                                    (-sin(mu_ix - mu_jx + 2 * mu_iy)) -
                                    (-sin(-mu_ix + mu_jx + 2 * mu_iy))) / 32
                            h00400r += const * beta_xij ** 0.5 * beta_yi * beta_yj * (-sin(mu_ix - mu_jx + 2 * mu_iy + 2 * mu_jy)) / 64
    
                            h31000i += const * beta_xij ** 1.5 * (cos(3 * mu_ix - mu_jx)) / 32
                            h40000i += const * beta_xij ** 1.5 * (cos(3 * mu_ix + mu_jx)) / 64
                            h20110i += const * beta_xij ** 0.5 * beta_yi * (
                                    beta_xj * ((cos(3 * mu_jx - mu_ix)) - (cos(mu_ix + mu_jx))) +
                                    2 * beta_yj * (cos(mu_ix + mu_jx + 2 * mu_iy - 2 * mu_jy))) / 32
                            h11200i += const * beta_xij ** 0.5 * beta_yi * (
                                    beta_xj * ((cos(-mu_ix + mu_jx + 2 * mu_iy)) - (cos(mu_ix - mu_jx + 2 * mu_iy))) +
                                    2 * beta_yj * ((cos(mu_ix - mu_jx + 2 * mu_iy)) + (cos(- mu_ix + mu_jx + 2 * mu_iy)))) / 32
                            h20020i += const * beta_xij ** 0.5 * beta_yi * (
                                    beta_xj * (cos(-mu_ix + 3 * mu_jx - 2 * mu_iy)) -
                                    (beta_xj + 4 * beta_yj) * (cos(mu_ix + mu_jx - 2 * mu_iy))) / 64
                            h20200i += const * beta_xij ** 0.5 * beta_yi * (
                                    beta_xj * (cos(-mu_ix + 3 * mu_jx + 2 * mu_iy))
                                    - (beta_xj - 4 * beta_yj) * ((cos(mu_ix + mu_jx + 2 * mu_iy)))) / 64
                            h00310i += const * beta_xij ** 0.5 * beta_yi * beta_yj * (
                                    (cos(mu_ix - mu_jx + 2 * mu_iy)) -
                                    (cos(-mu_ix + mu_jx + 2 * mu_iy))) / 32
                            h00400i += const * beta_xij ** 0.5 * beta_yi * beta_yj * (cos(mu_ix - mu_jx + 2 * mu_iy + 2 * mu_jy)) / 64
            f21000[current_ind*2] = sqrt(h21000r ** 2 + h21000i ** 2)
            f30000[current_ind*2] = sqrt(h30000r ** 2 + h30000i ** 2)
            f10110[current_ind*2] = sqrt(h10110r ** 2 + h10110i ** 2)
            f10200[current_ind*2] = sqrt(h10200r ** 2 + h10200i ** 2)
            f10020[current_ind*2] = sqrt(h10020r ** 2 + h10020i ** 2)
            qxx[current_ind*2] = Qxx
            qxy[current_ind*2] = Qxy
            qyy[current_ind*2] = Qyy
            f31000[current_ind*2] = sqrt(h31000r ** 2 + h31000i ** 2)
            f40000[current_ind*2] = sqrt(h40000r ** 2 + h40000i ** 2)
            f00310[current_ind*2] = sqrt(h00310r ** 2 + h00310i ** 2)
            f20020[current_ind*2] = sqrt(h20020r ** 2 + h20020i ** 2)
            f20110[current_ind*2] = sqrt(h20110r ** 2 + h20110i ** 2)
            f00400[current_ind*2] = sqrt(h00400r ** 2 + h00400i ** 2)
            f20200[current_ind*2] = sqrt(h20200r ** 2 + h20200i ** 2)
            f11200[current_ind*2] = sqrt(h11200r ** 2 + h11200i ** 2)
            f21000[current_ind*2 + 1] = f21000[current_ind*2]
            f30000[current_ind*2 + 1] = f30000[current_ind*2]
            f10110[current_ind*2 + 1] = f10110[current_ind*2]
            f10200[current_ind*2 + 1] = f10200[current_ind*2]
            f10020[current_ind*2 + 1] = f10020[current_ind*2]
            qxx[current_ind*2 + 1] = Qxx
            qxy[current_ind*2 + 1] = Qxy
            qyy[current_ind*2 + 1] = Qyy
            f31000[current_ind*2 + 1] = f31000[current_ind*2]
            f40000[current_ind*2 + 1] = f40000[current_ind*2]
            f00310[current_ind*2 + 1] = f00310[current_ind*2]
            f20020[current_ind*2 + 1] = f20020[current_ind*2]
            f20110[current_ind*2 + 1] = f20110[current_ind*2]
            f00400[current_ind*2 + 1] = f00400[current_ind*2]
            f20200[current_ind*2 + 1] = f20200[current_ind*2]
            f11200[current_ind*2 + 1] = f11200[current_ind*2]
            current_ind += 1
        s[-1] = ele_list[-1].s
        nonlinear = {'s': s, 'h21000': f21000, 'h30000': f30000, 'h10110': f10110, 'h10020': f10020,
                     'h10200': f10200, 'dQxx': qxx, 'dQxy': qxy, 'dQyy': qyy,
                     'h31000': f31000, 'h40000': f40000, 'h20110': f20110, 'h11200': f11200,
                     'h20020': f20020, 'h20200': f20200, 'h00310': f00310, 'h00400': f00400}
        for k in nonlinear:
            if k != 's':
                nonlinear[k][-1] = nonlinear[k][0]
                nonlinear[k][-2] = nonlinear[k][0]
        return nonlinear

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def driving_terms_plot_data(self):
        """Similar to DrivingTerms.fluctuation(). But the arrays in the result of driving_terms_plot_data() have the
        same length in order to plot figure, and the length is double in order to plot steps.
        
        Return:
            {'s': , 'h21000': , 'h30000': , 'h10110': , 'h10020': , 'h10200': , 
             'h20001': , 'h00201': , 'h10002': , 
             'h31000': , 'h40000': , 'h20110': , 'h11200': ,
             'h20020': , 'h20200': , 'h00310': , 'h00400':} each value is a np.ndarray.

        references:
        1. The Sextupole Scheme for the SLS: An Analytic Approach, SLS Note 09/97, Johan Bengtsson
        2. Explicit formulas for 2nd-order driving terms due to sextupoles and chromatic effects of quadrupoles, Chun-xi Wang
        3. Perspectives for future light source lattices incorporating yet uncommon magnets, S. C. Leemann and A. Streun"""

        cdef int geo_3rd_idx, geo_4th_idx, chr_3rd_idx, num_ele
        cdef complex h21000, h30000, h10110, h10020, h10200, h20001, h00201, h10002, jj
        cdef complex h12000, h01110, h01200, h01010, h12000j, h01110j, h01200j, h01010j
        cdef complex h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400, h22000, h11110, h00220
        cdef np.ndarray[dtype=np.complex128_t] f22000, f11110, f00220, f21000, f30000, f10110, f10020, f10200, f20001, f00201, f10002, f11001, f00111, f31000, f40000, f20110, f11200, f20020, f20200, f00310, f00400
        cdef np.ndarray[dtype=np.float64_t] s

        num_ele = len(self.elements)
        f21000 = np.zeros(num_ele, dtype='complex_')
        f30000 = np.zeros(num_ele, dtype='complex_')
        f10110 = np.zeros(num_ele, dtype='complex_')
        f10020 = np.zeros(num_ele, dtype='complex_')
        f10200 = np.zeros(num_ele, dtype='complex_')
        f20001 = np.zeros(num_ele, dtype='complex_')
        f00201 = np.zeros(num_ele, dtype='complex_')
        f10002 = np.zeros(num_ele, dtype='complex_')
        f22000 = np.zeros(num_ele, dtype='complex_')
        f11110 = np.zeros(num_ele, dtype='complex_')
        f00220 = np.zeros(num_ele, dtype='complex_')
        f31000 = np.zeros(num_ele, dtype='complex_')
        f40000 = np.zeros(num_ele, dtype='complex_')
        f20110 = np.zeros(num_ele, dtype='complex_')
        f11200 = np.zeros(num_ele, dtype='complex_')
        f20020 = np.zeros(num_ele, dtype='complex_')
        f20200 = np.zeros(num_ele, dtype='complex_')
        f00310 = np.zeros(num_ele, dtype='complex_')
        f00400 = np.zeros(num_ele, dtype='complex_')
        s = np.zeros(num_ele)
        h21000 = h30000 = h10110 = h10020 = h10200 = h20001 = h00201 = h10002 = 0
        h22000 = h11110 = h00220 = h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
        idx = 0
        jj = complex(0, 1)
        for ele in self.elements:  # TODO: quad-sext
            if isinstance(ele, Sextupole):     #   0        1      2       3       4       5       6       7  
                rdts = ele.driving_terms()   # h21000, h30000, h10110, h10020, h10200, h20001, h00201, h10002
                h12000j = rdts[0].conjugate()
                h01110j = rdts[2].conjugate()
                h01200j = rdts[3].conjugate()
                h01020j = rdts[4].conjugate()
                h12000 = h21000.conjugate()
                h01110 = h10110.conjugate()
                h01200 = h10020.conjugate()
                h01020 = h10200.conjugate()         
                h22000 += jj * ((h21000 * h12000j - h12000 * rdts[0]) * 3
                                +(h30000 * rdts[1].conjugate() - h30000.conjugate() * rdts[1]) * 9) + rdts[8]
                h11110 += jj * ((h21000 * h01110j - h01110 * rdts[0]) * 2
                                -(h12000 * rdts[2] - h10110 * h12000j) * 2
                                -(h10020 * h01200j - h01200 * rdts[3]) * 4
                                +(h10200 * h01020j - h01020 * rdts[4]) * 4) + rdts[9]
                h00220 += jj * ((h10020 * h01200j - h01200 * rdts[3])
                                +(h10200 * h01020j - h01020 * rdts[4])
                                +(h10110 * h01110j - h01110 * rdts[2])) + rdts[10]
                h31000 += jj * 6 * (h30000 * h12000j - h12000 * rdts[1]) + rdts[11]
                h40000 += jj * 3 * (h30000 * rdts[0] - h21000 * rdts[1]) + rdts[12]
                h20110 += jj * ((h30000 * h01110j - h01110 * rdts[1]) * 3 
                               -(h21000 * rdts[2] - h10110 * rdts[0])
                                +(h10200 * rdts[3] - h10020 * rdts[4]) * 4) + rdts[13]
                h11200 += jj * ((h10200 * h12000j - h12000 * rdts[4]) * 2
                                +(h21000 * h01200j - h01200 * rdts[0]) * 2
                                +(h10200 * h01110j - h01110 * rdts[4]) * 2
                                +(h10110 * h01200j - h01200 * rdts[2]) * (-2)) + rdts[14]
                h20020 += jj * (-(h21000 * rdts[3] - h10020 * rdts[0])
                                +(h30000 * h01020j - h01020 * rdts[1]) * 3
                                +(h10110 * rdts[3] - h10020 * rdts[2]) * 2) + rdts[15]
                h20200 += jj * ((h30000 * h01200j - h01200 * rdts[1]) * 3
                                +(h10200 * rdts[0] - h21000 * rdts[4])
                                +(h10110 * rdts[4] - h10200 * rdts[2]) * (-2)) + rdts[16]
                h00310 += jj * ((h10200 * h01110j - h01110 * rdts[4])
                                +(h10110 * h01200j - h01200 * rdts[2])) + rdts[17]
                h00400 += jj * (h10200 * h01200j - h01200 * rdts[4]) + rdts[18]
                
                h21000 = h21000 + rdts[0]
                h30000 = h30000 + rdts[1]
                h10110 = h10110 + rdts[2]
                h10020 = h10020 + rdts[3]
                h10200 = h10200 + rdts[4]
                h20001 += rdts[5]
                h00201 += rdts[6]
                h10002 += rdts[7]

                s[idx + 1] = ele.s
                s[idx + 2] = s[idx + 1]
                f22000[idx + 2] = h22000
                f11110[idx + 2] = h11110
                f00220[idx + 2] = h00220
                f31000[idx + 2] = h31000
                f40000[idx + 2] = h40000
                f20110[idx + 2] = h20110
                f11200[idx + 2] = h11200
                f20020[idx + 2] = h20020
                f20200[idx + 2] = h20200
                f00310[idx + 2] = h00310
                f00400[idx + 2] = h00400
                f21000[idx + 2] = h21000
                f30000[idx + 2] = h30000
                f10110[idx + 2] = h10110
                f10020[idx + 2] = h10020
                f10200[idx + 2] = h10200
                f20001[idx + 2] = h20001
                f00201[idx + 2] = h00201
                f10002[idx + 2] = h10002

                f22000[idx + 3] = f22000[idx + 2]
                f11110[idx + 3] = f11110[idx + 2]
                f00220[idx + 3] = f00220[idx + 2]
                f31000[idx + 3] = f31000[idx + 2]
                f40000[idx + 3] = f40000[idx + 2]
                f20110[idx + 3] = f20110[idx + 2]
                f11200[idx + 3] = f11200[idx + 2]
                f20020[idx + 3] = f20020[idx + 2]
                f20200[idx + 3] = f20200[idx + 2]
                f00310[idx + 3] = f00310[idx + 2]
                f00400[idx + 3] = f00400[idx + 2]
                f21000[idx + 3] = f21000[idx + 2]
                f30000[idx + 3] = f30000[idx + 2]
                f10110[idx + 3] = f10110[idx + 2]
                f10020[idx + 3] = f10020[idx + 2]
                f10200[idx + 3] = f10200[idx + 2]
                f20001[idx + 3] = f20001[idx + 2]
                f00201[idx + 3] = f00201[idx + 2]
                f10002[idx + 3] = f10002[idx + 2]
                idx += 2
            elif isinstance(ele, Octupole):
                rdts = ele.driving_terms()  # h22000, h11110, h00220, h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400
                h22000 += rdts[0]
                h11110 += rdts[1]
                h00220 += rdts[2]
                h31000 += rdts[3]
                h40000 += rdts[4]
                h20110 += rdts[5]
                h11200 += rdts[6]
                h20020 += rdts[7]
                h20200 += rdts[8]
                h00310 += rdts[9]
                h00400 += rdts[10]

                s[idx + 1] = ele.s
                s[idx + 2] = s[idx + 1]
                f22000[idx + 2] = h22000
                f11110[idx + 2] = h11110
                f00220[idx + 2] = h00220
                f31000[idx + 2] = h31000
                f40000[idx + 2] = h40000
                f20110[idx + 2] = h20110
                f11200[idx + 2] = h11200
                f20020[idx + 2] = h20020
                f20200[idx + 2] = h20200
                f00310[idx + 2] = h00310
                f00400[idx + 2] = h00400
                f21000[idx + 2] = h21000
                f30000[idx + 2] = h30000
                f10110[idx + 2] = h10110
                f10020[idx + 2] = h10020
                f10200[idx + 2] = h10200
                f20001[idx + 2] = h20001
                f00201[idx + 2] = h00201
                f10002[idx + 2] = h10002

                f22000[idx + 3] = f22000[idx + 2]
                f11110[idx + 3] = f11110[idx + 2]
                f00220[idx + 3] = f00220[idx + 2]
                f31000[idx + 3] = f31000[idx + 2]
                f40000[idx + 3] = f40000[idx + 2]
                f20110[idx + 3] = f20110[idx + 2]
                f11200[idx + 3] = f11200[idx + 2]
                f20020[idx + 3] = f20020[idx + 2]
                f20200[idx + 3] = f20200[idx + 2]
                f00310[idx + 3] = f00310[idx + 2]
                f00400[idx + 3] = f00400[idx + 2]
                f21000[idx + 3] = f21000[idx + 2]
                f30000[idx + 3] = f30000[idx + 2]
                f10110[idx + 3] = f10110[idx + 2]
                f10020[idx + 3] = f10020[idx + 2]
                f10200[idx + 3] = f10200[idx + 2]
                f20001[idx + 3] = f20001[idx + 2]
                f00201[idx + 3] = f00201[idx + 2]
                f10002[idx + 3] = f10002[idx + 2]
                idx += 2
            elif ele.k1:
                rdts = ele.driving_terms()  # h20001, h00201, h10002
                h20001 += rdts[0]
                h00201 += rdts[1]
                h10002 += rdts[2]
                s[idx + 1] = ele.s
                s[idx + 2] = s[idx + 1]
                f22000[idx + 2] = h22000
                f11110[idx + 2] = h11110
                f00220[idx + 2] = h00220
                f31000[idx + 2] = h31000
                f40000[idx + 2] = h40000
                f20110[idx + 2] = h20110
                f11200[idx + 2] = h11200
                f20020[idx + 2] = h20020
                f20200[idx + 2] = h20200
                f00310[idx + 2] = h00310
                f00400[idx + 2] = h00400
                f21000[idx + 2] = h21000
                f30000[idx + 2] = h30000
                f10110[idx + 2] = h10110
                f10020[idx + 2] = h10020
                f10200[idx + 2] = h10200
                f20001[idx + 2] = h20001
                f00201[idx + 2] = h00201
                f10002[idx + 2] = h10002

                f22000[idx + 3] = f22000[idx + 2]
                f11110[idx + 3] = f11110[idx + 2]
                f00220[idx + 3] = f00220[idx + 2]
                f31000[idx + 3] = f31000[idx + 2]
                f40000[idx + 3] = f40000[idx + 2]
                f20110[idx + 3] = f20110[idx + 2]
                f11200[idx + 3] = f11200[idx + 2]
                f20020[idx + 3] = f20020[idx + 2]
                f20200[idx + 3] = f20200[idx + 2]
                f00310[idx + 3] = f00310[idx + 2]
                f00400[idx + 3] = f00400[idx + 2]
                f21000[idx + 3] = f21000[idx + 2]
                f30000[idx + 3] = f30000[idx + 2]
                f10110[idx + 3] = f10110[idx + 2]
                f10020[idx + 3] = f10020[idx + 2]
                f10200[idx + 3] = f10200[idx + 2]
                f20001[idx + 3] = f20001[idx + 2]
                f00201[idx + 3] = f00201[idx + 2]
                f10002[idx + 3] = f10002[idx + 2]
                idx += 2
        s[idx + 1] = ele.s
        idx += 2
        RDTs_along_ring = {'s': s[:idx], 'h21000': f21000[:idx], 'h30000': f30000[:idx], 'h10110': f10110[:idx], 'h10020': f10020[:idx],
                     'h10200': f10200[:idx], 'h20001': f20001[:idx], 'h00201': f00201[:idx], 'h10002': f10002[:idx],
                     'h31000': f31000[:idx], 'h40000': f40000[:idx], 'h20110': f20110[:idx], 'h11200': f11200[:idx],
                     'h20020': f20020[:idx], 'h20200': f20200[:idx], 'h00310': f00310[:idx], 'h00400': f00400[:idx],
                     'h22000': f22000[:idx], 'h11110': f11110[:idx], 'h00220': f00220[:idx]
                    }
        return RDTs_along_ring

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def driving_terms_plot_data_old(self):
        """Similar to DrivingTerms.fluctuation(). But the arrays in the result of driving_terms_plot_data() have the
        same length in order to plot figure, and the length is double in order to plot steps.
        
        Return:
            {'s': , 'h21000': , 'h30000': , 'h10110': , 'h10020': , 'h10200': , 
             'h20001': , 'h00201': , 'h10002': , 'h11001': , 'h00111': ,
             'h31000': , 'h40000': , 'h20110': , 'h11200': ,
             'h20020': , 'h20200': , 'h00310': , 'h00400': ,
             'h22000': , 'h11110': , 'h00220': } each value is a np.ndarray.

        references:
        1. The Sextupole Scheme for the SLS: An Analytic Approach, SLS Note 09/97, Johan Bengtsson
        2. Explicit formulas for 2nd-order driving terms due to sextupoles and chromatic effects of quadrupoles, Chun-xi Wang
        3. Perspectives for future light source lattices incorporating yet uncommon magnets, S. C. Leemann and A. Streun"""
        
        cdef double b3l_i, b4l, b2l, beta_xi, eta_x, mu_ix, mu_iy, beta_yi
        cdef int current_ind, sext_num, i, j, oct_num, quad_num, total_ele_count
        cdef double hts21000r, hts21000i, hts30000r, hts30000i,hts10110r, hts10110i, hts10200r, hts10200i, hts10020r, hts10020i
        cdef double hts20001r, hts20001i, hts00201r, hts00201i, hts10002r, hts10002i, hts11001, hts00111, hts22000, hts00220, hts11110
        cdef double hts31000r, hts40000r, hts20110r, hts11200r, hts20020r, hts20200r, hts00310r, hts00400r, hts31000i, hts40000i, hts20110i, hts11200i, hts20020i, hts20200i, hts00310i, hts00400i
        cdef double b3l, beta_xj, beta_xij, beta_yj, mu_jx, mu_jy, mu_ijx, mu_ijy, beta_yij, beta_xiyj, beta_xjyi
        cdef list ele_list, quad_index, sext_index, oct_index, nonlinear_index
        cdef np.ndarray[dtype=np.float64_t] s, f22000, f11110, f00220, f21000, f30000, f10110, f10020, f10200, f20001, f00201, f10002, f11001, f00111, f31000, f40000, f20110, f11200, f20020, f20200, f00310, f00400
        cdef np.ndarray[ndim=2, dtype=np.int_t] nonlinear_num

        ele_list = []
        quad_index = []
        sext_index = []
        oct_index = []
        quad_num = sext_num = oct_num = 0
        dQxx = dQxy = dQyy = 0
        nonlinear_index = []
        nonlinear_num = np.array([[0, 0, 0], [0, 0, 0]])
        current_ind = 0
        for ele in self.elements:
            if isinstance(ele, Sextupole):
                ele_list += ele.slice(ele.n_slices)
                for i in range(ele.n_slices):
                    sext_index.append(current_ind)
                    nonlinear_index.append(current_ind)
                    current_ind += 1
                    sext_num += 1
                    nonlinear_num = np.vstack((nonlinear_num, (0, 1, 0)))  # 1 means the current magnet is Sextupole
            elif isinstance(ele, Octupole):
                ele_list += ele.slice(ele.n_slices)
                for i in range(ele.n_slices):
                    oct_index.append(current_ind)
                    nonlinear_index.append(current_ind)
                    current_ind += 1
                    oct_num += 1
                    nonlinear_num = np.vstack((nonlinear_num, (0, 0, 1)))
            elif ele.k1:
                ele_list += ele.slice(ele.n_slices)
                for i in range(ele.n_slices):
                    quad_index.append(current_ind)
                    nonlinear_index.append(current_ind)
                    current_ind += 1
                    quad_num += 1
                    nonlinear_num = np.vstack((nonlinear_num, (1, 0, 0)))
            else:
                ele_list.append(ele)
                current_ind += 1
        total_ele_count = current_ind
        nonlinear_num = np.delete(nonlinear_num, [0, 1], 0)
        # 1 dim array, [i, j] store the 3rd-RDTs produced by Quad_i
        cdef double[:] hqs11001  = np.zeros(quad_num)
        cdef double[:] hqs00111  = np.zeros(quad_num)
        cdef double[:] hqs20001r = np.zeros(quad_num)
        cdef double[:] hqs00201r = np.zeros(quad_num)
        cdef double[:] hqs10002r = np.zeros(quad_num)
        cdef double[:] hqs20001i = np.zeros(quad_num)
        cdef double[:] hqs00201i = np.zeros(quad_num)
        cdef double[:] hqs10002i = np.zeros(quad_num)
        # 1 dim array, [i, j] store the 3rd-RDTs produced by Sext_i
        cdef double[:] hs11001 = np.zeros(sext_num)
        cdef double[:] hs00111 = np.zeros(sext_num)
        cdef double[:] hs20001r = np.zeros(sext_num)
        cdef double[:] hs00201r = np.zeros(sext_num)
        cdef double[:] hs10002r = np.zeros(sext_num)
        cdef double[:] hs20001i = np.zeros(sext_num)
        cdef double[:] hs00201i = np.zeros(sext_num)
        cdef double[:] hs10002i = np.zeros(sext_num)
        cdef double[:] hs21000r = np.zeros(sext_num)
        cdef double[:] hs30000r = np.zeros(sext_num)
        cdef double[:] hs10110r = np.zeros(sext_num)
        cdef double[:] hs10020r = np.zeros(sext_num)
        cdef double[:] hs10200r = np.zeros(sext_num)
        cdef double[:] hs21000i = np.zeros(sext_num)
        cdef double[:] hs30000i = np.zeros(sext_num)
        cdef double[:] hs10110i = np.zeros(sext_num)
        cdef double[:] hs10020i = np.zeros(sext_num)
        cdef double[:] hs10200i = np.zeros(sext_num)
        # 1 dim array, [i, j] store the 4th-RDTs produced by Oct_i
        cdef double[:] hos00220  = np.zeros(oct_num)
        cdef double[:] hos11110  = np.zeros(oct_num)
        cdef double[:] hos22000  = np.zeros(oct_num)
        cdef double[:] hos31000r = np.zeros(oct_num)
        cdef double[:] hos40000r = np.zeros(oct_num)
        cdef double[:] hos20110r = np.zeros(oct_num)
        cdef double[:] hos11200r = np.zeros(oct_num)
        cdef double[:] hos20020r = np.zeros(oct_num)
        cdef double[:] hos20200r = np.zeros(oct_num)
        cdef double[:] hos00310r = np.zeros(oct_num)
        cdef double[:] hos00400r = np.zeros(oct_num)
        cdef double[:] hos31000i = np.zeros(oct_num)
        cdef double[:] hos40000i = np.zeros(oct_num)
        cdef double[:] hos20110i = np.zeros(oct_num)
        cdef double[:] hos11200i = np.zeros(oct_num)
        cdef double[:] hos20020i = np.zeros(oct_num)
        cdef double[:] hos20200i = np.zeros(oct_num)
        cdef double[:] hos00310i = np.zeros(oct_num)
        cdef double[:] hos00400i = np.zeros(oct_num)
        # 2 dim array, [i, j] store the 4th-RDTs produced by Sext_i, Sext_j
        cdef double[:, :] hs00220  = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs11110  = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs22000  = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs31000r = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs40000r = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs20110r = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs11200r = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs20020r = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs20200r = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs00310r = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs00400r = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs31000i = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs40000i = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs20110i = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs11200i = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs20020i = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs20200i = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs00310i = np.zeros((sext_num, sext_num))
        cdef double[:, :] hs00400i = np.zeros((sext_num, sext_num))
        # 1 dim array, store the RDTs of s_i
        f21000 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f30000 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f10110 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f10020 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f10200 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f20001 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f00201 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f10002 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))

        f11001 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f00111 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f22000 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f11110 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f00220 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))

        f31000 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f40000 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f20110 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f11200 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f20020 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f20200 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f00310 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        f00400 = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        s = np.zeros(2*(quad_num + sext_num + 1 + oct_num))
        for i in range(quad_num):
            b2l = ele_list[quad_index[i]].k1 * ele_list[quad_index[i]].length
            beta_xi = (ele_list[quad_index[i]].betax + ele_list[quad_index[i] + 1].betax) / 2
            eta_x = (ele_list[quad_index[i]].etax + ele_list[quad_index[i] + 1].etax) / 2
            beta_yi = (ele_list[quad_index[i]].betay + ele_list[quad_index[i] + 1].betay) / 2
            mu_ix = (ele_list[quad_index[i]].psix + ele_list[quad_index[i] + 1].psix) / 2
            mu_iy = (ele_list[quad_index[i]].psiy + ele_list[quad_index[i] + 1].psiy) / 2
            hqs11001[i]  = b2l * beta_xi / 4
            hqs00111[i]  = -b2l * beta_yi / 4
            hqs20001r[i] = b2l * beta_xi / 8 * cos(2 * mu_ix)
            hqs00201r[i] = -b2l * beta_yi / 8 * cos(2 * mu_iy)
            hqs10002r[i] = b2l * beta_xi ** 0.5 * eta_x / 2 * cos(mu_ix)
            hqs20001i[i] =  b2l * beta_xi / 8 * sin(2 * mu_ix)
            hqs00201i[i] = -b2l * beta_yi / 8 * sin(2 * mu_iy)
            hqs10002i[i] = b2l * beta_xi ** 0.5 * eta_x / 2 * sin(mu_ix)
        for i in range(sext_num):
            b3l_i = ele_list[sext_index[i]].k2 * ele_list[sext_index[i]].length / 2  # k2 = 2 * b3, k1 = b2
            beta_xi = (ele_list[sext_index[i]].betax + ele_list[sext_index[i] + 1].betax) / 2
            eta_x = (ele_list[sext_index[i]].etax + ele_list[sext_index[i] + 1].etax) / 2
            beta_yi = (ele_list[sext_index[i]].betay + ele_list[sext_index[i] + 1].betay) / 2
            mu_ix = (ele_list[sext_index[i]].psix + ele_list[sext_index[i] + 1].psix) / 2
            mu_iy = (ele_list[sext_index[i]].psiy + ele_list[sext_index[i] + 1].psiy) / 2

            hs11001[i] = -b3l_i * beta_xi * eta_x / 2
            hs00111[i] = b3l_i * beta_yi * eta_x / 2
            hs20001r[i] = -b3l_i * beta_xi * eta_x / 4 * cos(2 * mu_ix)
            hs00201r[i] =  b3l_i * beta_yi * eta_x / 4 * cos(2 * mu_iy)
            hs10002r[i] = -b3l_i * beta_xi ** 0.5 * eta_x ** 2 / 2 * cos(mu_ix)
            hs20001i[i] = -b3l_i * beta_xi * eta_x / 4 * sin(2 * mu_ix)
            hs00201i[i] =  b3l_i * beta_yi * eta_x / 4 * sin(2 * mu_iy)
            hs10002i[i] = -b3l_i * beta_xi ** 0.5 * eta_x ** 2 / 2 * sin(mu_ix)

            
            hs21000r[i] = - b3l_i * beta_xi ** 1.5 * cos(mu_ix) / 8
            hs30000r[i] = - b3l_i * beta_xi ** 1.5 * cos(3 * mu_ix) / 24
            hs10110r[i] = b3l_i * beta_xi ** 0.5 * beta_yi * cos(mu_ix) / 4
            hs10020r[i] = b3l_i * beta_xi ** 0.5 * beta_yi * cos(mu_ix - 2 * mu_iy) / 8
            hs10200r[i] = b3l_i * beta_xi ** 0.5 * beta_yi * cos(mu_ix + 2 * mu_iy) / 8
                
            hs21000i[i] = - b3l_i * beta_xi ** 1.5 * sin(mu_ix) / 8
            hs30000i[i] = - b3l_i * beta_xi ** 1.5 * sin(3 * mu_ix) / 24
            hs10110i[i] = b3l_i * beta_xi ** 0.5 * beta_yi * sin(mu_ix) / 4
            hs10020i[i] = b3l_i * beta_xi ** 0.5 * beta_yi * sin(mu_ix - 2 * mu_iy) / 8
            hs10200i[i] = b3l_i * beta_xi ** 0.5 * beta_yi * sin(mu_ix + 2 * mu_iy) / 8
            for j in range(i):
                b3l = b3l_i * ele_list[sext_index[j]].k2 * ele_list[sext_index[j]].length / 2
                beta_xj = (ele_list[sext_index[j]].betax + ele_list[sext_index[j] + 1].betax) / 2
                beta_yj = (ele_list[sext_index[j]].betay + ele_list[sext_index[j] + 1].betay) / 2
                mu_jx = (ele_list[sext_index[j]].psix + ele_list[sext_index[j] + 1].psix) / 2
                mu_ijx = mu_ix - mu_jx
                mu_jy = (ele_list[sext_index[j]].psiy + ele_list[sext_index[j] + 1].psiy) / 2
                mu_ijy = mu_iy - mu_jy
                beta_xij = beta_xj * beta_xi
                beta_yij = beta_yi * beta_yj
                beta_xiyj = beta_xi * beta_yj
                beta_xjyi = beta_xj * beta_yi

                hs22000[i, j]  = b3l * beta_xij ** 1.5 * (sin(3 * mu_ijx) + 3 * sin(mu_ijx)) / 32  # 2 / 64
                hs11110[i, j]  = b3l * beta_xij ** 0.5 * (-sin(mu_ijx) * (beta_xjyi + beta_xiyj) + beta_yij * (sin(mu_ijx + 2 * mu_ijy) - sin(mu_ijx - 2 * mu_ijy))) / 8  # 2 / 16
                hs00220[i, j]  = b3l * beta_xij ** 0.5 * beta_yij * (sin(mu_ijx + 2 * mu_ijy) + 4 * sin(mu_ijx) + sin(mu_ijx - 2 * mu_ijy)) / 32  # 2 / 64
                hs31000r[i, j] = b3l * beta_xij ** 1.5 * (sin(3 * mu_ix - mu_jx) - sin(3 * mu_jx - mu_ix)) / 32
                hs40000r[i, j] = b3l * beta_xij ** 1.5 * (sin(3 * mu_ix + mu_jx) - sin(3 * mu_jx + mu_ix)) / 64
                hs20110r[i, j] = b3l * beta_xij ** 0.5 * (-beta_xjyi * (sin(mu_ix - 3 * mu_jx) + sin(mu_ix + mu_jx)) + 
                        beta_xiyj * (sin(mu_jx - 3 * mu_ix) + sin(mu_ix + mu_jx))
                        + 2 * beta_yij * (sin(mu_ix + mu_jx + 2 * mu_ijy) - sin(mu_ix + mu_jx - 2 * mu_ijy))) / 32
                hs11200r[i, j] = b3l * beta_xij ** 0.5 * (-beta_xjyi * (sin(mu_ix - mu_jx - 2 * mu_iy) + sin(mu_ix - mu_jx + 2 * mu_iy))
                        + beta_xiyj * (sin(mu_jx - mu_ix - 2 * mu_jy) + sin(mu_jx - mu_ix + 2 * mu_jy))
                        + 2 * beta_yij * (sin(mu_ix - mu_jx + 2 * mu_iy) - sin(mu_ix - mu_jx - 2 * mu_iy) - sin(mu_jx - mu_ix + 2 * mu_jy) + sin(mu_jx - mu_ix - 2 * mu_jy))) / 32
                hs20020r[i, j] = b3l * beta_xij ** 0.5 * (-beta_xjyi * sin(mu_ix - 3 * mu_jx + 2 * mu_iy) - (beta_xjyi + 4 * beta_yij) * sin(mu_ix + mu_jx - 2 * mu_iy)
                        + beta_xiyj * sin(mu_jx - 3 * mu_ix + 2 * mu_jy) + (beta_xiyj + 4 * beta_yij) * sin(mu_jx + mu_ix - 2 * mu_jy)) / 64
                hs20200r[i, j] = b3l * beta_xij ** 0.5 * (-beta_xjyi * sin(mu_ix - 3 * mu_jx - 2 * mu_iy) - (beta_xjyi - 4 * beta_yij) * sin(mu_ix + mu_jx + 2 * mu_iy)
                        + beta_xiyj * sin(mu_jx - 3 * mu_ix - 2 * mu_jy) + (beta_xiyj - 4 * beta_yij) * sin(mu_jx + mu_ix + 2 * mu_jy)) / 64
                hs00310r[i, j] = b3l * beta_xij ** 0.5 * beta_yij * (sin(mu_ix - mu_jx + 2 * mu_iy) + sin(mu_ix - mu_jx - 2 * mu_iy)
                        - sin(mu_jx - mu_ix + 2 * mu_jy) - sin(mu_jx - mu_ix - 2 * mu_jy)) / 32
                hs00400r[i, j] = b3l * beta_xij ** 0.5 * beta_yij * (sin(mu_ijx + 2 * (mu_iy + mu_jy)) - sin(-mu_ijx + 2 * (mu_iy + mu_jy))) / 64

                hs31000i[i, j] = b3l * beta_xij ** 1.5 * (-cos(3 * mu_ix - mu_jx) + cos(3 * mu_jx - mu_ix)) / 32
                hs40000i[i, j] = b3l * beta_xij ** 1.5 * (-cos(3 * mu_ix + mu_jx) + cos(3 * mu_jx + mu_ix)) / 64
                hs20110i[i, j] = b3l * beta_xij ** 0.5 * (-beta_xjyi * (cos(mu_ix - 3 * mu_jx) - cos(mu_ix + mu_jx))
                        + beta_xiyj * (cos(mu_jx - 3 * mu_ix) - cos(mu_ix + mu_jx))
                        + 2 * beta_yij * (cos(mu_ix + mu_jx - 2 * mu_ijy) - cos(mu_ix + mu_jx + 2 * mu_ijy))) / 32
                hs11200i[i, j] = b3l * beta_xij ** 0.5 * (-beta_xjyi * (cos(mu_ix - mu_jx - 2 * mu_iy) - cos(mu_ix - mu_jx + 2 * mu_iy))
                        + beta_xiyj * (cos(mu_jx - mu_ix - 2 * mu_jy) - cos(mu_jx - mu_ix + 2 * mu_jy))
                        + 2 * beta_yij * (-cos(mu_ix - mu_jx + 2 * mu_iy) - cos(mu_ix - mu_jx - 2 * mu_iy) + cos(mu_jx - mu_ix + 2 * mu_jy) + cos(mu_jx - mu_ix - 2 * mu_jy))) / 32
                hs20020i[i, j] = b3l * beta_xij ** 0.5 * (-beta_xjyi * cos(mu_ix - 3 * mu_jx + 2 * mu_iy) + (beta_xjyi + 4 * beta_yij) * cos(mu_ix + mu_jx - 2 * mu_iy)
                        + beta_xiyj * cos(mu_jx - 3 * mu_ix + 2 * mu_jy) - (beta_xiyj + 4 * beta_yij) * cos(mu_jx + mu_ix - 2 * mu_jy)) / 64
                hs20200i[i, j] = b3l * beta_xij ** 0.5 * (-beta_xjyi * cos(mu_ix - 3 * mu_jx - 2 * mu_iy) + (beta_xjyi - 4 * beta_yij) * cos(mu_ix + mu_jx + 2 * mu_iy)
                        + beta_xiyj * cos(mu_jx - 3 * mu_ix - 2 * mu_jy) - (beta_xiyj - 4 * beta_yij) * cos(mu_jx + mu_ix + 2 * mu_jy)) / 64
                hs00310i[i, j] = b3l * beta_xij ** 0.5 * beta_yij * (-cos(mu_ix - mu_jx + 2 * mu_iy) + cos(mu_ix - mu_jx - 2 * mu_iy)
                        + cos(mu_jx - mu_ix + 2 * mu_jy) - cos(mu_jx - mu_ix - 2 * mu_jy)) / 32
                hs00400i[i, j] = b3l * beta_xij ** 0.5 * beta_yij * (-cos(mu_ijx + 2 * (mu_iy + mu_jy)) + cos(-mu_ijx + 2 * (mu_iy + mu_jy))) / 64

        for i in range(oct_num):
            b4l = ele_list[oct_index[i]].k3 * ele_list[oct_index[i]].length / 6
            beta_xi = (ele_list[oct_index[i]].betax + ele_list[oct_index[i] + 1].betax) / 2
            beta_yi = (ele_list[oct_index[i]].betay + ele_list[oct_index[i] + 1].betay) / 2
            mu_ix = (ele_list[oct_index[i]].psix + ele_list[oct_index[i] + 1].psix) / 2
            mu_iy = (ele_list[oct_index[i]].psiy + ele_list[oct_index[i] + 1].psiy) / 2
            hos22000[i] = -3 * b4l * beta_xi**2 / 32
            hos11110[i] = 3 * b4l * beta_xi * beta_yi / 8
            hos00220[i] = -3 * b4l * beta_yi**2 / 32
            hos31000r[i] = -b4l * beta_xi ** 2 * cos(2 * mu_ix) / 16
            hos40000r[i] = -b4l * beta_xi ** 2 * cos(4 * mu_ix) / 64
            hos20110r[i] = 3 * b4l * beta_xi * beta_yi * cos(2 * mu_ix) / 16
            hos11200r[i] = 3 * b4l * beta_xi * beta_yi * cos(2 * mu_iy) / 16
            hos20020r[i] = 3 * b4l * beta_xi * beta_yi * cos(2 * mu_ix - 2 * mu_iy) / 32
            hos20200r[i] = 3 * b4l * beta_xi * beta_yi * cos(2 * mu_ix + 2 * mu_iy) / 32
            hos00310r[i] = -b4l * beta_yi ** 2 * cos(2 * mu_iy) / 16
            hos00400r[i] = -b4l * beta_yi ** 2 * cos(4 * mu_iy) / 64
            hos31000i[i] = -b4l * beta_xi ** 2 * sin(2 * mu_ix) / 16
            hos40000i[i] = -b4l * beta_xi ** 2 * sin(4 * mu_ix) / 64
            hos20110i[i] = 3 * b4l * beta_xi * beta_yi * sin(2 * mu_ix) / 16
            hos11200i[i] = 3 * b4l * beta_xi * beta_yi * sin(2 * mu_iy) / 16
            hos20020i[i] = 3 * b4l * beta_xi * beta_yi * sin(2 * mu_ix - 2 * mu_iy) / 32
            hos20200i[i] = 3 * b4l * beta_xi * beta_yi * sin(2 * mu_ix + 2 * mu_iy) / 32
            hos00310i[i] = -b4l * beta_yi ** 2 * sin(2 * mu_iy) / 16
            hos00400i[i] = -b4l * beta_yi ** 2 * sin(4 * mu_iy) / 64
        current_ind = 0
        for k in nonlinear_index:    # 起点在直线段变化时，四阶项和ADTS项只关心相对相移，三阶项角度变化，绝对值不变，所以只计算六极铁处就够了
            s[current_ind*2 + 1] = ele_list[k].s
            s[current_ind*2 + 2] = ele_list[k].s
            current_ind += 1
        s[current_ind*2 + 1] = ele_list[total_ele_count-1].s
        current_ind = 1       
        hts00111 = hts11001 = hts20001r = hts20001i = hts00201r = hts00201i = hts10002r = hts10002i = hts22000 = hts00220 = hts11110 = 0
        hts21000r = hts21000i = hts30000r = hts30000i = hts10110r = hts10110i = hts10200r = hts10200i = hts10020r = hts10020i = 0
        hts31000r = hts40000r = hts20110r = hts11200r = hts20020r = hts20200r = hts00310r = hts00400r = hts31000i = hts40000i = hts20110i = hts11200i = hts20020i = hts20200i = hts00310i = hts00400i = 0 
        quad_num = 0
        sext_num = 0
        oct_num = 0
        for ind in range(nonlinear_num.shape[0]):  # [is_quad, is_sext, is_oct]
            if nonlinear_num[ind, 0]:
                hts20001r += hqs20001r[quad_num]
                hts00201r += hqs00201r[quad_num]
                hts10002r += hqs10002r[quad_num]
                hts20001i += hqs20001i[quad_num]
                hts00201i += hqs00201i[quad_num]
                hts10002i += hqs10002i[quad_num]
                hts11001 += hqs11001[quad_num]
                hts00111 += hqs00111[quad_num]
                quad_num += 1
            if nonlinear_num[ind, 1]:
                hts20001r += hs20001r[sext_num]
                hts00201r += hs00201r[sext_num]
                hts10002r += hs10002r[sext_num]
                hts20001i += hs20001i[sext_num]
                hts00201i += hs00201i[sext_num]
                hts10002i += hs10002i[sext_num]
                hts11001 += hs11001[sext_num]
                hts00111 += hs00111[sext_num]
                hts21000r += hs21000r[sext_num]
                hts21000i += hs21000i[sext_num]
                hts30000r += hs30000r[sext_num]
                hts30000i += hs30000i[sext_num]
                hts10110r += hs10110r[sext_num]
                hts10110i += hs10110i[sext_num]
                hts10200r += hs10200r[sext_num]
                hts10200i += hs10200i[sext_num]
                hts10020r += hs10020r[sext_num]
                hts10020i += hs10020i[sext_num]
                for j in range(sext_num):
                    hts22000 += hs22000[sext_num, j]
                    hts11110 += hs11110[sext_num, j]
                    hts00220 += hs00220[sext_num, j]
                    hts31000r += hs31000r[sext_num, j]
                    hts40000r += hs40000r[sext_num, j]
                    hts00310r += hs00310r[sext_num, j]
                    hts20020r += hs20020r[sext_num, j]
                    hts20110r += hs20110r[sext_num, j]
                    hts00400r += hs00400r[sext_num, j]
                    hts20200r += hs20200r[sext_num, j]
                    hts11200r += hs11200r[sext_num, j]
                    hts31000i += hs31000i[sext_num, j]
                    hts40000i += hs40000i[sext_num, j]
                    hts00310i += hs00310i[sext_num, j]
                    hts20020i += hs20020i[sext_num, j]
                    hts20110i += hs20110i[sext_num, j]
                    hts00400i += hs00400i[sext_num, j]
                    hts20200i += hs20200i[sext_num, j]
                    hts11200i += hs11200i[sext_num, j]
                sext_num += 1
            if nonlinear_num[ind, 2]:
                hts22000 += hos22000[oct_num]
                hts11110 += hos11110[oct_num]
                hts00220 += hos00220[oct_num]
                hts31000r += hos31000r[oct_num]
                hts40000r += hos40000r[oct_num]
                hts00310r += hos00310r[oct_num]
                hts20020r += hos20020r[oct_num]
                hts20110r += hos20110r[oct_num]
                hts00400r += hos00400r[oct_num]
                hts20200r += hos20200r[oct_num]
                hts11200r += hos11200r[oct_num]
                hts31000i += hos31000i[oct_num]
                hts40000i += hos40000i[oct_num]
                hts00310i += hos00310i[oct_num]
                hts20020i += hos20020i[oct_num]
                hts20110i += hos20110i[oct_num]
                hts00400i += hos00400i[oct_num]
                hts20200i += hos20200i[oct_num]
                hts11200i += hos11200i[oct_num]   
                oct_num += 1
            f20001[current_ind*2] = sqrt(hts20001r ** 2 + hts20001i ** 2)
            f00201[current_ind*2] = sqrt(hts00201r ** 2 + hts00201i ** 2)
            f10002[current_ind*2] = sqrt(hts10002r ** 2 + hts10002i ** 2)
            f11001[current_ind*2] = fabs(hts11001)
            f00111[current_ind*2] = fabs(hts00111)
            f20001[current_ind*2 + 1] = f20001[current_ind*2]
            f00201[current_ind*2 + 1] = f00201[current_ind*2]
            f10002[current_ind*2 + 1] = f10002[current_ind*2]
            f11001[current_ind*2 + 1] = f11001[current_ind*2]
            f00111[current_ind*2 + 1] = f00111[current_ind*2]
            f21000[current_ind*2] = sqrt(hts21000r ** 2 + hts21000i ** 2)
            f30000[current_ind*2] = sqrt(hts30000r ** 2 + hts30000i ** 2)
            f10110[current_ind*2] = sqrt(hts10110r ** 2 + hts10110i ** 2)
            f10200[current_ind*2] = sqrt(hts10200r ** 2 + hts10200i ** 2)
            f10020[current_ind*2] = sqrt(hts10020r ** 2 + hts10020i ** 2)
            f21000[current_ind*2 + 1] = f21000[current_ind*2]
            f30000[current_ind*2 + 1] = f30000[current_ind*2]
            f10110[current_ind*2 + 1] = f10110[current_ind*2]
            f10200[current_ind*2 + 1] = f10200[current_ind*2]
            f10020[current_ind*2 + 1] = f10020[current_ind*2]
            f31000[current_ind*2] = sqrt(hts31000r ** 2 + hts31000i ** 2)
            f40000[current_ind*2] = sqrt(hts40000r ** 2 + hts40000i ** 2)
            f00310[current_ind*2] = sqrt(hts00310r ** 2 + hts00310i ** 2)
            f20020[current_ind*2] = sqrt(hts20020r ** 2 + hts20020i ** 2)
            f20110[current_ind*2] = sqrt(hts20110r ** 2 + hts20110i ** 2)
            f00400[current_ind*2] = sqrt(hts00400r ** 2 + hts00400i ** 2)
            f20200[current_ind*2] = sqrt(hts20200r ** 2 + hts20200i ** 2)
            f11200[current_ind*2] = sqrt(hts11200r ** 2 + hts11200i ** 2)
            f22000[current_ind*2] = fabs(hts22000)
            f11110[current_ind*2] = fabs(hts11110)
            f00220[current_ind*2] = fabs(hts00220)
            f22000[current_ind*2 + 1] = f22000[current_ind*2]
            f11110[current_ind*2 + 1] = f11110[current_ind*2]
            f00220[current_ind*2 + 1] = f00220[current_ind*2]            
            f31000[current_ind*2 + 1] = f31000[current_ind*2]
            f40000[current_ind*2 + 1] = f40000[current_ind*2]
            f00310[current_ind*2 + 1] = f00310[current_ind*2]
            f20020[current_ind*2 + 1] = f20020[current_ind*2]
            f20110[current_ind*2 + 1] = f20110[current_ind*2]
            f00400[current_ind*2 + 1] = f00400[current_ind*2]
            f20200[current_ind*2 + 1] = f20200[current_ind*2]
            f11200[current_ind*2 + 1] = f11200[current_ind*2]
            current_ind += 1
        RDTs_along_ring = {'s': s, 'h21000': f21000, 'h30000': f30000, 'h10110': f10110, 'h10020': f10020,
                     'h10200': f10200, 'h20001': f20001, 'h00201': f00201, 'h10002': f10002,
                     'h11001': f11001, 'h00111': f00111,
                     'h31000': f31000, 'h40000': f40000, 'h20110': f20110, 'h11200': f11200,
                     'h20020': f20020, 'h20200': f20200, 'h00310': f00310, 'h00400': f00400,
                     'h22000': f22000, 'h11110': f11110, 'h00220': f00220
                    }
        return RDTs_along_ring

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def driving_terms(self, n_periods=None, printout=True):
        """Compute nonlinear terms fluctuations along the lattice.
        The starting position is fixed, and the ending position varies.
        
        Return:
            {'h21000': np.ndarray, 'h30000': np.ndarray, 'h10110': np.ndarray, 'h10020': np.ndarray,
             'h10200': np.ndarray, 'h20001': np.ndarray, 'h00201': np.ndarray, 'h10002': np.ndarray,
             'h11001': np.ndarray, 'h00111': np.ndarray,
             'h31000': np.ndarray, 'h40000': np.ndarray, 'h20110': np.ndarray, 'h11200': np.ndarray,
             'h20020': np.ndarray, 'h20200': np.ndarray, 'h00310': np.ndarray, 'h00400': np.ndarray,
             'h22000': np.ndarray, 'h11110': np.ndarray, 'h00220': np.ndarray}.

        references:
        1. The Sextupole Scheme for the SLS: An Analytic Approach, SLS Note 09/97, Johan Bengtsson
        2. Explicit formulas for 2nd-order driving terms due to sextupoles and chromatic effects of quadrupoles, Chun-xi Wang
        3. Perspectives for future light source lattices incorporating yet uncommon magnets, S. C. Leemann and A. Streun"""
        
        cdef int geo_3rd_idx, geo_4th_idx, chr_3rd_idx, num_ele
        cdef complex h21000, h30000, h10110, h10020, h10200, h20001, h00201, h10002, jj
        cdef complex h12000, h01110, h01200, h01010, h12000j, h01110j, h01200j, h01010j
        cdef complex h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400, h22000, h11110, h00220
        cdef np.ndarray[dtype=np.complex128_t] f22000, f11110, f00220, f21000, f30000, f10110, f10020, f10200, f20001, f00201, f10002, f11001, f00111, f31000, f40000, f20110, f11200, f20020, f20200, f00310, f00400

        num_ele = len(self.elements)
        f21000 = np.zeros(num_ele, dtype='complex_')
        f30000 = np.zeros(num_ele, dtype='complex_')
        f10110 = np.zeros(num_ele, dtype='complex_')
        f10020 = np.zeros(num_ele, dtype='complex_')
        f10200 = np.zeros(num_ele, dtype='complex_')
        f20001 = np.zeros(num_ele, dtype='complex_')
        f00201 = np.zeros(num_ele, dtype='complex_')
        f10002 = np.zeros(num_ele, dtype='complex_')
        f22000 = np.zeros(num_ele, dtype='complex_')
        f11110 = np.zeros(num_ele, dtype='complex_')
        f00220 = np.zeros(num_ele, dtype='complex_')
        f31000 = np.zeros(num_ele, dtype='complex_')
        f40000 = np.zeros(num_ele, dtype='complex_')
        f20110 = np.zeros(num_ele, dtype='complex_')
        f11200 = np.zeros(num_ele, dtype='complex_')
        f20020 = np.zeros(num_ele, dtype='complex_')
        f20200 = np.zeros(num_ele, dtype='complex_')
        f00310 = np.zeros(num_ele, dtype='complex_')
        f00400 = np.zeros(num_ele, dtype='complex_')
        h21000 = h30000 = h10110 = h10020 = h10200 = h20001 = h00201 = h10002 = 0
        h22000 = h11110 = h00220 = h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
        geo_3rd_idx = 0
        geo_4th_idx = 0
        chr_3rd_idx = 0
        jj = complex(0, 1)
        for ele in self.elements:  # TODO: quad-sext
            if isinstance(ele, Sextupole):     #   0        1      2       3       4       5       6       7  
                rdts = ele.driving_terms()   # h21000, h30000, h10110, h10020, h10200, h20001, h00201, h10002
                h12000j = rdts[0].conjugate()
                h01110j = rdts[2].conjugate()
                h01200j = rdts[3].conjugate()
                h01020j = rdts[4].conjugate()
                h12000 = h21000.conjugate()
                h01110 = h10110.conjugate()
                h01200 = h10020.conjugate()
                h01020 = h10200.conjugate()         
                h22000 += jj * ((h21000 * h12000j - h12000 * rdts[0]) * 3
                                +(h30000 * rdts[1].conjugate() - h30000.conjugate() * rdts[1]) * 9) + rdts[8]
                h11110 += jj * ((h21000 * h01110j - h01110 * rdts[0]) * 2
                                -(h12000 * rdts[2] - h10110 * h12000j) * 2
                                -(h10020 * h01200j - h01200 * rdts[3]) * 4
                                +(h10200 * h01020j - h01020 * rdts[4]) * 4) + rdts[9]
                h00220 += jj * ((h10020 * h01200j - h01200 * rdts[3])
                                +(h10200 * h01020j - h01020 * rdts[4])
                                +(h10110 * h01110j - h01110 * rdts[2])) + rdts[10]
                h31000 += jj * 6 * (h30000 * h12000j - h12000 * rdts[1]) + rdts[11]
                h40000 += jj * 3 * (h30000 * rdts[0] - h21000 * rdts[1]) + rdts[12]
                h20110 += jj * ((h30000 * h01110j - h01110 * rdts[1]) * 3 
                               -(h21000 * rdts[2] - h10110 * rdts[0])
                                +(h10200 * rdts[3] - h10020 * rdts[4]) * 4) + rdts[13]
                h11200 += jj * ((h10200 * h12000j - h12000 * rdts[4]) * 2
                                +(h21000 * h01200j - h01200 * rdts[0]) * 2
                                +(h10200 * h01110j - h01110 * rdts[4]) * 2
                                +(h10110 * h01200j - h01200 * rdts[2]) * (-2)) + rdts[14]
                h20020 += jj * (-(h21000 * rdts[3] - h10020 * rdts[0])
                                +(h30000 * h01020j - h01020 * rdts[1]) * 3
                                +(h10110 * rdts[3] - h10020 * rdts[2]) * 2) + rdts[15]
                h20200 += jj * ((h30000 * h01200j - h01200 * rdts[1]) * 3
                                +(h10200 * rdts[0] - h21000 * rdts[4])
                                +(h10110 * rdts[4] - h10200 * rdts[2]) * (-2)) + rdts[16]
                h00310 += jj * ((h10200 * h01110j - h01110 * rdts[4])
                                +(h10110 * h01200j - h01200 * rdts[2])) + rdts[17]
                h00400 += jj * (h10200 * h01200j - h01200 * rdts[4]) + rdts[18]
                f22000[geo_4th_idx] = h22000
                f11110[geo_4th_idx] = h11110
                f00220[geo_4th_idx] = h00220
                f31000[geo_4th_idx] = h31000
                f40000[geo_4th_idx] = h40000
                f20110[geo_4th_idx] = h20110
                f11200[geo_4th_idx] = h11200
                f20020[geo_4th_idx] = h20020
                f20200[geo_4th_idx] = h20200
                f00310[geo_4th_idx] = h00310
                f00400[geo_4th_idx] = h00400
                geo_4th_idx += 1            
                h21000 = h21000 + rdts[0]
                h30000 = h30000 + rdts[1]
                h10110 = h10110 + rdts[2]
                h10020 = h10020 + rdts[3]
                h10200 = h10200 + rdts[4]
                f21000[geo_3rd_idx] = h21000
                f30000[geo_3rd_idx] = h30000
                f10110[geo_3rd_idx] = h10110
                f10020[geo_3rd_idx] = h10020
                f10200[geo_3rd_idx] = h10200
                geo_3rd_idx += 1
                h20001 += rdts[5]
                h00201 += rdts[6]
                h10002 += rdts[7]
                f20001[chr_3rd_idx] = h20001
                f00201[chr_3rd_idx] = h00201
                f10002[chr_3rd_idx] = h10002
                chr_3rd_idx += 1
            elif isinstance(ele, Octupole):
                rdts = ele.driving_terms()  # h22000, h11110, h00220, h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400
                h22000 += rdts[0]
                h11110 += rdts[1]
                h00220 += rdts[2]
                h31000 += rdts[3]
                h40000 += rdts[4]
                h20110 += rdts[5]
                h11200 += rdts[6]
                h20020 += rdts[7]
                h20200 += rdts[8]
                h00310 += rdts[9]
                h00400 += rdts[10]
                f21000[geo_3rd_idx] = f21000[geo_3rd_idx - 1]
                f30000[geo_3rd_idx] = f30000[geo_3rd_idx - 1]
                f10110[geo_3rd_idx] = f10110[geo_3rd_idx - 1]
                f10020[geo_3rd_idx] = f10020[geo_3rd_idx - 1]
                f10200[geo_3rd_idx] = f10200[geo_3rd_idx - 1]
                geo_3rd_idx += 1
                f22000[geo_4th_idx] = h22000
                f11110[geo_4th_idx] = h11110
                f00220[geo_4th_idx] = h00220
                f31000[geo_4th_idx] = h31000
                f40000[geo_4th_idx] = h40000
                f20110[geo_4th_idx] = h20110
                f11200[geo_4th_idx] = h11200
                f20020[geo_4th_idx] = h20020
                f20200[geo_4th_idx] = h20200
                f00310[geo_4th_idx] = h00310
                f00400[geo_4th_idx] = h00400
                geo_4th_idx += 1 
            elif ele.k1:
                rdts = ele.driving_terms()  # h20001, h00201, h10002
                h20001 += rdts[0]
                h00201 += rdts[1]
                h10002 += rdts[2]
                f20001[chr_3rd_idx] = h20001
                f00201[chr_3rd_idx] = h00201
                f10002[chr_3rd_idx] = h10002
                chr_3rd_idx += 1
        
        phix = ele.psix
        phiy = ele.psiy
        R21000 = h21000 / (1 - cos(phix) - sin(phix) * jj)
        R30000 = h30000 / (1 - cos(phix * 3) - sin(phix * 3) * jj)
        R10110 = h10110 / (1 - cos(phix) - sin(phix) * jj)
        R10020 = h10020 / (1 - cos(phix - 2 * phiy) - sin(phix - 2 * phiy) * jj)
        R10200 = h10200 / (1 - cos(phix + 2 * phiy) - sin(phix + 2 * phiy) * jj)
        R20001 = h20001 / (1 - cos(2 * phix) - jj * sin(2 * phix))
        R00201 = h00201 / (1 - cos(2 * phiy) - jj * sin(2 * phiy))
        R10002 = h10002 / (1 - cos(phix) - jj * sin(phix))
        R12000 = R21000.conjugate()
        R01110 = R10110.conjugate()
        R01200 = R10020.conjugate()
        R01020 = R10200.conjugate()
        h12000 = h21000.conjugate()
        h01110 = h10110.conjugate()
        h01200 = h10020.conjugate()
        h01020 = h10200.conjugate()

        h22000 = jj * ((h21000 * R12000 - h12000 * R21000) * 3
                        +(h30000 * R30000.conjugate() - h30000.conjugate() * R30000) * 9) + h22000
        h11110 = jj * ((h21000 * R01110 - h01110 * R21000) * 2
                        -(h12000 * R10110 - h10110 * R12000) * 2
                        -(h10020 * R01200 - h01200 * R10020) * 4
                        +(h10200 * R01020 - h01020 * R10200) * 4) + h11110
        h00220 = jj * ((h10020 * R01200 - h01200 * R10020)
                        +(h10200 * R01020 - h01020 * R10200)
                        +(h10110 * R01110 - h01110 * R10110)) + h00220
        h31000 = jj * 6 * (h30000 * R12000 - h12000 * R30000) + h31000
        h40000 = jj * 3 * (h30000 * R21000 - h21000 * R30000) + h40000
        h20110 = jj * ((h30000 * R01110 - h01110 * R30000) * 3 
                       -(h21000 * R10110 - h10110 * R21000)
                        +(h10200 * R10020 - h10020 * R10200) * 4) + h20110
        h11200 = jj * ((h10200 * R12000 - h12000 * R10200) * 2
                        +(h21000 * R01200 - h01200 * R21000) * 2
                        +(h10200 * R01110 - h01110 * R10200) * 2
                        +(h10110 * R01200 - h01200 * R10110) * (-2)) + h11200
        h20020 = jj * (-(h21000 * R10020 - h10020 * R21000)
                        +(h30000 * R01020 - h01020 * R30000) * 3
                        +(h10110 * R10020 - h10020 * R10110) * 2) + h20020
        h20200 = jj * ((h30000 * R01200 - h01200 * R30000) * 3
                        +(h10200 * R21000 - h21000 * R10200)
                        +(h10110 * R10200 - h10200 * R10110) * (-2)) + h20200
        h00310 = jj * ((h10200 * R01110 - h01110 * R10200)
                        +(h10110 * R01200 - h01200 * R10110)) + h00310
        h00400 = jj * (h10200 * R01200 - h01200 * R10200) + h00400

        R31000 = h31000 / (1 - cos(2 * phix) - jj * sin(2 * phix))
        R40000 = h40000 / (1 - cos(4 * phix) - jj * sin(4 * phix))
        R20110 = h20110 / (1 - cos(2 * phix) - jj * sin(2 * phix))
        R11200 = h11200 / (1 - cos(2 * phiy) - jj * sin(2 * phiy))
        R20020 = h20020 / (1 - cos(2 * phix - 2 * phiy) - jj * sin(2 * phix - 2 * phiy))
        R20200 = h20200 / (1 - cos(2 * phix + 2 * phiy) - jj * sin(2 * phix + 2 * phiy))
        R00310 = h00310 / (1 - cos(2 * phiy) - jj * sin(2 * phiy))
        R00400 = h00400 / (1 - cos(4 * phiy) - jj * sin(4 * phiy))

        n_periods = self.n_periods if n_periods is None else n_periods
        driving_terms = DrivingTerms(n_periods, phix, phiy,
                R21000, R30000, R10110, R10020, R10200, R20001, R00201, R10002,
                h22000, h11110, h00220, R31000, R40000, R20110, R11200, R20020, R20200, R00310, R00400, 
                f21000[:geo_3rd_idx], f30000[:geo_3rd_idx], f10110[:geo_3rd_idx], f10020[:geo_3rd_idx],
                f10200[:geo_3rd_idx], f20001[:chr_3rd_idx], f00201[:chr_3rd_idx], f10002[:chr_3rd_idx],
                # f22000[:geo_4th_idx], f11110[:geo_4th_idx], f00220[:geo_4th_idx],
                f31000[:geo_4th_idx], f40000[:geo_4th_idx], f20110[:geo_4th_idx], f11200[:geo_4th_idx],
                f20020[:geo_4th_idx], f20200[:geo_4th_idx], f00310[:geo_4th_idx], f00400[:geo_4th_idx])
        if printout:
            print(driving_terms)
        return driving_terms

    def higher_order_chromaticity(self, printout=True, order=3, delta=1e-3, matrix_precision=1e-9, resdl_limit=1e-16):
        """compute higher order chromaticity with the tunes of 4d off-momentum closed orbit.
            
            try to reset the value of delta, precision and resdl_limit if the result is wrong.
        you can call track_4d_closed_orbit() function to see the magnitude of the closed orbit, and the matrix_precision
        should be much smaller than it.

        Args:
            order: 2 or 3.
            delta: the momentum deviation.
            matrix_precision: the small deviation to calculate transfer matrix by tracking.
            resdl_limit: the limit to judge if the orbit is closed.

        Returns:
            a dictionary of chromaticities.
             {'xi2x': float,
              'xi2y': float,
              'xi3x': float,
              'xi3y': float}

        """

        def closed_orbit_tune(deviation):
            """reference: SAMM: Simple Accelerator Modelling in Matlab, A. Wolski, 2013"""
            cdef double[6] particle0, particle1, particle2, particle3, particle4#, particle5, particle6
            cdef double precision
            cdef np.ndarray[dtype=np.float64_t, ndim=2] matrix = np.zeros([4, 4])
            cdef double[:, :] mv = matrix
            xco = np.array([0.0, 0.0, 0.0, 0.0])
            resdl = 1
            iter_times = 1
            precision = matrix_precision
            d = np.zeros(4)
            while iter_times <= 10 and resdl > resdl_limit:
                particle0 = [0, 0, 0, 0, 0, 0]
                particle1 = [precision, 0, 0, 0, 0, 0]
                particle2 = [0, precision, 0, 0, 0, 0]
                particle3 = [0, 0, precision, 0, 0, 0]
                particle4 = [0, 0, 0, precision, 0, 0]
                for i in range(4):
                    particle0[i] = particle0[i] + xco[i]
                    particle1[i] = particle1[i] + xco[i]
                    particle2[i] = particle2[i] + xco[i]
                    particle3[i] = particle3[i] + xco[i]
                    particle4[i] = particle4[i] + xco[i]
                particle0[5] = deviation
                particle1[5] = deviation
                particle2[5] = deviation
                particle3[5] = deviation
                particle4[5] = deviation
                for nper in range(self.n_periods):
                    for ele in self.elements:
                        flag0 = symplectic_track_ele(ele, particle0)
                        flag1 = symplectic_track_ele(ele, particle1)
                        flag2 = symplectic_track_ele(ele, particle2)
                        flag3 = symplectic_track_ele(ele, particle3)
                        flag4 = symplectic_track_ele(ele, particle4)
                        if (flag0 + flag1 + flag2 + flag3 + flag4) != 0:
                            raise Exception(f'particle lost at {ele.s}')
                for i in range(4):
                    mv[i, 0] = (particle1[i] - particle0[i]) / precision
                    mv[i, 1] = (particle2[i] - particle0[i]) / precision
                    mv[i, 2] = (particle3[i] - particle0[i]) / precision
                    mv[i, 3] = (particle4[i] - particle0[i]) / precision
                for i in range(4):
                    d[i] = particle0[i] - xco[i]
                dco = np.linalg.inv(np.identity(4) - matrix).dot(d)
                xco = xco + dco
                resdl = dco.dot(dco.T)
                iter_times += 1
            cos_mu = (mv[0, 0] + mv[1, 1]) / 2
            assert fabs(cos_mu) < 1, "can not find period solution, UNSTABLE!!!"
            nux = acos(cos_mu) * np.sign(mv[0, 1]) / 2 / pi
            nuy = acos((mv[2, 2] + mv[3, 3]) / 2) * np.sign(mv[2, 3]) / 2 / pi
            return nux - np.floor(nux), nuy - np.floor(nuy)
        
        try:
            nux1, nuy1 = closed_orbit_tune(delta)
            nux_1, nuy_1 = closed_orbit_tune(-delta)
            if order == 3:
                nux3, nuy3 = closed_orbit_tune(3 * delta)
                nux_3, nuy_3 = closed_orbit_tune(-3 * delta)
        except Exception as e:
            if printout:
                print(e)
                print('!!!!!!!\ncan not find off-momentum closed orbit, try smaller delta.\n    !!!! you may need to change matrix_precision, too.')
            return {'xi2x': 1e9, 'xi2y': 1e9, 'xi3x': 1e9, 'xi3y': 1e9}
        xi2x = (nux1 + nux_1 - 2 * (self.nux - int(self.nux))) / 2 / delta ** 2
        xi2y = (nuy1 + nuy_1 - 2 * (self.nuy - int(self.nuy))) / 2 / delta ** 2
        if order == 3:
            xi3x = (nux3 - nux_3 + 3 * nux_1 - 3 * nux1) / (delta * 2) ** 3 / 6
            xi3y = (nuy3 - nuy_3 + 3 * nuy_1 - 3 * nuy1) / (delta * 2) ** 3 / 6
        else:
            xi3x = 0
            xi3y = 0
        if printout:
            print(f'xi2x: {xi2x:.2f}, xi2y: {xi2y:.2f}, xi3x: {xi3x:.2f}, xi3y: {xi3y:.2f}')
        return {'xi2x': xi2x, 'xi2y': xi2y, 'xi3x': xi3x, 'xi3y': xi3y}

    def output_matrix(self, file_name: str = u'matrix.txt'):
        """output uncoupled matrix for each element and contained matrix"""

        matrix = np.identity(6)
        file = open(file_name, 'w')
        location = 0.0
        for ele in self.elements:
            file.write(f'{ele.type} {ele.name} at s={location:.6f},  {ele.magnets_data()}\n')
            location = location + ele.length
            file.write(str(ele.matrix) + '\n')
            file.write('contained matrix:\n')
            matrix = ele.matrix.dot(matrix)
            file.write(str(matrix))
            file.write('\n\n--------------------------\n\n')
        file.close()

    def output_twiss(self, file_name: str = u'twiss_data.txt'):
        """ output_twiss(self, file_name: str = 'twiss_data.txt')
        columns: (s, ElementName, betax, alphax, psix, betay, alphay, psiy, etax, etaxp)"""

        file1 = open(file_name, 'w')
        file1.write('& s, ElementName, betax, alphax, psix, betay, alphay, psiy, etax, etaxp\n')
        for ele in self.elements:
            file1.write(f'{ele.s:.6e} {ele.name:10} {ele.betax:.6e}  {ele.alphax:.6e}  {ele.psix / 2 / pi:.6e}  '
                        f'{ele.betay:.6e}  {ele.alphay:.6e}  {ele.psiy / 2 / pi:.6e}  {ele.etax:.6e}  {ele.etaxp:.6e}\n')
        file1.close()

    def __mul__(self, other):
        assert isinstance(other, int), 'can only multiply int.'
        newlattice = CSLattice(self.elements * other)
        return newlattice
        
    def __str__(self):
        val = ""
        val += f'{str("Length ="):11} {self.length:9.3f} m'
        val += f'\n{str("angle ="):11} {self.angle:9.3f}'
        val += f'\n{str("abs_angle ="):11} {self.abs_angle:9.3f}'
        val += f'\n{str("nux ="):11} {self.nux:9.4f}'
        val += f'\n{str("nuy ="):11} {self.nuy:9.4f}'
        val += f'\n{str("I1 ="):11} {self.I1:9.5e}'
        val += f'\n{str("I2 ="):11} {self.I2:9.5e}'
        val += f'\n{str("I3 ="):11} {self.I3:9.5e}'
        val += f'\n{str("I4 ="):11} {self.I4:9.5e}'
        val += f'\n{str("I5 ="):11} {self.I5:9.5e}'
        val += f'\n{str("energy ="):11} {refenergy:9.2e} MeV'
        # val += f'\n{str("gamma ="):11} {refgamma:9.2f}'
        val += f'\n{str("U0 ="):11} {self.U0 * 1000:9.2f} keV'
        val += f'\n{str("sigma_e ="):11} {self.sigma_e:9.3e}'
        val += f'\n{str("emittance ="):11} {self.emittance:9.3e} m*rad'
        val += f'\n{str("Jx ="):11} {self.Jx:9.4f}'
        val += f'\n{str("Js ="):11} {self.Js:9.4f}'
        val += f'\n{str("Tperiod ="):11} {1 / self.f_c:9.3e} sec'
        val += f'\n{str("alpha ="):11} {self.alpha:9.3e}'
        val += f'\n{str("eta_p ="):11} {self.etap:9.3e}'
        val += f'\n{str("tau_e ="):11} {self.tau_s * 1e3:9.2f} msec'
        val += f'\n{str("tau_x ="):11} {self.tau_x * 1e3:9.2f} msec'
        val += f'\n{str("tau_y ="):11} {self.tau_y * 1e3:9.2f} msec'
        val += f'\n{str("natural_xi_x ="):11} {self.natural_xi_x:9.2f}'
        val += f'\n{str("natural_xi_y ="):11} {self.natural_xi_y:9.2f}'
        val += f'\n{str("xi_x ="):11} {self.xi_x:9.2f}'
        val += f'\n{str("xi_y ="):11} {self.xi_y:9.2f}'
        if self.sigma_z is not None:
            val += ("\nnuz =       " + str(self.nuz))
            val += ("\nsigma_z =   " + str(self.sigma_z))
        return val

