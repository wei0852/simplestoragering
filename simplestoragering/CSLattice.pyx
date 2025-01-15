# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: profile=False
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
from .exceptions import Unstable
from libc.math cimport isnan, cos, sin, pow, sqrt, acos, fabs


class CSLattice(object):
    """CSLattice(ele_list: list[Elements], n_periods: int = 1, coupling: float = 0.00)
    lattice object, solve by Courant-Snyder method.

    Attributes:
        length: float.
        n_periods: int, number of periods.
        angle, abs_angle: float.
        elements: list of Elements.
        mark: Dictionary of Mark in lattice. The key is the name of Mark, and the value is a list of Mark with the same name.
        initial_twiss: np.ndarray, [betax, alphax, betay, alphay, etax, etaxp, etay, etayp], eta_y0=[0,0] because the coupled motion has not be considered yet.
        nux, nuy: Tunes.
        xi_x, xi_y: Chromaticities (Calculated with linear model. use track_chromaticity for more accurate results.).
        natural_xi_x, natural_xi_y: Natural chromaticities.
        I1, I2, I3, I4, I5: radiation integrals.
        Jx, Jy, Js: horizontal / vertical / longitudinal damping partition number.
        sigma_e: natural energy spread.
        emittance: natural emittance.
        U0: energy loss [MeV].
        f_c: frequency.
        tau_s, tau_x, tau_y: longitudinal / horizontal / vertical damping time.
        alpha: Momentum compaction.
        etap: phase slip factor.
    
    Methods:
        set_initial_twiss(betax, alphax, betay, alphay, etax, etaxp, etay, etayp)
        linear_optics(periodicity=True, line_mode=False)
        driving_terms(verbose=True)
        adts(verbose=True)
        track_chromaticity(order=2, verbose=True, delta=None, matrix_precision=1e-9, resdl_limit=1e-12)
        slice_elements(drift_maxlength=10.0, bend_maxlength=10.0, quad_maxlength=10.0, sext_maxlength=10.0)
        output_twiss(file_name: str = u'twiss_data.txt')
    """

    def __init__(self, ele_list: list, n_periods: int = 1, delta: float = 0):
        self.delta = delta
        self.length = 0
        self.n_periods = n_periods
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
        # initialize twiss: betax, alphax, betay, alphay, etax, etaxp, etay, etayp
        self.initial_twiss = None
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

        periodicity: if True, the periodic solution will be the initial twiss data. Otherwise initial twiss should be set by CSLattice.set_initial_twiss(betax, alphax, betay, alphay, etax, etaxp, etay, etayp)
        line_mode: if True, the storage ring parameters such as emittance and damping time are not calculated."""

        if periodicity:
            self._the_periodic_solution()
        else:
            if self.initial_twiss is None:
                raise Exception('need initial twiss data. use set_initial_twiss() or linear_optics(periodicity=True)')
        self.__solve_along()
        #  global_parameters
        self.U0 = Cr * refenergy ** 4 * self.I2 / (2 * pi)
        if not line_mode:
            self.f_c = c * refbeta / self.length
            self.Jx = 1 - self.I4 / self.I2
            self.Jy = 1
            self.Js = 2 + self.I4 / self.I2
            if self.Js <= 0:
                raise Unstable('invalid longitudinal damping partition number')
            self.sigma_e = refgamma * np.sqrt(Cq * self.I3 / (self.Js * self.I2))
            self.emittance = Cq * refgamma * refgamma * self.I5 / (self.Jx * self.I2)
            self.tau0 = 2 * refenergy / self.U0 / self.f_c
            self.tau_s = self.tau0 / self.Js
            self.tau_x = self.tau0 / self.Jx
            self.tau_y = self.tau0 / self.Jy
            self.alpha = self.I1 * self.f_c / c  # momentum compaction factor
            self.etap = self.alpha - 1 / refgamma ** 2  # phase slip factor

    def _the_periodic_solution(self, delta=0.0, use_track_matrix=False):
        """compute periodic solution and initialize twiss"""
        def track_matrix(delta, matrix_precision=1e-9, resdl_limit=1e-12, verbose=True):
            cdef double[6] particle0, particle1, particle2, particle3, particle4, particle5
            cdef double precision
            cdef np.ndarray[dtype=np.float64_t, ndim=2] matrix = np.zeros([6, 6])
            cdef double[:, :] mv = matrix
            
            xco = np.array([0.0, 0.0, 0.0, 0.0])
            resdl = 1
            iter_times = 1
            precision = matrix_precision
            d = np.zeros(4)
            while iter_times <= 10 and resdl > resdl_limit:
                particle0 = [0, 0, 0, 0, 0, delta]
                particle1 = [precision, 0, 0, 0, 0, delta]
                particle2 = [0, precision, 0, 0, 0, delta]
                particle3 = [0, 0, precision, 0, 0, delta]
                particle4 = [0, 0, 0, precision, 0, delta]
                particle5 = [0, 0, 0, 0, 0 ,precision + delta]
                for i in range(4):
                    particle0[i] = particle0[i] + xco[i]
                    particle1[i] = particle1[i] + xco[i]
                    particle2[i] = particle2[i] + xco[i]
                    particle3[i] = particle3[i] + xco[i]
                    particle4[i] = particle4[i] + xco[i]
                    particle5[i] = particle5[i] + xco[i]
                for nper in range(self.n_periods):
                    for ele in self.elements:
                        ele.closed_orbit = particle0
                        flag0 = symplectic_track_ele(ele, particle0)
                        flag1 = symplectic_track_ele(ele, particle1)
                        flag2 = symplectic_track_ele(ele, particle2)
                        flag3 = symplectic_track_ele(ele, particle3)
                        flag4 = symplectic_track_ele(ele, particle4)
                        flag5 = symplectic_track_ele(ele, particle5)
                        if (flag0 + flag1 + flag2 + flag3 + flag4 + flag5) != 0:
                            raise Unstable(f'particle lost at {ele.s}')
                for i in range(4):
                    mv[i, 0] = (particle1[i] - particle0[i]) / precision
                    mv[i, 1] = (particle2[i] - particle0[i]) / precision
                    mv[i, 2] = (particle3[i] - particle0[i]) / precision
                    mv[i, 3] = (particle4[i] - particle0[i]) / precision
                    mv[i, 5] = (particle5[i] - particle0[i]) / precision
                for i in range(4):
                    d[i] = particle0[i] - xco[i]
                dco = np.linalg.inv(np.identity(4) - matrix[:4, :4]).dot(d)
                xco = xco + dco
                resdl = sum(dco ** 2) ** 0.5
                iter_times += 1
            return matrix
        if delta != 0 or use_track_matrix:
            matrix = track_matrix(delta)
        else:
            matrix = line_matrix(self.elements)
        cos_mu = (matrix[0, 0] + matrix[1, 1]) / 2
        if not fabs(cos_mu) < 1:
            raise Unstable('can not find period solution')
        mux = acos(cos_mu) * np.sign(matrix[0, 1])
        cos_mu = (matrix[2, 2] + matrix[3, 3]) / 2
        if not fabs(cos_mu) < 1:
            raise Unstable('can not find period solution')      
        muy = acos(cos_mu) * np.sign(matrix[2, 3])
        self.initial_twiss = np.zeros(8)
        # x direction
        self.initial_twiss[0] = matrix[0, 1] / sin(mux)
        self.initial_twiss[1] = (matrix[0, 0] - matrix[1, 1]) / (2 * sin(mux))
        # y direction
        self.initial_twiss[2] = matrix[2, 3] / sin(muy)
        self.initial_twiss[3] = (matrix[2, 2] - matrix[3, 3]) / (2 * sin(muy))
        # solve eta
        sub_matrix_x = matrix[0:2, 0:2]
        matrix_etax = np.array([matrix[0, 5], matrix[1, 5]])
        self.initial_twiss[4:6] = np.linalg.inv(np.identity(2) - sub_matrix_x).dot(matrix_etax)
        sub_matrix_y = matrix[2:4, 2:4]
        matrix_etay = np.array([matrix[2, 5], matrix[3, 5]])
        self.initial_twiss[6:8] = np.linalg.inv(np.identity(2) - sub_matrix_y).dot(matrix_etay)

    def set_initial_twiss(self, betax, alphax, betay, alphay, etax, etaxp, etay, etayp):
        """set_initial_twiss(betax, alphax, betay, alphay, etax, etaxp, etay, etayp)
        work if run CSLattice.linear_optics() with periodicity=False.
        """
        self.initial_twiss = np.array([betax, alphax, betay, alphay,
                                      etax, etaxp, etay, etayp])

    def __solve_along(self):
        [betax, alphax, betay, alphay, etax, etaxp, etay, etayp] = self.initial_twiss
        gammax = (1 + alphax**2) / betax
        gammay = (1 + alphay**2) / betay
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

    def off_momentum_optics(self, delta=0.0):
        """off_momentum_optics(self, delta=0.0)
        calculate the matrix by tracking, then find the 4d closed orbit and compute twiss parameters.
        delta: momentum deviation.
        """
        self.delta = delta
        self._the_periodic_solution(delta, True)
        [betax, alphax, betay, alphay, etax, etaxp, etay, etayp] = self.initial_twiss
        gammax = (1 + alphax**2) / betax
        gammay = (1 + alphay**2) / betay
        psix = psiy  = 0
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
            twiss = ele.off_momentum_optics(delta)
        self.nux = self.elements[-1].nux * self.n_periods
        self.nuy = self.elements[-1].nuy * self.n_periods
        # Clear the storage ring parameters to avoid misinformation.
        # TODO: Is it necessary to calculate the off-momentum ring parameters?
        self.I1 = None
        self.I2 = None
        self.I3 = None
        self.I4 = None
        self.I5 = None
        self.natural_xi_x = None
        self.natural_xi_y = None
        self.xi_x = None
        self.xi_y = None
        self.U0 = None
        self.f_c = None
        self.Jx = None
        self.Jy = None
        self.Js = None
        self.sigma_e = None
        self.emittance = None
        self.tau0 = None
        self.tau_s = None
        self.tau_x = None
        self.tau_y = None
        self.alpha = None
        self.etap = None

    def slice_elements(self, drift_length=10.0, bend_length=10.0, quad_length=10.0, sext_length=10.0, oct_length=10.0):
        """slice_elements(drift_length=10.0, bend_length=10.0, quad_length=10.0, sext_length=10.0) -> list[Element]
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
            elif isinstance(ele, Octupole):
                ele_slices += ele.slice(max(int(ele.length / oct_length), 1))
            else:
                ele_slices += ele.slice(1)
        return ele_slices
    
    @cython.cdivision(True)
    def adts(self, n_periods=None, verbose=True):
        """adts(self, verbose=True) -> dict
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
            if verbose:
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
        if verbose:
            print(f'ADTS terms, {n_periods} periods:')
            for k, b4l in nonlinear_terms.items():
                print(f'    {str(k):7}: {b4l:.2f}')
        return nonlinear_terms

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def driving_terms(self, n_periods=None, verbose=True):
        """driving_terms(self, n_periods=None, verbose=True) -> DrivingTerms
        Compute driving terms and the build-up fluctuations.
        
        Return DrivingTerms

        references:
        1. The Sextupole Scheme for the SLS: An Analytic Approach, SLS Note 09/97, Johan Bengtsson
        2. Explicit formulas for 2nd-order driving terms due to sextupoles and chromatic effects of quadrupoles, Chun-xi Wang
        3. Perspectives for future light source lattices incorporating yet uncommon magnets, S. C. Leemann and A. Streun"""
        
        cdef int geo_idx, chr_idx, num_ele
        cdef complex h21000, h30000, h10110, h10020, h10200, jj
        cdef complex h12000, h01110, h01200, h01010, h12000j, h01110j, h01200j, h01010j
        cdef complex h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400, h22000, h11110, h00220
        cdef np.ndarray[dtype=np.complex128_t] h22000s, h11110s, h00220s, h21000s, h30000s, h10110s, h10020s, h10200s, h31000s, h40000s, h20110s, h11200s, h20020s, h20200s, h00310s, h00400s

        num_ele = len(self.elements)
        h21000s = np.zeros(num_ele, dtype='complex_')
        h30000s = np.zeros(num_ele, dtype='complex_')
        h10110s = np.zeros(num_ele, dtype='complex_')
        h10020s = np.zeros(num_ele, dtype='complex_')
        h10200s = np.zeros(num_ele, dtype='complex_')
        h22000s = np.zeros(num_ele, dtype='complex_')
        h11110s = np.zeros(num_ele, dtype='complex_')
        h00220s = np.zeros(num_ele, dtype='complex_')
        h31000s = np.zeros(num_ele, dtype='complex_')
        h40000s = np.zeros(num_ele, dtype='complex_')
        h20110s = np.zeros(num_ele, dtype='complex_')
        h11200s = np.zeros(num_ele, dtype='complex_')
        h20020s = np.zeros(num_ele, dtype='complex_')
        h20200s = np.zeros(num_ele, dtype='complex_')
        h00310s = np.zeros(num_ele, dtype='complex_')
        h00400s = np.zeros(num_ele, dtype='complex_')
        h21000 = h30000 = h10110 = h10020 = h10200 = 0
        h22000 = h11110 = h00220 = h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
        geo_idx = 0  # k_i != 0, i >= 2
        jj = complex(0, 1)
        for ele in self.elements:  # TODO: quad-sext
            if ele.k2 or ele.k3:     #   0        1      2       3       4  
                rdts = ele.driving_terms(self.delta)   # h21000, h30000, h10110, h10020, h10200
                h12000j = rdts[0].conjugate()
                h01110j = rdts[2].conjugate()
                h01200j = rdts[3].conjugate()
                h01020j = rdts[4].conjugate()
                h12000 = h21000.conjugate()
                h01110 = h10110.conjugate()
                h01200 = h10020.conjugate()
                h01020 = h10200.conjugate()         
                h22000 += jj * ((h21000 * h12000j - h12000 * rdts[0]) * 3
                                +(h30000 * rdts[1].conjugate() - h30000.conjugate() * rdts[1]) * 9) + rdts[5]
                h11110 += jj * ((h21000 * h01110j - h01110 * rdts[0]) * 2
                                -(h12000 * rdts[2] - h10110 * h12000j) * 2
                                -(h10020 * h01200j - h01200 * rdts[3]) * 4
                                +(h10200 * h01020j - h01020 * rdts[4]) * 4) + rdts[6]
                h00220 += jj * ((h10020 * h01200j - h01200 * rdts[3])
                                +(h10200 * h01020j - h01020 * rdts[4])
                                +(h10110 * h01110j - h01110 * rdts[2])) + rdts[7]
                h31000 += jj * 6 * (h30000 * h12000j - h12000 * rdts[1]) + rdts[8]
                h40000 += jj * 3 * (h30000 * rdts[0] - h21000 * rdts[1]) + rdts[9]
                h20110 += jj * ((h30000 * h01110j - h01110 * rdts[1]) * 3 
                               -(h21000 * rdts[2] - h10110 * rdts[0])
                                +(h10200 * rdts[3] - h10020 * rdts[4]) * 4) + rdts[10]
                h11200 += jj * ((h10200 * h12000j - h12000 * rdts[4]) * 2
                                +(h21000 * h01200j - h01200 * rdts[0]) * 2
                                +(h10200 * h01110j - h01110 * rdts[4]) * 2
                                +(h10110 * h01200j - h01200 * rdts[2]) * (-2)) + rdts[11]
                h20020 += jj * (-(h21000 * rdts[3] - h10020 * rdts[0])
                                +(h30000 * h01020j - h01020 * rdts[1]) * 3
                                +(h10110 * rdts[3] - h10020 * rdts[2]) * 2) + rdts[12]
                h20200 += jj * ((h30000 * h01200j - h01200 * rdts[1]) * 3
                                +(h10200 * rdts[0] - h21000 * rdts[4])
                                +(h10110 * rdts[4] - h10200 * rdts[2]) * (-2)) + rdts[13]
                h00310 += jj * ((h10200 * h01110j - h01110 * rdts[4])
                                +(h10110 * h01200j - h01200 * rdts[2])) + rdts[14]
                h00400 += jj * (h10200 * h01200j - h01200 * rdts[4]) + rdts[15]
                h22000s[geo_idx] = h22000
                h11110s[geo_idx] = h11110
                h00220s[geo_idx] = h00220
                h31000s[geo_idx] = h31000
                h40000s[geo_idx] = h40000
                h20110s[geo_idx] = h20110
                h11200s[geo_idx] = h11200
                h20020s[geo_idx] = h20020
                h20200s[geo_idx] = h20200
                h00310s[geo_idx] = h00310
                h00400s[geo_idx] = h00400
                h21000 = h21000 + rdts[0]
                h30000 = h30000 + rdts[1]
                h10110 = h10110 + rdts[2]
                h10020 = h10020 + rdts[3]
                h10200 = h10200 + rdts[4]
                h21000s[geo_idx] = h21000
                h30000s[geo_idx] = h30000
                h10110s[geo_idx] = h10110
                h10020s[geo_idx] = h10020
                h10200s[geo_idx] = h10200
                geo_idx += 1
        
        phix = ele.psix
        phiy = ele.psiy
        f21000 = h21000 / (1 - cos(phix) - sin(phix) * jj)
        f30000 = h30000 / (1 - cos(phix * 3) - sin(phix * 3) * jj)
        f10110 = h10110 / (1 - cos(phix) - sin(phix) * jj)
        f10020 = h10020 / (1 - cos(phix - 2 * phiy) - sin(phix - 2 * phiy) * jj)
        f10200 = h10200 / (1 - cos(phix + 2 * phiy) - sin(phix + 2 * phiy) * jj)
        f12000 = f21000.conjugate()
        f01110 = f10110.conjugate()
        f01200 = f10020.conjugate()
        f01020 = f10200.conjugate()
        h12000 = h21000.conjugate()
        h01110 = h10110.conjugate()
        h01200 = h10020.conjugate()
        h01020 = h10200.conjugate()

        h22000 = jj * ((h21000 * f12000 - h12000 * f21000) * 3
                        +(h30000 * f30000.conjugate() - h30000.conjugate() * f30000) * 9) + h22000
        h11110 = jj * ((h21000 * f01110 - h01110 * f21000) * 2
                        -(h12000 * f10110 - h10110 * f12000) * 2
                        -(h10020 * f01200 - h01200 * f10020) * 4
                        +(h10200 * f01020 - h01020 * f10200) * 4) + h11110
        h00220 = jj * ((h10020 * f01200 - h01200 * f10020)
                        +(h10200 * f01020 - h01020 * f10200)
                        +(h10110 * f01110 - h01110 * f10110)) + h00220
        h31000 = jj * 6 * (h30000 * f12000 - h12000 * f30000) + h31000
        h40000 = jj * 3 * (h30000 * f21000 - h21000 * f30000) + h40000
        h20110 = jj * ((h30000 * f01110 - h01110 * f30000) * 3 
                       -(h21000 * f10110 - h10110 * f21000)
                        +(h10200 * f10020 - h10020 * f10200) * 4) + h20110
        h11200 = jj * ((h10200 * f12000 - h12000 * f10200) * 2
                        +(h21000 * f01200 - h01200 * f21000) * 2
                        +(h10200 * f01110 - h01110 * f10200) * 2
                        +(h10110 * f01200 - h01200 * f10110) * (-2)) + h11200
        h20020 = jj * (-(h21000 * f10020 - h10020 * f21000)
                        +(h30000 * f01020 - h01020 * f30000) * 3
                        +(h10110 * f10020 - h10020 * f10110) * 2) + h20020
        h20200 = jj * ((h30000 * f01200 - h01200 * f30000) * 3
                        +(h10200 * f21000 - h21000 * f10200)
                        +(h10110 * f10200 - h10200 * f10110) * (-2)) + h20200
        h00310 = jj * ((h10200 * f01110 - h01110 * f10200)
                        +(h10110 * f01200 - h01200 * f10110)) + h00310
        h00400 = jj * (h10200 * f01200 - h01200 * f10200) + h00400

        f31000 = h31000 / (1 - cos(2 * phix) - jj * sin(2 * phix))
        f40000 = h40000 / (1 - cos(4 * phix) - jj * sin(4 * phix))
        f20110 = h20110 / (1 - cos(2 * phix) - jj * sin(2 * phix))
        f11200 = h11200 / (1 - cos(2 * phiy) - jj * sin(2 * phiy))
        f20020 = h20020 / (1 - cos(2 * phix - 2 * phiy) - jj * sin(2 * phix - 2 * phiy))
        f20200 = h20200 / (1 - cos(2 * phix + 2 * phiy) - jj * sin(2 * phix + 2 * phiy))
        f00310 = h00310 / (1 - cos(2 * phiy) - jj * sin(2 * phiy))
        f00400 = h00400 / (1 - cos(4 * phiy) - jj * sin(4 * phiy))

        n_periods = self.n_periods if n_periods is None else n_periods
        driving_terms = DrivingTerms(n_periods, phix, phiy,
                f21000, f30000, f10110, f10020, f10200,
                h22000, h11110, h00220, f31000, f40000, f20110, f11200, f20020, f20200, f00310, f00400, 
                h21000s[:geo_idx], h30000s[:geo_idx], h10110s[:geo_idx], h10020s[:geo_idx],
                h10200s[:geo_idx],
                # h22000s[:geo_idx], h11110s[:geo_idx], h00220s[:geo_idx],
                h31000s[:geo_idx], h40000s[:geo_idx], h20110s[:geo_idx], h11200s[:geo_idx],
                h20020s[:geo_idx], h20200s[:geo_idx], h00310s[:geo_idx], h00400s[:geo_idx])
        if verbose:
            print(driving_terms)
        return driving_terms

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def chromatic_driving_terms(self, n_periods=None, verbose=True):
        """chromatic_driving_terms(self, n_periods=None, verbose=True)
        compute third-order chromatic driving terms.
        
        Return:
            one_turn: dictionary of complex {h11001, h00111, h20001, h00201, h10002},
            buildup: dictionary of lists of complex {h11001, h00111, h20001, h00201, h10002},
            natural: dictionary of lists of float {f11001, f00111, f20001, f00201, f10002}"""
        
        @cython.cdivision(True)
        def __betadrift(double beta0, double beta1, double alpha0, double L):
            cdef double avebeta, gamma0
            gamma0 = (alpha0 * alpha0 + 1) / beta0
            avebeta = 0.5 * (beta0 + beta1) - gamma0 * L * L / 6
            return avebeta

        @cython.cdivision(True)
        def __betafoc(double beta1, double alpha0, double alpha1, double K, double L):
            cdef double gamma1, avebeta
            gamma1 = (alpha1 * alpha1 + 1) / beta1
            avebeta = 0.5 * ((gamma1 + K * beta1) * L + alpha1 - alpha0) / K / L
            return avebeta
    
        @cython.cdivision(True)
        def __dispfoc(double dispp0, double dispp1, double K, double L):
            cdef double avedisp
            avedisp = (dispp0 - dispp1) / K / L
            return avedisp

        cdef int num_ele
        cdef complex h20001, h00201, h10002, jj, f00201, f20001, f10002
        cdef double h11001, h00111, b2l, betax, betay, etax, mux, muy, b3letax, phix, phiy
        cdef np.ndarray[dtype=np.float64_t] h11001s, h00111s
        cdef np.ndarray[dtype=np.complex128_t] h20001s, h00201s, h10002s, f20001s, f00201s, f10002s

        num_ele = len(self.elements) - 1
        n_periods = self.n_periods if n_periods is None else n_periods
        h11001s = np.zeros(num_ele * n_periods + 1)
        h00111s = np.zeros(num_ele * n_periods + 1)
        h20001s = np.zeros(num_ele * n_periods + 1, dtype='complex_')
        h00201s = np.zeros(num_ele * n_periods + 1, dtype='complex_')
        h10002s = np.zeros(num_ele * n_periods + 1, dtype='complex_')
        f20001s = np.zeros(num_ele + 1, dtype='complex_')
        f00201s = np.zeros(num_ele + 1, dtype='complex_')
        f10002s = np.zeros(num_ele + 1, dtype='complex_')
        jj = complex(0, 1)
        h20001 = h00201 = h10002 = h11001 = h00111 =0
        for i in range(num_ele):
            ele = self.elements[i]
            if ele.k1 != 0:
                b2l = ele.k1 * ele.length
                ele2 = self.elements[i+1]
                betax = __betafoc(ele2.betax, ele.alphax, ele2.alphax, ele.k1, ele.length)
                betay = __betafoc(ele2.betay, ele.alphay, ele2.alphay, -ele.k1, ele.length)
                etax = __dispfoc(ele.etax, ele2.etax, ele.k1, ele.length)
                mux = (ele.psix + ele2.psix) / 2
                muy = (ele.psiy + ele2.psiy) / 2
                h11001 += b2l * betax / 4
                h00111 += -b2l * betay / 4
                h20001 += b2l * betax * (cos(2 * mux) + jj * sin(2 * mux)) / 8
                h00201 += -b2l * betay * (cos(2 * muy) + jj * sin(2 * muy)) / 8
                h10002 += b2l * etax * betax ** 0.5 * (cos(mux) + jj * sin(mux)) / 2
            elif ele.k2 != 0:
                # b3l = ele.k2 * ele.length / 2
                ele2 = self.elements[i+1]
                betax = __betadrift(ele.betax, ele2.betax, ele.alphax, ele.length)
                betay = __betadrift(ele.betay, ele2.betay, ele.alphay, ele.length)
                etax = (ele.etax + ele2.etax) / 2
                mux = (ele.psix + ele2.psix) / 2
                muy = (ele.psiy + ele2.psiy) / 2
                b3letax = ele.k2 * ele.length * etax / 2
                h11001 += -b3letax * betax / 2
                h00111 += b3letax * betay / 2
                h20001 += -b3letax * betax * (cos(2 * mux) + jj * sin(2 * mux)) / 4
                h00201 += b3letax * betay * (cos(2 * muy) + jj * sin(2 * muy)) / 4
                h10002 += -b3letax * etax * (betax ** 0.5) * (cos(mux) + jj * sin(mux)) / 2
            h11001s[i+1] = h11001
            h00111s[i+1] = h00111
            h20001s[i+1] = h20001
            h00201s[i+1] = h00201
            h10002s[i+1] = h10002
        
        phix = self.elements[num_ele].psix
        phiy = self.elements[num_ele].psiy
        f20001 = h20001 / (1 - cos(2 * phix) - jj * sin(2 * phix))
        f00201 = h00201 / (1 - cos(2 * phiy) - jj * sin(2 * phiy))
        f10002 = h10002 / (1 - cos(phix) - jj * sin(phix))
        f20001s = h20001s[:num_ele+1] - f20001
        f00201s = h00201s[:num_ele+1] - f00201
        f10002s = h10002s[:num_ele+1] - f10002
        if n_periods > 1:
            q20001 = (cos(2 * phix) + jj * sin(2 * phix))
            q00201 = (cos(2 * phiy) + jj * sin(2 * phiy))
            q10002 = (cos(phix) + jj * sin(phix))
            for j in range(n_periods-1):
                i = j + 1
                h11001s[i * num_ele+1: (i + 1) * num_ele+1] = h11001s[1:num_ele+1] + h11001 * i
                h00111s[i * num_ele+1: (i + 1) * num_ele+1] = h00111s[1:num_ele+1] + h00111 * i
                h20001s[i * num_ele+1: (i + 1) * num_ele+1] = f20001 + (h20001s[1:num_ele+1] - f20001) * q20001 ** i
                h00201s[i * num_ele+1: (i + 1) * num_ele+1] = f00201 + (h00201s[1:num_ele+1] - f00201) * q00201 ** i
                h10002s[i * num_ele+1: (i + 1) * num_ele+1] = f10002 + (h10002s[1:num_ele+1] - f10002) * q10002 ** i
            h20001 = f20001 * (1 - q20001 ** n_periods)
            h00201 = f00201 * (1 - q00201 ** n_periods)
            h10002 = f10002 * (1 - q10002 ** n_periods)
            h11001 = h11001 * n_periods
            h00111 = h00111 * n_periods
        one_turn = {'h11001': h11001, 'h00111': h00111, 'h20001':h20001, 'h00201': h00201, 'h10002': h10002}
        buildup = {'h11001': h11001s, 'h00111': h00111s, 'h20001': h20001s, 'h00201': h00201s, 'h10002': h10002s}
        natural = {'f20001': np.abs(f20001s), 'f00201': np.abs(f00201s), 'f10002': np.abs(f10002s)}
        if verbose:
            for k, v in one_turn.items():
                print(f'{k}: {v:.2f}')
        return one_turn, buildup, natural

    def track_chromaticity(self, order=2, verbose=True, delta=None, matrix_precision=1e-9, resdl_limit=1e-12):
        """track_chromaticity(self, order=2, verbose=True, delta=None, matrix_precision=1e-9, resdl_limit=1e-12)
        compute chromaticity with the tunes of 4d off-momentum closed orbit.
            
            try to reset the value of delta, precision and resdl_limit if the result is wrong.
        you can call track_4d_closed_orbit() function to see the magnitude of the closed orbit, and the matrix_precision should be much smaller than it.

        Args:
            order: int 2 or 3.
            verbose: True.
            delta: the momentum deviation.
            matrix_precision: the small deviation to calculate transfer matrix by tracking.
            resdl_limit: the limit to judge if the orbit is closed.

        Returns:
            a dictionary of chromaticities.
             {'xi1x': float,
              'xi1y': float,
              'xi2x': float,
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
                resdl = sum(dco ** 2) ** 0.5
                iter_times += 1
            cos_mu = (mv[0, 0] + mv[1, 1]) / 2
            assert fabs(cos_mu) < 1, "can not find period solution, UNSTABLE!!!"
            nux = acos(cos_mu) * np.sign(mv[0, 1]) / 2 / pi
            cos_mu = (mv[2, 2] + mv[3, 3]) / 2
            assert fabs(cos_mu) < 1, "can not find period solution, UNSTABLE!!!"
            nuy = acos(cos_mu) * np.sign(mv[2, 3]) / 2 / pi
            return nux - np.floor(nux), nuy - np.floor(nuy)
        if delta is None:
            if order > 2:
                delta = 1e-2
            else:
                delta = 1e-5
        try:
            nux0, nuy0 = closed_orbit_tune(0.0)
            nux1, nuy1 = closed_orbit_tune(delta)
            nux_1, nuy_1 = closed_orbit_tune(-delta)
            if order > 2:
                nux2, nuy2 = closed_orbit_tune(2 * delta)
                nux_2, nuy_2 = closed_orbit_tune(-2 * delta)
        except Exception as e:
            raise Unstable('can not find off-momentum closed orbit, try smaller delta.')
        xi1x = self.n_periods * (nux1 - nux_1) / 2 / delta
        xi1y = self.n_periods * (nuy1 - nuy_1) / 2 / delta
        xi2x = self.n_periods * (nux1 + nux_1 - 2 * nux0) / 2 / delta ** 2
        xi2y = self.n_periods * (nuy1 + nuy_1 - 2 * nuy0) / 2 / delta ** 2
        if order > 2:
            xi3x = self.n_periods * (nux2 - 2 * nux1 + 2 * nux_1 - nux_2) / delta ** 3 / 12
            xi3y = self.n_periods * (nuy2 - 2 * nuy1 + 2 * nuy_1 - nuy_2) / delta ** 3 / 12
            xi4x = self.n_periods * (nux2 - 4 * nux1 + 6 * nux0 - 4 * nux_1 + nux_2) / delta ** 4 / 24
            xi4y = self.n_periods * (nuy2 - 4 * nuy1 + 6 * nuy0 - 4 * nuy_1 + nuy_2) / delta ** 4 / 24
            out = {'xi1x': xi1x, 'xi1y': xi1y, 'xi2x': xi2x, 'xi2y': xi2y, 'xi3x': xi3x, 'xi3y': xi3y, 'xi4x': xi4x, 'xi4y': xi4y}
        else:
            out = {'xi1x': xi1x, 'xi1y': xi1y, 'xi2x': xi2x, 'xi2y': xi2y}
        if verbose:
            print(f'chromaticities calculated using the tunes of 4d off-momentum closed orbit:')
            for k, v in out.items():
                print(f'    {k}: {v:.2f}')
        return out

    def output_matrix(self, file_name: str = u'matrix.txt'):
        """output_matrix(self, file_name: str = u'matrix.txt')
        output uncoupled matrix for each element and contained matrix"""

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
        newlattice = CSLattice(self.elements[:-1] * other)
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

