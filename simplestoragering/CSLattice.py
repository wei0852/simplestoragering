# -*- coding: utf-8 -*-
import numpy as np
from .components import LineEnd, Mark
from .RFCavity import RFCavity
from .globalvars import pi, c, Cq, Cr, RefParticle
from .HBend import HBend
from .Drift import Drift
from .Quadrupole import Quadrupole
from .Sextupole import Sextupole
from .Octupole import Octupole


class CSLattice(object):
    """lattice object, solve by Courant-Snyder method"""

    def __init__(self, ele_list: list, n_periods: int = 1, coupling: float = 0.00):
        self.length = 0
        self.n_periods = n_periods
        self.coup = coupling
        self.elements = []
        self.mark = {}
        self.rf_cavity = None
        self.angle = 0
        self.abs_angle = 0
        current_s = 0
        current_identifier = 0
        for oe in ele_list:
            ele = oe.copy()
            ele.s = current_s
            ele.identifier = current_identifier
            if isinstance(ele, Mark):
                if ele.name in self.mark:
                    self.mark[ele.name].append(ele)
                else:
                    self.mark[ele.name] = [ele]
            if isinstance(ele, RFCavity):
                self.rf_cavity = ele
            self.elements.append(ele)
            self.length = self.length + ele.length
            if isinstance(ele, HBend):
                self.angle += ele.theta
                self.abs_angle += abs(ele.theta)
            current_identifier += 1
            current_s = current_s + ele.length
        last_ele = LineEnd(s=self.length, identifier=current_identifier)
        self.elements.append(last_ele)
        self.length = self.length * n_periods
        self.angle = self.angle * 180 / pi * n_periods
        self.abs_angle = self.abs_angle * 180 / pi * n_periods
        # initialize twiss
        self.twiss_x0 = None
        self.twiss_y0 = None
        self.eta_x0 = None
        self.eta_y0 = None
        # solve twiss
        self.nux = None
        self.nuy = None
        self.nuz = None
        # self.__solve_along()
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
        # self.radiation_integrals()
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
        # self.global_parameters()

    def linear_optics(self, periodicity=True, line_mode=False):
        """calculate optical functions.
        n_periods: if True, the periodic solution will be the initial twiss data. Otherwise initial twiss should be set
                    by CSLattice.set_initial_twiss(betax, alphax, betay, alphay, etax, etaxp, etay, etayp)
        line_mode: if True, the storage ring parameters such as emittance and damping time are not calculated."""

        if periodicity:
            self._the_periodic_solution()
        else:
            if self.twiss_x0 is None or self.twiss_y0 is None or self.eta_x0 is None or self.eta_y0 is None:
                raise Exception('need initial twiss data. use set_initial_twiss() or linear_optics(n_periods=True)')
        self.__solve_along()
        self.U0 = Cr * RefParticle.energy ** 4 * self.I2 / (2 * pi)
        if not line_mode:
            np.seterr(all='raise')
            self.f_c = c * RefParticle.beta / self.length
            self.Jx = 1 - self.I4 / self.I2
            self.Jy = 1
            self.Js = 2 + self.I4 / self.I2
            self.sigma_e = RefParticle.gamma * np.sqrt(Cq * self.I3 / (self.Js * self.I2))
            self.emittance = Cq * RefParticle.gamma * RefParticle.gamma * self.I5 / (self.Jx * self.I2)
            self.tau0 = 2 * RefParticle.energy / self.U0 / self.f_c
            self.tau_s = self.tau0 / self.Js
            self.tau_x = self.tau0 / self.Jx
            self.tau_y = self.tau0 / self.Jy
            self.alpha = self.I1 * self.f_c / c  # momentum compaction factor
            self.emitt_x = self.emittance / (1 + self.coup)
            self.emitt_y = self.emittance * self.coup / (1 + self.coup)
            self.etap = self.alpha - 1 / RefParticle.gamma ** 2  # phase slip factor
            if self.rf_cavity is not None:
                self.rf_cavity.f_c = self.f_c
                self.nuz = (self.rf_cavity.voltage * self.rf_cavity.omega_rf * abs(
                    np.cos(self.rf_cavity.phase) * self.etap)
                            * self.length / RefParticle.energy / c) ** 0.5 / 2 / pi
                self.sigma_z = self.sigma_e * abs(self.etap) * self.length / (2 * pi * self.nuz)

    def _the_periodic_solution(self):
        """compute periodic solution and initialize twiss"""
        matrix = np.identity(6)
        for ele in self.elements:
            matrix = ele.matrix.dot(matrix)
        # x direction
        cos_mu = (matrix[0, 0] + matrix[1, 1]) / 2
        assert abs(cos_mu) < 1, "can not find period solution, UNSTABLE!!!"
        mu = np.arccos(cos_mu) * np.sign(matrix[0, 1])
        beta = matrix[0, 1] / np.sin(mu)
        alpha = (matrix[0, 0] - matrix[1, 1]) / (2 * np.sin(mu))
        gamma = - matrix[1, 0] / np.sin(mu)
        self.twiss_x0 = np.array([beta, alpha, gamma])
        # y direction
        cos_mu = (matrix[2, 2] + matrix[3, 3]) / 2
        assert abs(cos_mu) < 1, "can not find period solution, UNSTABLE!!!"
        mu = np.arccos(cos_mu) * np.sign(matrix[2, 3])
        beta = matrix[2, 3] / np.sin(mu)
        alpha = (matrix[2, 2] - matrix[3, 3]) / (2 * np.sin(mu))
        gamma = - matrix[3, 2] / np.sin(mu)
        self.twiss_y0 = np.array([beta, alpha, gamma])
        # solve eta
        sub_matrix_x = matrix[0:2, 0:2]
        matrix_etax = np.array([matrix[0, 5], matrix[1, 5]])
        self.eta_x0 = np.linalg.inv(np.identity(2) - sub_matrix_x).dot(matrix_etax)
        sub_matrix_y = matrix[2:4, 2:4]
        matrix_etay = np.array([matrix[2, 5], matrix[3, 5]])
        self.eta_y0 = np.linalg.inv(np.identity(2) - sub_matrix_y).dot(matrix_etay)

    def set_initial_twiss(self, betax, alphax, betay, alphay, etax, etaxp, etay, etayp):
        self.twiss_x0 = np.array([betax, alphax, (1 + alphax ** 2) / betax])
        self.twiss_y0 = np.array([betay, alphay, (1 + alphay ** 2) / betay])
        self.eta_x0 = np.array([etax, etaxp])
        self.eta_y0 = np.array([etay, etayp])

    def __solve_along(self):
        [betax, alphax, gammax] = self.twiss_x0
        [betay, alphay, gammay] = self.twiss_y0
        [etax, etaxp] = self.eta_x0
        [etay, etayp] = self.eta_y0
        psix = psiy = integral1 = integral2 = integral3 = integral4 = integral5 = 0
        natural_xi_x = sextupole_part_xi_x = natural_xi_y = sextupole_part_xi_y = 0
        ele = self.elements[0]
        ele.betax = betax
        ele.betay = betay
        ele.alphax = alphax
        ele.alphay = alphay
        ele.gammax = gammax
        ele.gammay = gammay
        ele.etax = etax
        ele.etay = etay
        ele.etaxp = etaxp
        ele.etayp = etayp
        ele.psix = psix
        ele.psiy = psiy
        for i in range(len(self.elements) - 1):
            [i1, i2, i3, i4, i5, xix, xiy], twiss = self.elements[i].linear_optics()
            self.elements[i + 1].betax = twiss[0]
            self.elements[i + 1].alphax = twiss[1]
            self.elements[i + 1].gammax = twiss[2]
            self.elements[i + 1].betay = twiss[3]
            self.elements[i + 1].alphay = twiss[4]
            self.elements[i + 1].gammay = twiss[5]
            self.elements[i + 1].etax = twiss[6]
            self.elements[i + 1].etaxp = twiss[7]
            self.elements[i + 1].etay = twiss[8]
            self.elements[i + 1].etayp = twiss[9]
            self.elements[i + 1].psix = twiss[10]
            self.elements[i + 1].psiy = twiss[11]
            integral1 += i1
            integral2 += i2
            integral3 += i3
            integral4 += i4
            integral5 += i5
            if self.elements[i].type == 'Sextupole':
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
        """slice elements of ring, the twiss data of each slice will be calculated.
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

    def s_dependent_nonlinear_terms(self):
        """compute resonance driving terms. return a dictionary, each value is a np.ndarray.
                nonlinear_terms = {'s':, 'h21000': , 'h30000': , 'h10110': , 'h10020': ,
                                   'h10200': , 'Qxx': , 'Qxy': , 'Qyy': ,
                                   'h31000': , 'h40000': , 'h20110': , 'h11200': ,
                                   'h20020': , 'h20200': , 'h00310': , 'h00400': }

                references:
                1. The Sextupole Scheme for the SLS: An Analytic Approach, SLS Note 09/97, Johan Bengtsson
                2. Explicit formulas for 2nd-order driving terms due to sextupoles and chromatic effects of quadrupoles, Chun-xi Wang"""

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
        qxx = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        qxy = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        qyy = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        f21000 = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        f30000 = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        f10110 = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        f10020 = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        f10200 = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        f31000 = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        f40000 = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        f20110 = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        f11200 = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        f20020 = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        f20200 = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        f00310 = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        f00400 = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        s = np.zeros(2 * (len(sext_index) + 1), dtype=np.float64)
        current_ind = 0
        for k in sext_index:  # 起点在直线段变化时，四阶项和ADTS项只关心相对相移，三阶项角度变化，绝对值不变，所以只计算六极铁处就够了
            s[current_ind * 2 + 1] = ele_list[k].s
            s[current_ind * 2 + 2] = ele_list[k].s
            Qxx = Qxy = Qyy = 0
            h21000 = h30000 = h10110 = h10020 = h10200 = 0
            h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
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
                    h21000 += - b3l_i * beta_xi ** 1.5 * np.exp(complex(0, mu_ix)) / 8
                    h30000 += - b3l_i * beta_xi ** 1.5 * np.exp(complex(0, 3 * mu_ix)) / 24
                    h10110 += b3l_i * beta_xi ** 0.5 * beta_yi * np.exp(complex(0, mu_ix)) / 4
                    h10020 += b3l_i * beta_xi ** 0.5 * beta_yi * np.exp(complex(0, mu_ix - 2 * mu_iy)) / 8
                    h10200 += b3l_i * beta_xi ** 0.5 * beta_yi * np.exp(complex(0, mu_ix + 2 * mu_iy)) / 8
                    for j in sext_index:
                        b3l_j = ele_list[j].k2 * ele_list[j].length / 2
                        b3l = b3l_j * b3l_i
                        if b3l != 0:
                            beta_xj = (ele_list[j].betax + ele_list[j + 1].betax) / 2
                            beta_yj = (ele_list[j].betay + ele_list[j + 1].betay) / 2
                            mu_jx = psix_list[j]
                            mu_ijx = abs(mu_ix - mu_jx)
                            mu_jy = psiy_list[j]
                            mu_ijy = abs(mu_iy - mu_jy)
                            beta_xij = beta_xj * beta_xi
                            mu_ij_x2y = mu_ijx + 2 * mu_ijy
                            mu_ij_x_2y = mu_ijx - 2 * mu_ijy
                            Qxx += b3l / (-16 * pi) * pow(beta_xi * beta_xj, 1.5) * (
                                    3 * np.cos(mu_ijx - pi_nux) / np.sin(pi_nux)
                                    + np.cos(3 * mu_ijx - 3 * pi_nux) / np.sin(3 * pi_nux))
                            Qxy += b3l / (8 * pi) * pow(beta_xij, 0.5) * beta_yj * (
                                    2 * beta_xi * np.cos(mu_ijx - pi_nux) / np.sin(pi_nux)
                                    - beta_yi * np.cos(mu_ij_x2y - pi_nux - 2 * pi_nuy) / np.sin(pi_nux + 2 * pi_nuy)
                                    + beta_yi * np.cos(mu_ij_x_2y - pi_nux + 2 * pi_nuy) / np.sin(pi_nux - 2 * pi_nuy))
                            Qyy += b3l / (-16 * pi) * pow(beta_xij, 0.5) * beta_yj * beta_yi * (
                                    4 * np.cos(mu_ijx - pi_nux) / np.sin(pi_nux)
                                    + np.cos(mu_ij_x2y - pi_nux - 2 * pi_nuy) / np.sin(pi_nux + 2 * pi_nuy)
                                    + np.cos(mu_ij_x_2y - pi_nux + 2 * pi_nuy) / np.sin(pi_nux - 2 * pi_nuy))
                            sign = - np.sign(mu_ix - mu_jx)
                            jj = complex(0, 1)
                            const = sign * jj * b3l
                            h31000 += const * beta_xij ** 1.5 * np.exp(complex(0, 3 * mu_ix - mu_jx)) / 32
                            h40000 += const * beta_xij ** 1.5 * np.exp(complex(0, 3 * mu_ix + mu_jx)) / 64
                            h20110 += const * beta_xij ** 0.5 * beta_yi * (
                                    beta_xj * (
                                    np.exp(complex(0, 3 * mu_jx - mu_ix)) - np.exp(complex(0, mu_ix + mu_jx))) +
                                    2 * beta_yj * np.exp(complex(0, mu_ix + mu_jx + 2 * mu_iy - 2 * mu_jy))) / 32
                            h11200 += const * beta_xij ** 0.5 * beta_yi * (
                                    beta_xj * (np.exp(complex(0, -mu_ix + mu_jx + 2 * mu_iy)) - np.exp(
                                complex(0, mu_ix - mu_jx + 2 * mu_iy))) +
                                    2 * beta_yj * (np.exp(complex(0, mu_ix - mu_jx + 2 * mu_iy)) + np.exp(
                                complex(0, - mu_ix + mu_jx + 2 * mu_iy)))) / 32
                            h20020 += const * beta_xij ** 0.5 * beta_yi * (
                                    beta_xj * np.exp(complex(0, -mu_ix + 3 * mu_jx - 2 * mu_iy)) -
                                    (beta_xj + 4 * beta_yj) * np.exp(complex(0, mu_ix + mu_jx - 2 * mu_iy))) / 64
                            h20200 += const * beta_xij ** 0.5 * beta_yi * (
                                    beta_xj * np.exp(complex(0, -mu_ix + 3 * mu_jx + 2 * mu_iy))
                                    - (beta_xj - 4 * beta_yj) * np.exp((complex(0, mu_ix + mu_jx + 2 * mu_iy)))) / 64
                            h00310 += const * beta_xij ** 0.5 * beta_yi * beta_yj * (
                                    np.exp(complex(0, mu_ix - mu_jx + 2 * mu_iy)) -
                                    np.exp(complex(0, -mu_ix + mu_jx + 2 * mu_iy))) / 32
                            h00400 += const * beta_xij ** 0.5 * beta_yi * beta_yj * np.exp(
                                complex(0, mu_ix - mu_jx + 2 * mu_iy + 2 * mu_jy)) / 64
            f21000[current_ind * 2] = abs(h21000)
            f30000[current_ind * 2] = abs(h30000)
            f10110[current_ind * 2] = abs(h10110)
            f10200[current_ind * 2] = abs(h10200)
            f10020[current_ind * 2] = abs(h10020)
            qxx[current_ind * 2] = Qxx
            qxy[current_ind * 2] = Qxy
            qyy[current_ind * 2] = Qyy
            f31000[current_ind * 2] = abs(h31000)
            f40000[current_ind * 2] = abs(h40000)
            f00310[current_ind * 2] = abs(h00310)
            f20020[current_ind * 2] = abs(h20020)
            f20110[current_ind * 2] = abs(h20110)
            f00400[current_ind * 2] = abs(h00400)
            f20200[current_ind * 2] = abs(h20200)
            f11200[current_ind * 2] = abs(h11200)
            f21000[current_ind * 2 + 1] = abs(h21000)
            f30000[current_ind * 2 + 1] = abs(h30000)
            f10110[current_ind * 2 + 1] = abs(h10110)
            f10200[current_ind * 2 + 1] = abs(h10200)
            f10020[current_ind * 2 + 1] = abs(h10020)
            qxx[current_ind * 2 + 1] = Qxx
            qxy[current_ind * 2 + 1] = Qxy
            qyy[current_ind * 2 + 1] = Qyy
            f31000[current_ind * 2 + 1] = abs(h31000)
            f40000[current_ind * 2 + 1] = abs(h40000)
            f00310[current_ind * 2 + 1] = abs(h00310)
            f20020[current_ind * 2 + 1] = abs(h20020)
            f20110[current_ind * 2 + 1] = abs(h20110)
            f00400[current_ind * 2 + 1] = abs(h00400)
            f20200[current_ind * 2 + 1] = abs(h20200)
            f11200[current_ind * 2 + 1] = abs(h11200)
            current_ind += 1
        s[-1] = ele_list[-1].s
        nonlinear = {'s': s, 'h21000': f21000, 'h30000': f30000, 'h10110': f10110, 'h10020': f10020,
                     'h10200': f10200, 'Qxx': qxx, 'Qxy': qxy, 'Qyy': qyy,
                     'h31000': f31000, 'h40000': f40000, 'h20110': f20110, 'h11200': f11200,
                     'h20020': f20020, 'h20200': f20200, 'h00310': f00310, 'h00400': f00400}
        for k in nonlinear:
            if k != 's':
                nonlinear[k][-1] = nonlinear[k][0]
                nonlinear[k][-2] = nonlinear[k][0]
        return nonlinear

    def nonlinear_terms(self, periodicity: int = None, printout=True):
        """nonlinear_terms(self, printout=True)
        compute resonance driving terms.

        Return:
            NonlinearTerms:
               {'h21000': , 'h30000': , 'h10110': , 'h10020': ,
                'h10200': , 'h20001': , 'h00201': , 'h10002': ,
                'h11001': , 'h00111': , 'dQxx': , 'dQxy': , 'dQyy': ,
                'h31000': , 'h40000': , 'h20110': , 'h11200': ,
                'h20020': , 'h20200': , 'h00310': , 'h00400': ,
                'h22000': , 'h11110': , 'h00220': }
                .set_periods(n_periods)

        references:
        1. The Sextupole Scheme for the SLS: An Analytic Approach, SLS Note 09/97, Johan Bengtsson
        2. Explicit formulas for 2nd-order driving terms due to sextupoles and chromatic effects of quadrupoles, Chun-xi Wang
        3. Perspectives for future light source lattices incorporating yet uncommon magnets, S. C. Leemann and A. Streun"""

        ele_list = []
        quad_index = []
        sext_index = []
        oct_index = []
        periodicity = periodicity if periodicity is not None else self.n_periods
        current_ind = 0
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
            elif ele.k1:
                ele_list += ele.slice(ele.n_slices)
                for i in range(ele.n_slices):
                    quad_index.append(current_ind)
                    current_ind += 1
            else:
                ele_list.append(ele)
                current_ind += 1
        del current_ind
        dQxx = dQxy = dQyy = 0
        h20001 = h00201 = h10002 = h11001 = h00111 = 0
        h21000 = h30000 = h10110 = h10020 = h10200 = 0
        h31000s1 = h40000s1 = h20110s1 = h11200s1 = h20020s1 = h20200s1 = h00310s1 = h00400s1 = 0  # sext, same period
        h30000h12000 = 0  # h31000
        h30000h21000 = 0  # h40000
        h30000h01110 = h21000h10110 = h10200h10020 = 0  # h20110
        h10200h12000 = h21000h01200 = 0  # h11200
        h21000h10020 = h30000h01020 = h10110h10020 = 0  # h20020
        h30000h01200 = h10200h21000 = h10110h10200 = 0  # h20200
        h10200h01110 = h10110h01200 = 0  # h00310 & h11200
        h10200h01200 = 0  # h00400
        h31000o = h40000o = h20110o = h11200o = h20020o = h20200o = h00310o = h00400o = 0
        pi_nux = self.elements[-1].psix / 2
        pi_nuy = self.elements[-1].psiy / 2
        jj = complex(0, 1)
        for i in quad_index:
            b2l = ele_list[i].k1 * ele_list[i].length
            beta_x = (ele_list[i].betax + ele_list[i + 1].betax) / 2
            eta_x = (ele_list[i].etax + ele_list[i + 1].etax) / 2
            beta_y = (ele_list[i].betay + ele_list[i + 1].betay) / 2
            mu_x = (ele_list[i].psix + ele_list[i + 1].psix) / 2
            mu_y = (ele_list[i].psiy + ele_list[i + 1].psiy) / 2
            h11001 += b2l * beta_x / 4
            h00111 += -b2l * beta_y / 4
            h20001 += b2l * beta_x / 8 * np.exp(complex(0, 2 * mu_x))
            h00201 += -b2l * beta_y / 8 * np.exp(complex(0, 2 * mu_y))
            h10002 += b2l * beta_x ** 0.5 * eta_x / 2 * np.exp(complex(0, mu_x))
        for j in range(len(sext_index)):
            b3l_j = ele_list[sext_index[j]].k2 * ele_list[sext_index[j]].length / 2  # k2 = 2 * b3, k1 = b2
            beta_xj = (ele_list[sext_index[j]].betax + ele_list[sext_index[j] + 1].betax) / 2
            eta_xj = (ele_list[sext_index[j]].etax + ele_list[sext_index[j] + 1].etax) / 2
            beta_yj = (ele_list[sext_index[j]].betay + ele_list[sext_index[j] + 1].betay) / 2
            mu_jx = (ele_list[sext_index[j]].psix + ele_list[sext_index[j] + 1].psix) / 2
            mu_jy = (ele_list[sext_index[j]].psiy + ele_list[sext_index[j] + 1].psiy) / 2
            h11001 += -b3l_j * beta_xj * eta_xj / 2
            h00111 += b3l_j * beta_yj * eta_xj / 2
            h20001 += -b3l_j * beta_xj * eta_xj / 4 * np.exp(complex(0, 2 * mu_jx))
            h00201 += b3l_j * beta_yj * eta_xj / 4 * np.exp(complex(0, 2 * mu_jy))
            h10002 += -b3l_j * beta_xj ** 0.5 * eta_xj ** 2 / 2 * np.exp(complex(0, mu_jx))
            h21000j = - b3l_j * beta_xj ** 1.5 * np.exp(complex(0, mu_jx)) / 8
            h30000j = - b3l_j * beta_xj ** 1.5 * np.exp(complex(0, 3 * mu_jx)) / 24
            h10110j = b3l_j * beta_xj ** 0.5 * beta_yj * np.exp(complex(0, mu_jx)) / 4
            h10020j = b3l_j * beta_xj ** 0.5 * beta_yj * np.exp(complex(0, mu_jx - 2 * mu_jy)) / 8
            h10200j = b3l_j * beta_xj ** 0.5 * beta_yj * np.exp(complex(0, mu_jx + 2 * mu_jy)) / 8
            h12000j = np.conj(h21000j)
            h01110j = np.conj(h10110j)
            h01020j = np.conj(h10200j)
            h01200j = np.conj(h10020j)
            h21000 += h21000j
            h30000 += h30000j
            h10110 += h10110j
            h10020 += h10020j
            h10200 += h10200j
            h30000h12000 += h30000j * h12000j
            h30000h21000 += h30000j * h21000j
            h30000h01110 += h30000j * np.conj(h10110j)
            h21000h10110 += h21000j * h10110j
            h10200h10020 += h10200j * h10020j
            h10200h12000 += h10200j * np.conj(h21000j)
            h21000h01200 += h21000j * np.conj(h10020j)
            h10200h01110 += h10200j * np.conj(h10110j)
            h10110h01200 += h10110j * np.conj(h10020j)
            h21000h10020 += h21000j * h10020j
            h30000h01020 += h30000j * np.conj(h10200j)
            h10110h10020 += h10110j * h10020j
            h30000h01200 += h30000j * np.conj(h10020j)
            h10200h21000 += h10200j * h21000j
            h10110h10200 += h10110j * h10200j
            h10200h01200 += h10200j * np.conj(h10020j)
            dQxx += b3l_j ** 2 / (-16 * pi) * beta_xj ** 3 * (
                    3 * np.cos(pi_nux) / np.sin(pi_nux)
                    + np.cos(3 * pi_nux) / np.sin(3 * pi_nux))
            dQxy += b3l_j ** 2 / (8 * pi) * beta_xj * beta_yj * (
                    2 * beta_xj * np.cos(pi_nux) / np.sin(pi_nux)
                    - beta_yj * np.cos(- pi_nux - 2 * pi_nuy) / np.sin(pi_nux + 2 * pi_nuy)
                    + beta_yj * np.cos(- pi_nux + 2 * pi_nuy) / np.sin(pi_nux - 2 * pi_nuy))
            dQyy += b3l_j ** 2 / (-16 * pi) * beta_xj * beta_yj ** 2 * (
                    4 * np.cos(pi_nux) / np.sin(pi_nux)
                    + np.cos(pi_nux + 2 * pi_nuy) / np.sin(pi_nux + 2 * pi_nuy)
                    + np.cos(pi_nux - 2 * pi_nuy) / np.sin(pi_nux - 2 * pi_nuy))
            for i in range(j):
                b3l_i = ele_list[sext_index[i]].k2 * ele_list[sext_index[i]].length / 2
                b3l = b3l_i * b3l_j
                beta_xi = (ele_list[sext_index[i]].betax + ele_list[sext_index[i] + 1].betax) / 2
                beta_yi = (ele_list[sext_index[i]].betay + ele_list[sext_index[i] + 1].betay) / 2
                mu_ix = (ele_list[sext_index[i]].psix + ele_list[sext_index[i] + 1].psix) / 2
                mu_ijx = abs(mu_ix - mu_jx)
                mu_iy = (ele_list[sext_index[i]].psiy + ele_list[sext_index[i] + 1].psiy) / 2
                mu_ijy = abs(mu_iy - mu_jy)
                beta_xij = beta_xj * beta_xi
                mu_ij_x2y = mu_ijx + 2 * mu_ijy
                mu_ij_x_2y = mu_ijx - 2 * mu_ijy
                dQxx += 2 * b3l / (-16 * pi) * pow(beta_xi * beta_xj, 1.5) * (
                        3 * np.cos(mu_ijx - pi_nux) / np.sin(pi_nux)
                        + np.cos(3 * mu_ijx - 3 * pi_nux) / np.sin(3 * pi_nux))
                dQxy += 2 * b3l / (8 * pi) * pow(beta_xij, 0.5) * beta_yi * (
                        2 * beta_xj * np.cos(mu_ijx - pi_nux) / np.sin(pi_nux)
                        - beta_yj * np.cos(mu_ij_x2y - pi_nux - 2 * pi_nuy) / np.sin(pi_nux + 2 * pi_nuy)
                        + beta_yj * np.cos(mu_ij_x_2y - pi_nux + 2 * pi_nuy) / np.sin(pi_nux - 2 * pi_nuy))
                dQyy += 2 * b3l / (-16 * pi) * pow(beta_xij, 0.5) * beta_yj * beta_yi * (
                        4 * np.cos(mu_ijx - pi_nux) / np.sin(pi_nux)
                        + np.cos(mu_ij_x2y - pi_nux - 2 * pi_nuy) / np.sin(pi_nux + 2 * pi_nuy)
                        + np.cos(mu_ij_x_2y - pi_nux + 2 * pi_nuy) / np.sin(pi_nux - 2 * pi_nuy))
                h21000i = - b3l_i * beta_xi ** 1.5 * np.exp(complex(0, mu_ix)) / 8
                h30000i = - b3l_i * beta_xi ** 1.5 * np.exp(complex(0, 3 * mu_ix)) / 24
                h10110i = b3l_i * beta_xi ** 0.5 * beta_yi * np.exp(complex(0, mu_ix)) / 4
                h10020i = b3l_i * beta_xi ** 0.5 * beta_yi * np.exp(complex(0, mu_ix - 2 * mu_iy)) / 8
                h10200i = b3l_i * beta_xi ** 0.5 * beta_yi * np.exp(complex(0, mu_ix + 2 * mu_iy)) / 8
                h12000i = np.conj(h21000i)
                h01110i = np.conj(h10110i)
                h01020i = np.conj(h10200i)
                h01200i = np.conj(h10020i)
                h31000s1 += (h30000i * h12000j - h30000j * h12000i) * jj * 6
                h40000s1 += (h30000i * h21000j - h30000j * h21000i) * jj * 3

                h20110s1 += (h30000i * h01110j - h30000j * h01110i) * jj * 3
                h20110s1 += (h21000i * h10110j - h21000j * h10110i) * jj * (-1)
                h20110s1 += (h10200i * h10020j - h10200j * h10020i) * jj * 4
                h11200s1 += (h10200i * np.conj(h21000j) - h10200j * np.conj(h21000i)) * jj * 2
                h11200s1 += (h21000i * np.conj(h10020j) - h21000j * np.conj(h10020i)) * jj * 2
                h11200s1 += (h10200i * np.conj(h10110j) - h10200j * np.conj(h10110i)) * jj * 2
                h11200s1 += (h10110i * np.conj(h10020j) - h10110j * np.conj(h10020i)) * jj * (-2)
                h20020s1 += (h21000i * h10020j - h21000j * h10020i) * jj * (-1)
                h20020s1 += (h30000i * np.conj(h10200j) - h30000j * np.conj(h10200i)) * jj * 3
                h20020s1 += (h10110i * h10020j - h10110j * h10020i) * jj * 2
                h20200s1 += (h30000i * np.conj(h10020j) - h30000j * np.conj(h10020i)) * jj * 3
                h20200s1 += (h10200i * h21000j - h10200j * h21000i) * jj * 1
                h20200s1 += (h10110i * h10200j - h10110j * h10200i) * jj * (-2)
                h00310s1 += (h10200i * np.conj(h10110j) - h10200j * np.conj(h10110i)) * jj * 1
                h00310s1 += (h10110i * np.conj(h10020j) - h10110j * np.conj(h10020i)) * jj * 1
                h00400s1 += (h10200i * h01200j - h10200j * h01200i) * jj * 1

                h30000h12000 += (h30000i * h12000j + h30000j * h12000i)
                h30000h21000 += (h30000i * h21000j + h30000j * h21000i)
                h30000h01110 += (h30000i * np.conj(h10110j) + h30000j * np.conj(h10110i))
                h21000h10110 += (h21000i * h10110j + h21000j * h10110i)
                h10200h10020 += (h10200i * h10020j + h10200j * h10020i)
                h10200h12000 += (h10200i * np.conj(h21000j) + h10200j * np.conj(h21000i))
                h21000h01200 += (h21000i * np.conj(h10020j) + h21000j * np.conj(h10020i))
                h10200h01110 += (h10200i * np.conj(h10110j) + h10200j * np.conj(h10110i))
                h10110h01200 += (h10110i * np.conj(h10020j) + h10110j * np.conj(h10020i))
                h21000h10020 += (h21000i * h10020j + h21000j * h10020i)
                h30000h01020 += (h30000i * np.conj(h10200j) + h30000j * np.conj(h10200i))
                h10110h10020 += (h10110i * h10020j + h10110j * h10020i)
                h30000h01200 += (h30000i * np.conj(h10020j) + h30000j * np.conj(h10020i))
                h10200h21000 += (h10200i * h21000j + h10200j * h21000i)
                h10110h10200 += (h10110i * h10200j + h10110j * h10200i)
                h10200h01200 += (h10200i * np.conj(h10020j) + h10200j * np.conj(h10020i))
        for i in oct_index:
            b4l = ele_list[i].k3 * ele_list[i].length / 6
            beta_x = (ele_list[i].betax + ele_list[i + 1].betax) / 2
            beta_y = (ele_list[i].betay + ele_list[i + 1].betay) / 2
            mu_x = (ele_list[i].psix + ele_list[i + 1].psix) / 2
            mu_y = (ele_list[i].psiy + ele_list[i + 1].psiy) / 2
            dQxx += 3 * b4l * beta_x ** 2 / 8 / pi
            dQxy -= 3 * b4l * beta_x * beta_y / (4 * pi)
            dQyy += 3 * b4l * beta_y ** 2 / 8 / pi
            h31000o += -b4l * beta_x ** 2 * np.exp(complex(0, 2 * mu_x)) / 16
            h40000o += -b4l * beta_x ** 2 * np.exp(complex(0, 4 * mu_x)) / 64
            h20110o += 3 * b4l * beta_x * beta_y * np.exp(complex(0, 2 * mu_x)) / 16
            h11200o += 3 * b4l * beta_x * beta_y * np.exp(complex(0, 2 * mu_y)) / 16
            h20020o += 3 * b4l * beta_x * beta_y * np.exp(complex(0, 2 * mu_x - 2 * mu_y)) / 32
            h20200o += 3 * b4l * beta_x * beta_y * np.exp(complex(0, 2 * mu_x + 2 * mu_y)) / 32
            h00310o += -b4l * beta_y ** 2 * np.exp(complex(0, 2 * mu_y)) / 16
            h00400o += -b4l * beta_y ** 2 * np.exp(complex(0, 4 * mu_y)) / 64

        nonlinear_terms = NonlinearTerm(periodicity, 2 * pi_nux, 2 * pi_nuy, h21000, h30000, h10110, h10020, h10200,
                                        h11001, h00111, h20001, h00201, h10002, h31000s1, h40000s1, h20110s1, h11200s1,
                                        h20020s1, h20200s1, h00310s1, h00400s1, h30000h12000, h30000h21000,
                                        h30000h01110, h21000h10110, h10200h10020, h10200h12000, h21000h01200,
                                        h10200h01110, h10110h01200, h21000h10020, h30000h01020, h10110h10020,
                                        h30000h01200, h10200h21000, h10110h10200, h10200h01200, h31000o, h40000o,
                                        h20110o, h11200o, h20020o, h20200o, h00310o, h00400o, dQxx, dQxy, dQyy)
        if printout:
            print(nonlinear_terms)
        return nonlinear_terms

    def nonlinear_terms_along_ring(self):
        """different from s_dependent_nonlinear_terms().
        compute resonance driving terms. return a dictionary, each value is a np.ndarray.
                nonlinear_terms = {'s':, 'h21000': , 'h30000': , 'h10110': , 'h10020': ,
                                   'h10200': , 'Qxx': , 'Qxy': , 'Qyy': ,
                                   'h31000': , 'h40000': , 'h20110': , 'h11200': ,
                                   'h20020': , 'h20200': , 'h00310': , 'h00400': }

        references:
        1. The Sextupole Scheme for the SLS: An Analytic Approach, SLS Note 09/97, Johan Bengtsson
        2. Explicit formulas for 2nd-order driving terms due to sextupoles and chromatic effects of quadrupoles, Chun-xi Wang"""

        ele_list = []
        sext_index = []
        quad_index = []
        oct_index = []
        nonlinear_index = []
        sext_num = 0
        oct_num = 0
        quad_num = 0
        nonlinear_num = np.zeros(3, dtype=np.int)
        current_ind = 0
        for ele in self.elements:
            if isinstance(ele, Sextupole):
                ele_list += ele.slice(ele.n_slices)
                for i in range(ele.n_slices):
                    sext_index.append(current_ind)
                    nonlinear_index.append(current_ind)
                    current_ind += 1
                    sext_num += 1
                    nonlinear_num = np.vstack((nonlinear_num, (quad_num, sext_num, oct_num)))
            elif isinstance(ele, Octupole):
                ele_list += ele.slice(ele.n_slices)
                for i in range(ele.n_slices):
                    oct_index.append(current_ind)
                    nonlinear_index.append(current_ind)
                    current_ind += 1
                    oct_num += 1
                    nonlinear_num = np.vstack((nonlinear_num, (quad_num, sext_num, oct_num)))
            elif ele.k1:
                n_slices = 4
                ele_list += ele.slice(n_slices)
                for i in range(n_slices):
                    quad_index.append(current_ind)
                    nonlinear_index.append(current_ind)
                    current_ind += 1
                    quad_num += 1
                    nonlinear_num = np.vstack((nonlinear_num, (quad_num, sext_num, oct_num)))
            else:
                ele_list.append(ele)
                current_ind += 1
        nonlinear_num = np.delete(nonlinear_num, 0, 0)
        f20001 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f00201 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f10002 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f11001 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f00111 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f21000 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f30000 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f10110 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f10020 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f10200 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f31000 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f40000 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f20110 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f11200 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f20020 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f20200 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f00310 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f00400 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f22000 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f00220 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        f11110 = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))
        s = np.zeros(2 * (quad_num + sext_num + 1 + oct_num))

        hq20001 = np.zeros(quad_num, dtype='complex_')
        hq00201 = np.zeros(quad_num, dtype='complex_')
        hq10002 = np.zeros(quad_num, dtype='complex_')
        hq11001 = np.zeros(quad_num)
        hq00111 = np.zeros(quad_num)
        hs20001 = np.zeros(sext_num, dtype='complex_')
        hs00201 = np.zeros(sext_num, dtype='complex_')
        hs10002 = np.zeros(sext_num, dtype='complex_')
        hs11001 = np.zeros(sext_num)
        hs00111 = np.zeros(sext_num)

        hs21000 = np.zeros(sext_num, dtype='complex_')
        hs30000 = np.zeros(sext_num, dtype='complex_')
        hs10110 = np.zeros(sext_num, dtype='complex_')
        hs10020 = np.zeros(sext_num, dtype='complex_')
        hs10200 = np.zeros(sext_num, dtype='complex_')

        h31000 = np.zeros((sext_num, sext_num), dtype='complex_')
        h40000 = np.zeros((sext_num, sext_num), dtype='complex_')
        h20110 = np.zeros((sext_num, sext_num), dtype='complex_')
        h11200 = np.zeros((sext_num, sext_num), dtype='complex_')
        h20020 = np.zeros((sext_num, sext_num), dtype='complex_')
        h20200 = np.zeros((sext_num, sext_num), dtype='complex_')
        h00310 = np.zeros((sext_num, sext_num), dtype='complex_')
        h00400 = np.zeros((sext_num, sext_num), dtype='complex_')
        h22000 = np.zeros((sext_num, sext_num), dtype='complex_')
        h00220 = np.zeros((sext_num, sext_num), dtype='complex_')
        h11110 = np.zeros((sext_num, sext_num), dtype='complex_')

        ho22000 = np.zeros(oct_num)
        ho00220 = np.zeros(oct_num)
        ho11110 = np.zeros(oct_num)
        ho31000 = np.zeros(oct_num, dtype='complex_')
        ho40000 = np.zeros(oct_num, dtype='complex_')
        ho20110 = np.zeros(oct_num, dtype='complex_')
        ho11200 = np.zeros(oct_num, dtype='complex_')
        ho20020 = np.zeros(oct_num, dtype='complex_')
        ho20200 = np.zeros(oct_num, dtype='complex_')
        ho00310 = np.zeros(oct_num, dtype='complex_')
        ho00400 = np.zeros(oct_num, dtype='complex_')

        for i in range(quad_num):
            b2l = ele_list[quad_index[i]].k1 * ele_list[quad_index[i]].length
            beta_x = (ele_list[quad_index[i]].betax + ele_list[quad_index[i] + 1].betax) / 2
            eta_x = (ele_list[quad_index[i]].etax + ele_list[quad_index[i] + 1].etax) / 2
            beta_y = (ele_list[quad_index[i]].betay + ele_list[quad_index[i] + 1].betay) / 2
            mu_x = (ele_list[quad_index[i]].psix + ele_list[quad_index[i] + 1].psix) / 2
            mu_y = (ele_list[quad_index[i]].psiy + ele_list[quad_index[i] + 1].psiy) / 2
            hq11001[i] = b2l * beta_x / 4
            hq00111[i] = -b2l * beta_y / 4
            hq20001[i] = b2l * beta_x / 8 * np.exp(complex(0, 2 * mu_x))
            hq00201[i] = -b2l * beta_y / 8 * np.exp(complex(0, 2 * mu_y))
            hq10002[i] = b2l * beta_x ** 0.5 * eta_x / 2 * np.exp(complex(0, mu_x))

        for i in range(sext_num):
            b3l_i = ele_list[sext_index[i]].k2 * ele_list[sext_index[i]].length / 2  # k2 = 2 * b3, k1 = b2
            if b3l_i != 0:
                beta_xi = (ele_list[sext_index[i]].betax + ele_list[sext_index[i] + 1].betax) / 2
                eta_xi = (ele_list[sext_index[i]].etax + ele_list[sext_index[i] + 1].etax) / 2
                beta_yi = (ele_list[sext_index[i]].betay + ele_list[sext_index[i] + 1].betay) / 2
                mu_ix = (ele_list[sext_index[i]].psix + ele_list[sext_index[i] + 1].psix) / 2
                mu_iy = (ele_list[sext_index[i]].psiy + ele_list[sext_index[i] + 1].psiy) / 2
                hs11001[i] = -b3l_i * beta_xi * eta_xi / 2
                hs00111[i] = b3l_i * beta_yi * eta_xi / 2
                hs20001[i] = -b3l_i * beta_xi * eta_xi / 4 * np.exp(complex(0, 2 * mu_ix))
                hs00201[i] = b3l_i * beta_yi * eta_xi / 4 * np.exp(complex(0, 2 * mu_iy))
                hs10002[i] = -b3l_i * beta_xi ** 0.5 * eta_xi ** 2 / 2 * np.exp(complex(0, mu_ix))
                hs21000[i] = - b3l_i * beta_xi ** 1.5 * np.exp(complex(0, mu_ix)) / 8
                hs30000[i] = - b3l_i * beta_xi ** 1.5 * np.exp(complex(0, 3 * mu_ix)) / 24
                hs10110[i] = + b3l_i * beta_xi ** 0.5 * beta_yi * np.exp(complex(0, mu_ix)) / 4
                hs10020[i] = + b3l_i * beta_xi ** 0.5 * beta_yi * np.exp(complex(0, mu_ix - 2 * mu_iy)) / 8
                hs10200[i] = + b3l_i * beta_xi ** 0.5 * beta_yi * np.exp(complex(0, mu_ix + 2 * mu_iy)) / 8
                for j in range(sext_num):
                    b3l_j = ele_list[sext_index[j]].k2 * ele_list[sext_index[j]].length / 2
                    b3l = b3l_j * b3l_i
                    if b3l != 0:
                        beta_xj = (ele_list[sext_index[j]].betax + ele_list[sext_index[j] + 1].betax) / 2
                        beta_yj = (ele_list[sext_index[j]].betay + ele_list[sext_index[j] + 1].betay) / 2
                        mu_jx = (ele_list[sext_index[j]].psix + ele_list[sext_index[j] + 1].psix) / 2
                        mu_jy = (ele_list[sext_index[j]].psiy + ele_list[sext_index[j] + 1].psiy) / 2
                        beta_xij = beta_xj * beta_xi
                        sign = - np.sign(mu_ix - mu_jx)
                        jj = complex(0, 1)
                        const = sign * jj * b3l
                        h22000[i, j] = const * beta_xij ** 1.5 * (np.exp(complex(0, 3 * (mu_ix - mu_jx))) + 3 * np.exp(
                            complex(0, mu_ix - mu_jx))) / 64
                        h11110[i, j] = const * beta_xij ** 0.5 * beta_yi * (
                                beta_xj * (np.exp(complex(0, mu_jx - mu_ix)) - np.exp(complex(0, mu_ix - mu_jx))) +
                                beta_yj * (np.exp(complex(0, mu_ix - mu_jx + 2 * mu_iy - 2 * mu_jy)) +
                                           np.exp(complex(0, -mu_ix + mu_jx + 2 * mu_iy - 2 * mu_jy)))) / 16
                        h00220[i, j] = const * beta_xij ** 0.5 * beta_yi * beta_yj * (
                                np.exp(complex(0, mu_ix - mu_jx + 2 * mu_iy - 2 * mu_jy)) +
                                4 * np.exp(complex(0, mu_ix - mu_jx)) -
                                np.exp(complex(0, -mu_ix + mu_jx + 2 * mu_iy - 2 * mu_jy))) / 64
                        h31000[i, j] = const * beta_xij ** 1.5 * np.exp(complex(0, 3 * mu_ix - mu_jx)) / 32
                        h40000[i, j] = const * beta_xij ** 1.5 * np.exp(complex(0, 3 * mu_ix + mu_jx)) / 64
                        h20110[i, j] = const * beta_xij ** 0.5 * beta_yi * (
                                beta_xj * (np.exp(complex(0, 3 * mu_jx - mu_ix)) - np.exp(complex(0, mu_ix + mu_jx))) +
                                2 * beta_yj * np.exp(complex(0, mu_ix + mu_jx + 2 * mu_iy - 2 * mu_jy))) / 32
                        h11200[i, j] = const * beta_xij ** 0.5 * beta_yi * (
                                beta_xj * (np.exp(complex(0, -mu_ix + mu_jx + 2 * mu_iy)) - np.exp(
                            complex(0, mu_ix - mu_jx + 2 * mu_iy))) +
                                2 * beta_yj * (np.exp(complex(0, mu_ix - mu_jx + 2 * mu_iy)) + np.exp(
                            complex(0, - mu_ix + mu_jx + 2 * mu_iy)))) / 32
                        h20020[i, j] = const * beta_xij ** 0.5 * beta_yi * (
                                beta_xj * np.exp(complex(0, -mu_ix + 3 * mu_jx - 2 * mu_iy)) -
                                (beta_xj + 4 * beta_yj) * np.exp(complex(0, mu_ix + mu_jx - 2 * mu_iy))) / 64
                        h20200[i, j] = const * beta_xij ** 0.5 * beta_yi * (
                                beta_xj * np.exp(complex(0, -mu_ix + 3 * mu_jx + 2 * mu_iy))
                                - (beta_xj - 4 * beta_yj) * np.exp((complex(0, mu_ix + mu_jx + 2 * mu_iy)))) / 64
                        h00310[i, j] = const * beta_xij ** 0.5 * beta_yi * beta_yj * (
                                np.exp(complex(0, mu_ix - mu_jx + 2 * mu_iy)) -
                                np.exp(complex(0, -mu_ix + mu_jx + 2 * mu_iy))) / 32
                        h00400[i, j] = const * beta_xij ** 0.5 * beta_yi * beta_yj * np.exp(
                            complex(0, mu_ix - mu_jx + 2 * mu_iy + 2 * mu_jy)) / 64

        for i in range(oct_num):
            b4l = ele_list[oct_index[i]].k3 * ele_list[oct_index[i]].length / 6
            beta_x = (ele_list[oct_index[i]].betax + ele_list[oct_index[i] + 1].betax) / 2
            beta_y = (ele_list[oct_index[i]].betay + ele_list[oct_index[i] + 1].betay) / 2
            mu_x = (ele_list[oct_index[i]].psix + ele_list[oct_index[i] + 1].psix) / 2
            mu_y = (ele_list[oct_index[i]].psiy + ele_list[oct_index[i] + 1].psiy) / 2
            ho22000[i] = -3 * b4l * beta_x ** 2 / 32
            ho11110[i] = 3 * b4l * beta_x * beta_y / 8
            ho00220[i] = -3 * b4l * beta_y ** 2 / 32
            ho31000[i] = -b4l * beta_x ** 2 * np.exp(complex(0, 2 * mu_x)) / 16
            ho40000[i] = -b4l * beta_x ** 2 * np.exp(complex(0, 4 * mu_x)) / 64
            ho20110[i] = 3 * b4l * beta_x * beta_y * np.exp(complex(0, 2 * mu_x)) / 16
            ho11200[i] = 3 * b4l * beta_x * beta_y * np.exp(complex(0, 2 * mu_y)) / 16
            ho20020[i] = 3 * b4l * beta_x * beta_y * np.exp(complex(0, 2 * mu_x - 2 * mu_y)) / 32
            ho20200[i] = 3 * b4l * beta_x * beta_y * np.exp(complex(0, 2 * mu_x + 2 * mu_y)) / 32
            ho00310[i] = -b4l * beta_y ** 2 * np.exp(complex(0, 2 * mu_y)) / 16
            ho00400[i] = -b4l * beta_y ** 2 * np.exp(complex(0, 4 * mu_y)) / 64
        current_ind = 0
        for k in nonlinear_index:  # 四阶项和ADTS项只关心相对相移，三阶项角度变化，绝对值不变，所以只计算六极铁处就够了
            s[current_ind * 2 + 1] = ele_list[k].s
            s[current_ind * 2 + 2] = ele_list[k].s
            current_ind += 1
        s[-1] = ele_list[-1].s
        current_ind = 1
        for (quad_num, sext_num, oct_num) in nonlinear_num:
            hts11001 = 0
            hts00111 = 0
            hts20001 = 0
            hts00201 = 0
            hts10002 = 0
            hts21000 = 0
            hts30000 = 0
            hts10110 = 0
            hts10020 = 0
            hts10200 = 0
            hts31000 = 0
            hts40000 = 0
            hts00310 = 0
            hts20020 = 0
            hts20110 = 0
            hts00400 = 0
            hts20200 = 0
            hts11200 = 0
            hts22000 = 0
            hts00220 = 0
            hts11110 = 0
            for i in range(quad_num):
                hts20001 += hq20001[i]
                hts00201 += hq00201[i]
                hts10002 += hq10002[i]
                hts11001 += hq11001[i]
                hts00111 += hq00111[i]
            for i in range(sext_num):
                hts20001 += hs20001[i]
                hts00201 += hs00201[i]
                hts10002 += hs10002[i]
                hts11001 += hs11001[i]
                hts00111 += hs00111[i]
                hts21000 += hs21000[i]
                hts30000 += hs30000[i]
                hts10110 += hs10110[i]
                hts10020 += hs10020[i]
                hts10200 += hs10200[i]
                for j in range(sext_num):
                    hts22000 += h22000[i, j]
                    hts00220 += h00220[i, j]
                    hts11110 += h11110[i, j]
                    hts31000 += h31000[i, j]
                    hts40000 += h40000[i, j]
                    hts00310 += h00310[i, j]
                    hts20020 += h20020[i, j]
                    hts20110 += h20110[i, j]
                    hts00400 += h00400[i, j]
                    hts20200 += h20200[i, j]
                    hts11200 += h11200[i, j]
            for i in range(oct_num):
                hts22000 += ho22000[i]
                hts00220 += ho00220[i]
                hts11110 += ho11110[i]
                hts31000 += ho31000[i]
                hts40000 += ho40000[i]
                hts00310 += ho00310[i]
                hts20020 += ho20020[i]
                hts20110 += ho20110[i]
                hts00400 += ho00400[i]
                hts20200 += ho20200[i]
                hts11200 += ho11200[i]
            f20001[current_ind * 2] = abs(hts20001)
            f00201[current_ind * 2] = abs(hts00201)
            f10002[current_ind * 2] = abs(hts10002)
            f11001[current_ind * 2] = abs(hts11001)
            f00111[current_ind * 2] = abs(hts00111)
            f21000[current_ind * 2] = abs(hts21000)
            f30000[current_ind * 2] = abs(hts30000)
            f10110[current_ind * 2] = abs(hts10110)
            f10020[current_ind * 2] = abs(hts10020)
            f10200[current_ind * 2] = abs(hts10200)
            f20001[current_ind * 2 + 1] = f20001[current_ind * 2]
            f00201[current_ind * 2 + 1] = f00201[current_ind * 2]
            f10002[current_ind * 2 + 1] = f10002[current_ind * 2]
            f11001[current_ind * 2 + 1] = f11001[current_ind * 2]
            f00111[current_ind * 2 + 1] = f00111[current_ind * 2]
            f21000[current_ind * 2 + 1] = f21000[current_ind * 2]
            f30000[current_ind * 2 + 1] = f30000[current_ind * 2]
            f10110[current_ind * 2 + 1] = f10110[current_ind * 2]
            f10020[current_ind * 2 + 1] = f10020[current_ind * 2]
            f10200[current_ind * 2 + 1] = f10200[current_ind * 2]
            f31000[current_ind * 2] = abs(hts31000)
            f40000[current_ind * 2] = abs(hts40000)
            f00310[current_ind * 2] = abs(hts00310)
            f20020[current_ind * 2] = abs(hts20020)
            f20110[current_ind * 2] = abs(hts20110)
            f00400[current_ind * 2] = abs(hts00400)
            f20200[current_ind * 2] = abs(hts20200)
            f11200[current_ind * 2] = abs(hts11200)
            f22000[current_ind * 2] = abs(hts22000)
            f00220[current_ind * 2] = abs(hts00220)
            f11110[current_ind * 2] = abs(hts11110)
            f22000[current_ind * 2 + 1] = f22000[current_ind * 2]
            f00220[current_ind * 2 + 1] = f00220[current_ind * 2]
            f11110[current_ind * 2 + 1] = f11110[current_ind * 2]
            f31000[current_ind * 2 + 1] = f31000[current_ind * 2]
            f40000[current_ind * 2 + 1] = f40000[current_ind * 2]
            f00310[current_ind * 2 + 1] = f00310[current_ind * 2]
            f20020[current_ind * 2 + 1] = f20020[current_ind * 2]
            f20110[current_ind * 2 + 1] = f20110[current_ind * 2]
            f00400[current_ind * 2 + 1] = f00400[current_ind * 2]
            f20200[current_ind * 2 + 1] = f20200[current_ind * 2]
            f11200[current_ind * 2 + 1] = f11200[current_ind * 2]
            current_ind += 1
        nonlinear = {'s': s, 'h21000': f21000, 'h30000': f30000, 'h10110': f10110, 'h10020': f10020,
                     'h10200': f10200, 'h20001': f20001, 'h00201': f00201, 'h10002': f10002,
                     'h11001': f11001, 'h00111': f00111, 'h22000': f22000, 'h11110': f11110, 'h00220': f00220,
                     'h31000': f31000, 'h40000': f40000, 'h20110': f20110, 'h11200': f11200,
                     'h20020': f20020, 'h20200': f20200, 'h00310': f00310, 'h00400': f00400}
        return nonlinear

    def higher_order_chromaticity(self, delta=1e-3, matrix_precision=1e-9, resdl_limit=1e-16):
        """compute higher order chromaticity with the tunes of 4d off-momentum closed orbit.
         delta: the momentum deviation.
         matrix_precision: the small deviation to calculate transfer matrix by tracking.
         resdl_limit: the limit to judge if the orbit is closed.

                try to reset the value of delta, precision and resdl_limit if the result is wrong.
        you can call track_4d_closed_orbit() function to see the magnitude of the closed orbit, and the matrix_precision
        should be much smaller than it.
        return a dictionary
        cr = {'xi2x': float,
              'xi2y': float,
              'xi3x': float,
              'xi3y': float}

        """

        def closed_orbit_tune(deviation):
            xco = np.array([0, 0, 0, 0])
            matrix = np.zeros([4, 4])
            resdl = 1
            j = 1
            precision = matrix_precision
            while j <= 10 and resdl > resdl_limit:
                beam = np.eye(6, 7) * precision
                for i in range(7):
                    beam[:4, i] = beam[:4, i] + xco
                    beam[5, i] = beam[5, i] + deviation
                for ele in self.elements:
                    beam = ele.symplectic_track(beam)
                for i in range(4):
                    matrix[:, i] = (beam[:4, i] - beam[:4, 6]) / precision
                d = beam[:4, 6] - xco
                dco = np.linalg.inv(np.identity(4) - matrix).dot(d)
                xco = xco + dco
                resdl = dco.dot(dco.T)
                j += 1
            cos_mu = (matrix[0, 0] + matrix[1, 1]) / 2
            assert abs(cos_mu) < 1, "can not find period solution, UNSTABLE!!!"
            nux = np.arccos(cos_mu) * np.sign(matrix[0, 1]) / 2 / pi
            nuy = np.arccos((matrix[2, 2] + matrix[3, 3]) / 2) * np.sign(matrix[2, 3]) / 2 / pi
            return nux - np.floor(nux), nuy - np.floor(nuy)

        try:
            nux3, nuy3 = closed_orbit_tune(3 * delta)
            nux1, nuy1 = closed_orbit_tune(delta)
            nux_1, nuy_1 = closed_orbit_tune(-delta)
            nux_3, nuy_3 = closed_orbit_tune(-3 * delta)
        except Exception as e:
            print(e)
            print('!!!!!!!\ncan not find off-momentum closed orbit, try smaller delta.\n '
                  '   !!!! you may need to change matrix_precision, too.')
            return {'xi2x': 1e9, 'xi2y': 1e9, 'xi3x': 1e9, 'xi3y': 1e9}
        xi2x = (nux1 + nux_1 - 2 * (self.nux - int(self.nux))) / 2 / delta ** 2
        xi2y = (nuy1 + nuy_1 - 2 * (self.nuy - int(self.nuy))) / 2 / delta ** 2
        xi3x = (nux3 - nux_3 + 3 * nux_1 - 3 * nux1) / (delta * 2) ** 3 / 6
        xi3y = (nuy3 - nuy_3 + 3 * nuy_1 - 3 * nuy1) / (delta * 2) ** 3 / 6
        print(f'xi2x: {xi2x:.2f}, xi2y: {xi2y:.2f}, xi3x: {xi3x:.2f}, xi3y: {xi3y:.2f}')
        return {'xi2x': xi2x, 'xi2y': xi2y, 'xi3x': xi3x, 'xi3y': xi3y}

    def output_matrix(self, file_name: str = 'matrix.txt'):
        """output uncoupled matrix for each element and contained matrix"""

        matrix = np.identity(6)
        file = open(file_name, 'w')
        location = 0.0
        for ele in self.elements:
            file.write(f'{ele.type()} {ele.name} at s={location},  {ele.magnets_data()}\n')
            location = location + ele.length
            file.write(str(ele.matrix) + '\n')
            file.write('contained matrix:\n')
            matrix = ele.matrix.dot(matrix)
            file.write(str(matrix))
            file.write('\n\n--------------------------\n\n')
        file.close()

    def output_twiss(self, file_name: str = 'twiss_data.txt'):
        """output s, ElementName, betax, alphax, psix, betay, alphay, psiy, etax, etaxp"""

        file1 = open(file_name, 'w')
        file1.write('& s, ElementName, betax, alphax, psix, betay, alphay, psiy, etax, etaxp\n')
        last_identifier = 123465
        for ele in self.elements:
            if ele.identifier != last_identifier:
                file1.write(f'{ele.s:.6e} {ele.name:10} {ele.betax:.6e}  {ele.alphax:.6e}  {ele.psix / 2 / pi:.6e}  '
                            f'{ele.betay:.6e}  {ele.alphay:.6e}  {ele.psiy / 2 / pi:.6e}  {ele.etax:.6e}  {ele.etaxp:.6e}\n')
                last_identifier = ele.identifier
        file1.close()

    def __add__(self, other):
        assert isinstance(other, CSLattice), 'can only add CSLattice.'
        newlattice = CSLattice(self.elements * self.n_periods + other.elements * other.n_periods)
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
        val += f'\n{str("energy ="):11} {RefParticle.energy:9.2e} MeV'
        # val += f'\n{str("gamma ="):11} {RefParticle.gamma:9.2f}'
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


class NonlinearTerm(object):

    def __init__(self, n_periods, phix, phiy,
                 h21000, h30000, h10110, h10020, h10200, h11001, h00111, h20001, h00201, h10002,
                 h31000s1, h40000s1, h20110s1, h11200s1, h20020s1, h20200s1, h00310s1, h00400s1,
                 h30000h12000, h30000h21000, h30000h01110, h21000h10110, h10200h10020, h10200h12000, h21000h01200,
                 h10200h01110, h10110h01200, h21000h10020, h30000h01020, h10110h10020, h30000h01200, h10200h21000,
                 h10110h10200, h10200h01200,
                 h31000o, h40000o, h20110o, h11200o, h20020o, h20200o, h00310o, h00400o, dQxx, dQxy, dQyy):
        self.n_periods = n_periods
        self.phix = phix
        self.phiy = phiy
        self.h21000 = h21000
        self.h30000 = h30000
        self.h10110 = h10110
        self.h10020 = h10020
        self.h10200 = h10200
        self.h11001 = h11001
        self.h00111 = h00111
        self.h20001 = h20001
        self.h00201 = h00201
        self.h10002 = h10002
        self.h31000s1 = h31000s1
        self.h40000s1 = h40000s1
        self.h20110s1 = h20110s1
        self.h11200s1 = h11200s1
        self.h20020s1 = h20020s1
        self.h20200s1 = h20200s1
        self.h00310s1 = h00310s1
        self.h00400s1 = h00400s1
        self.h30000h12000 = h30000h12000
        self.h30000h21000 = h30000h21000
        self.h30000h01110 = h30000h01110
        self.h21000h10110 = h21000h10110
        self.h10200h10020 = h10200h10020
        self.h10200h12000 = h10200h12000
        self.h21000h01200 = h21000h01200
        self.h10200h01110 = h10200h01110
        self.h10110h01200 = h10110h01200
        self.h21000h10020 = h21000h10020
        self.h30000h01020 = h30000h01020
        self.h10110h10020 = h10110h10020
        self.h30000h01200 = h30000h01200
        self.h10200h21000 = h10200h21000
        self.h10110h10200 = h10110h10200
        self.h10200h01200 = h10200h01200
        self.h31000o = h31000o
        self.h40000o = h40000o
        self.h20110o = h20110o
        self.h11200o = h11200o
        self.h20020o = h20020o
        self.h20200o = h20200o
        self.h00310o = h00310o
        self.h00400o = h00400o
        self.dQxx = dQxx
        self.dQxy = dQxy
        self.dQyy = dQyy
        self.terms = {}
        self.set_periods(n_periods)

    def set_periods(self, n_periods):
        self.n_periods = n_periods
        jj = complex(0, 1)
        q21000 = np.exp(complex(0, self.phix))
        q30000 = np.exp(complex(0, self.phix * 3))
        q10110 = np.exp(complex(0, self.phix))
        q10020 = np.exp(complex(0, self.phix - 2 * self.phiy))
        q10200 = np.exp(complex(0, self.phix + 2 * self.phiy))
        q20001 = np.exp(complex(0, 2 * self.phix))
        q00201 = np.exp(complex(0, 2 * self.phiy))
        q10002 = np.exp(complex(0, self.phix))
        h21000 = self.h21000 * (1 - q21000 ** n_periods) / (1 - q21000)
        h30000 = self.h30000 * (1 - q30000 ** n_periods) / (1 - q30000)
        h10110 = self.h10110 * (1 - q10110 ** n_periods) / (1 - q10110)
        h10020 = self.h10020 * (1 - q10020 ** n_periods) / (1 - q10020)
        h10200 = self.h10200 * (1 - q10200 ** n_periods) / (1 - q10200)
        h11001 = self.h11001 * n_periods
        h00111 = self.h00111 * n_periods
        h20001 = self.h20001 * (1 - q20001 ** n_periods) / (1 - q20001)
        h00201 = self.h00201 * (1 - q00201 ** n_periods) / (1 - q00201)
        h10002 = self.h10002 * (1 - q10002 ** n_periods) / (1 - q10002)
        q31000 = np.exp(complex(0, 2 * self.phix))
        q40000 = np.exp(complex(0, 4 * self.phix))
        q20110 = np.exp(complex(0, 2 * self.phix))
        q11200 = np.exp(complex(0, 2 * self.phiy))
        q20020 = np.exp(complex(0, 2 * self.phix - 2 * self.phiy))
        q20200 = np.exp(complex(0, 2 * self.phix + 2 * self.phiy))
        q00310 = np.exp(complex(0, 2 * self.phiy))
        q00400 = np.exp(complex(0, 4 * self.phiy))
        h31000o = (self.h31000o + self.h31000s1) * (1 - q31000 ** n_periods) / (1 - q31000)
        h40000o = (self.h40000o + self.h40000s1) * (1 - q40000 ** n_periods) / (1 - q40000)
        h20110o = (self.h20110o + self.h20110s1) * (1 - q20110 ** n_periods) / (1 - q20110)
        h11200o = (self.h11200o + self.h11200s1) * (1 - q11200 ** n_periods) / (1 - q11200)
        h20020o = (self.h20020o + self.h20020s1) * (1 - q20020 ** n_periods) / (1 - q20020)
        h20200o = (self.h20200o + self.h20200s1) * (1 - q20200 ** n_periods) / (1 - q20200)
        h00310o = (self.h00310o + self.h00310s1) * (1 - q00310 ** n_periods) / (1 - q00310)
        h00400o = (self.h00400o + self.h00400s1) * (1 - q00400 ** n_periods) / (1 - q00400)
        h30000h12000 = self.h30000h12000 * (self.__coeff_s2(3, 0, 0, 0, 1, 2, 0, 0) - self.__coeff_s2(1, 2, 0, 0, 3, 0, 0, 0))
        h30000h21000 = self.h30000h21000 * (self.__coeff_s2(3, 0, 0, 0, 2, 1, 0, 0) - self.__coeff_s2(2, 1, 0, 0, 3, 0, 0, 0))
        h30000h01110 = self.h30000h01110 * (self.__coeff_s2(3, 0, 0, 0, 0, 1, 1, 1) - self.__coeff_s2(0, 1, 1, 1, 3, 0, 0, 0))
        h21000h10110 = self.h21000h10110 * (self.__coeff_s2(2, 1, 0, 0, 1, 0, 1, 1) - self.__coeff_s2(1, 0, 1, 1, 2, 1, 0, 0))
        h10200h10020 = self.h10200h10020 * (self.__coeff_s2(1, 0, 2, 0, 1, 0, 0, 2) - self.__coeff_s2(1, 0, 0, 2, 1, 0, 2, 0))
        h10200h12000 = self.h10200h12000 * (self.__coeff_s2(1, 0, 2, 0, 1, 2, 0, 0) - self.__coeff_s2(1, 2, 0, 0, 1, 0, 2, 0))
        h21000h01200 = self.h21000h01200 * (self.__coeff_s2(2, 1, 0, 0, 0, 1, 2, 0) - self.__coeff_s2(0, 1, 2, 0, 2, 1, 0, 0))
        h10200h01110 = self.h10200h01110 * (self.__coeff_s2(1, 0, 2, 0, 0, 1, 1, 1) - self.__coeff_s2(0, 1, 1, 1, 1, 0, 2, 0))
        h10110h01200 = self.h10110h01200 * (self.__coeff_s2(1, 0, 1, 1, 0, 1, 2, 0) - self.__coeff_s2(0, 1, 2, 0, 1, 0, 1, 1))
        h21000h10020 = self.h21000h10020 * (self.__coeff_s2(2, 1, 0, 0, 1, 0, 0, 2) - self.__coeff_s2(1, 0, 0, 2, 2, 1, 0, 0))
        h30000h01020 = self.h30000h01020 * (self.__coeff_s2(3, 0, 0, 0, 0, 1, 0, 2) - self.__coeff_s2(0, 1, 0, 2, 3, 0, 0, 0))
        h10110h10020 = self.h10110h10020 * (self.__coeff_s2(1, 0, 1, 1, 1, 0, 0, 2) - self.__coeff_s2(1, 0, 0, 2, 1, 0, 1, 1))
        h30000h01200 = self.h30000h01200 * (self.__coeff_s2(3, 0, 0, 0, 0, 1, 2, 0) - self.__coeff_s2(0, 1, 2, 0, 3, 0, 0, 0))
        h10200h21000 = self.h10200h21000 * (self.__coeff_s2(1, 0, 2, 0, 2, 1, 0, 0) - self.__coeff_s2(2, 1, 0, 0, 1, 0, 2, 0))
        h10110h10200 = self.h10110h10200 * (self.__coeff_s2(1, 0, 1, 1, 1, 0, 2, 0) - self.__coeff_s2(1, 0, 2, 0, 1, 0, 1, 1))
        h10200h01200 = self.h10200h01200 * (self.__coeff_s2(1, 0, 2, 0, 0, 1, 2, 0) - self.__coeff_s2(0, 1, 2, 0, 1, 0, 2, 0))
        h31000s = jj * (6 * h30000h12000)
        h40000s = jj * (3 * h30000h21000)
        h20110s = jj * (3 * h30000h01110 - h21000h10110 + 4 * h10200h10020)
        h11200s = jj * (2 * h10200h12000 + 2 * h21000h01200 + 2 * h10200h01110 - 2 * h10110h01200)
        h20020s = jj * (-h21000h10020 + 3 * h30000h01020 + 2 * h10110h10020)
        h20200s = jj * (3 * h30000h01200 + h10200h21000 - 2 * h10110h10200)
        h00310s = jj * (h10200h01110 + h10110h01200)
        h00400s = jj * (h10200h01200)
        dQxx = self.dQxx * n_periods
        dQxy = self.dQxy * n_periods
        dQyy = self.dQyy * n_periods
        self.terms = {'h21000': abs(h21000), 'h30000': abs(h30000), 'h10110': abs(h10110), 'h10020': abs(h10020),
                      'h10200': abs(h10200), 'h20001': abs(h20001), 'h00201': abs(h00201), 'h10002': abs(h10002),
                      'h11001': abs(h11001), 'h00111': abs(h00111), 'dQxx': dQxx, 'dQxy': dQxy, 'dQyy': dQyy,
                      'h31000': abs(h31000o + h31000s), 'h40000': abs(h40000o + h40000s),
                      'h20110': abs(h20110o + h20110s), 'h11200': abs(h11200o + h11200s),
                      'h20020': abs(h20020o + h20020s), 'h20200': abs(h20200o + h20200s),
                      'h00310': abs(h00310o + h00310s), 'h00400': abs(h00400o + h00400s)}

    def n_periods_list(self, n_periods):
        jj = complex(0, 1)
        h21000 = np.zeros(n_periods)
        h30000 = np.zeros(n_periods)
        h10110 = np.zeros(n_periods)
        h10020 = np.zeros(n_periods)
        h10200 = np.zeros(n_periods)
        h11001 = np.zeros(n_periods)
        h00111 = np.zeros(n_periods)
        h20001 = np.zeros(n_periods)
        h00201 = np.zeros(n_periods)
        h10002 = np.zeros(n_periods)
        h31000 = np.zeros(n_periods)
        h40000 = np.zeros(n_periods)
        h20110 = np.zeros(n_periods)
        h11200 = np.zeros(n_periods)
        h20020 = np.zeros(n_periods)
        h20200 = np.zeros(n_periods)
        h00310 = np.zeros(n_periods)
        h00400 = np.zeros(n_periods)
        dQxx = np.zeros(n_periods)
        dQxy = np.zeros(n_periods)
        dQyy = np.zeros(n_periods)
        q21000 = np.exp(complex(0, self.phix))
        q30000 = np.exp(complex(0, self.phix * 3))
        q10110 = np.exp(complex(0, self.phix))
        q10020 = np.exp(complex(0, self.phix - 2 * self.phiy))
        q10200 = np.exp(complex(0, self.phix + 2 * self.phiy))
        q20001 = np.exp(complex(0, 2 * self.phix))
        q00201 = np.exp(complex(0, 2 * self.phiy))
        q10002 = np.exp(complex(0, self.phix))
        q31000 = np.exp(complex(0, 2 * self.phix))
        q40000 = np.exp(complex(0, 4 * self.phix))
        q20110 = np.exp(complex(0, 2 * self.phix))
        q11200 = np.exp(complex(0, 2 * self.phiy))
        q20020 = np.exp(complex(0, 2 * self.phix - 2 * self.phiy))
        q20200 = np.exp(complex(0, 2 * self.phix + 2 * self.phiy))
        q00310 = np.exp(complex(0, 2 * self.phiy))
        q00400 = np.exp(complex(0, 4 * self.phiy))
        for i in range(n_periods):
            h21000[i] = abs(self.h21000 * (1 - q21000 ** i) / (1 - q21000))
            h30000[i] = abs(self.h30000 * (1 - q30000 ** i) / (1 - q30000))
            h10110[i] = abs(self.h10110 * (1 - q10110 ** i) / (1 - q10110))
            h10020[i] = abs(self.h10020 * (1 - q10020 ** i) / (1 - q10020))
            h10200[i] = abs(self.h10200 * (1 - q10200 ** i) / (1 - q10200))
            h11001[i] = abs(self.h11001 * i)
            h00111[i] = abs(self.h00111 * i)
            h20001[i] = abs(self.h20001 * (1 - q20001 ** i) / (1 - q20001))
            h00201[i] = abs(self.h00201 * (1 - q00201 ** i) / (1 - q00201))
            h10002[i] = abs(self.h10002 * (1 - q10002 ** i) / (1 - q10002))
            h30000h12000 = self.h30000h12000 * (self.__coeff_s2(3, 0, 0, 0, 1, 2, 0, 0) - self.__coeff_s2(1, 2, 0, 0, 3, 0, 0, 0))
            h30000h21000 = self.h30000h21000 * (self.__coeff_s2(3, 0, 0, 0, 2, 1, 0, 0) - self.__coeff_s2(2, 1, 0, 0, 3, 0, 0, 0))
            h30000h01110 = self.h30000h01110 * (self.__coeff_s2(3, 0, 0, 0, 0, 1, 1, 1) - self.__coeff_s2(0, 1, 1, 1, 3, 0, 0, 0))
            h21000h10110 = self.h21000h10110 * (self.__coeff_s2(2, 1, 0, 0, 1, 0, 1, 1) - self.__coeff_s2(1, 0, 1, 1, 2, 1, 0, 0))
            h10200h10020 = self.h10200h10020 * (self.__coeff_s2(1, 0, 2, 0, 1, 0, 0, 2) - self.__coeff_s2(1, 0, 0, 2, 1, 0, 2, 0))
            h10200h12000 = self.h10200h12000 * (self.__coeff_s2(1, 0, 2, 0, 1, 2, 0, 0) - self.__coeff_s2(1, 2, 0, 0, 1, 0, 2, 0))
            h21000h01200 = self.h21000h01200 * (self.__coeff_s2(2, 1, 0, 0, 0, 1, 2, 0) - self.__coeff_s2(0, 1, 2, 0, 2, 1, 0, 0))
            h10200h01110 = self.h10200h01110 * (self.__coeff_s2(1, 0, 2, 0, 0, 1, 1, 1) - self.__coeff_s2(0, 1, 1, 1, 1, 0, 2, 0))
            h10110h01200 = self.h10110h01200 * (self.__coeff_s2(1, 0, 1, 1, 0, 1, 2, 0) - self.__coeff_s2(0, 1, 2, 0, 1, 0, 1, 1))
            h21000h10020 = self.h21000h10020 * (self.__coeff_s2(2, 1, 0, 0, 1, 0, 0, 2) - self.__coeff_s2(1, 0, 0, 2, 2, 1, 0, 0))
            h30000h01020 = self.h30000h01020 * (self.__coeff_s2(3, 0, 0, 0, 0, 1, 0, 2) - self.__coeff_s2(0, 1, 0, 2, 3, 0, 0, 0))
            h10110h10020 = self.h10110h10020 * (self.__coeff_s2(1, 0, 1, 1, 1, 0, 0, 2) - self.__coeff_s2(1, 0, 0, 2, 1, 0, 1, 1))
            h30000h01200 = self.h30000h01200 * (self.__coeff_s2(3, 0, 0, 0, 0, 1, 2, 0) - self.__coeff_s2(0, 1, 2, 0, 3, 0, 0, 0))
            h10200h21000 = self.h10200h21000 * (self.__coeff_s2(1, 0, 2, 0, 2, 1, 0, 0) - self.__coeff_s2(2, 1, 0, 0, 1, 0, 2, 0))
            h10110h10200 = self.h10110h10200 * (self.__coeff_s2(1, 0, 1, 1, 1, 0, 2, 0) - self.__coeff_s2(1, 0, 2, 0, 1, 0, 1, 1))
            h10200h01200 = self.h10200h01200 * (self.__coeff_s2(1, 0, 2, 0, 0, 1, 2, 0) - self.__coeff_s2(0, 1, 2, 0, 1, 0, 2, 0))
            h31000[i] = abs((self.h31000o + self.h31000s1) * (1 - q31000 ** i) / (1 - q31000) + jj * (6 * h30000h12000))
            h40000[i] = abs((self.h40000o + self.h40000s1) * (1 - q40000 ** i) / (1 - q40000) + jj * (3 * h30000h21000))
            h20110[i] = abs((self.h20110o + self.h20110s1) * (1 - q20110 ** i) / (1 - q20110) + jj * (3 * h30000h01110 - h21000h10110 + 4 * h10200h10020))
            h11200[i] = abs((self.h11200o + self.h11200s1) * (1 - q11200 ** i) / (1 - q11200) + jj * (2 * h10200h12000 + 2 * h21000h01200 + 2 * h10200h01110 - 2 * h10110h01200))
            h20020[i] = abs((self.h20020o + self.h20020s1) * (1 - q20020 ** i) / (1 - q20020) + jj * (-h21000h10020 + 3 * h30000h01020 + 2 * h10110h10020))
            h20200[i] = abs((self.h20200o + self.h20200s1) * (1 - q20200 ** i) / (1 - q20200) + jj * (3 * h30000h01200 + h10200h21000 - 2 * h10110h10200))
            h00310[i] = abs((self.h00310o + self.h00310s1) * (1 - q00310 ** i) / (1 - q00310) + jj * (h10200h01110 + h10110h01200))
            h00400[i] = abs((self.h00400o + self.h00400s1) * (1 - q00400 ** i) / (1 - q00400) + jj * (h10200h01200))
            dQxx[i] = self.dQxx * i
            dQxy[i] = self.dQxy * i
            dQyy[i] = self.dQyy * i
        return {'h21000': h21000, 'h30000': h30000, 'h10110': h10110, 'h10020': h10020,
                'h10200': h10200, 'h20001': h20001, 'h00201': h00201, 'h10002': h10002,
                'h11001': h11001, 'h00111': h00111, 'dQxx': dQxx, 'dQxy': dQxy, 'dQyy': dQyy,
                'h31000': h31000, 'h40000': h40000,
                'h20110': h20110, 'h11200': h11200,
                'h20020': h20020, 'h20200': h20200,
                'h00310': h00310, 'h00400': h00400}

    def __coeff_s2(self, j1, k1, l1, m1, j2, k2, l2, m2, n_periods=None):
        n_periods = n_periods if n_periods else self.n_periods
        pmu = (j1 - k1) * self.phix + (l1 - m1) * self.phiy  # m1
        qmu = (j2 - k2) * self.phix + (l2 - m2) * self.phiy  # m2
        return ((np.exp(complex(0, qmu)) * (1 - np.exp(complex(0, (n_periods - 1) * (pmu + qmu))))
                 / (1 - np.exp(complex(0, pmu + qmu)))
                 - np.exp(complex(0, n_periods * qmu)) * (1 - np.exp(complex(0, (n_periods - 1) * pmu)))
                 / (1 - np.exp(complex(0, pmu))))
                / (1 - np.exp(complex(0, qmu))))

    def __getitem__(self, item):
        return self.terms[item]

    def __str__(self):
        text = f'nonlinear terms: {self.n_periods:d} periods\n'
        for i, j in self.terms.items():
            text += f'    {str(i):7}: {j:.2f}\n'
        return text
