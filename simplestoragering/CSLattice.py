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
from .DrivingTerms import DrivingTerms


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
        for oe in ele_list:
            ele = oe.copy()
            ele.s = current_s
            if isinstance(ele, Mark):
                if ele.name in self.mark:
                    self.mark[ele.name].append(ele)
                else:
                    self.mark[ele.name] = [ele]
            if isinstance(ele, RFCavity):
                self.rf_cavity = ele
            if isinstance(ele, LineEnd):
                continue
            self.elements.append(ele)
            self.length = self.length + ele.length
            if isinstance(ele, HBend):
                self.angle += ele.theta
                self.abs_angle += abs(ele.theta)
            current_s = current_s + ele.length
        last_ele = LineEnd(s=self.length)
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
        """set_initial_twiss(betax, alphax, betay, alphay, etax, etaxp, etay, etayp)
        work if run CSLattice.linear_optics() with periodicity=False.
        """
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

    def s_dependent_driving_terms(self):
        """compute resonance driving terms of n-period map, and the starting position varies along one period.

        Return: a dictionary, each value is a np.ndarray.
                {'s':, 'f21000': , 'f30000': , 'f10110': , 'f10020': ,
                 'f10200': ,
                 'f31000': , 'f40000': , 'f20110': , 'f11200': ,
                 'f20020': , 'f20200': , 'f00310': , 'f00400': }

        references:
        1. The Sextupole Scheme for the SLS: An Analytic Approach, SLS Note 09/97, Johan Bengtsson
        2. Explicit formulas for 2nd-order driving terms due to sextupoles and chromatic effects of quadrupoles, Chun-xi Wang
        3. Perspectives for future light source lattices incorporating yet uncommon magnets, S. C. Leemann and A. Streun
        4. First simultaneous measurement of sextupolar and octupolar resonance driving terms in a circular accelerator from turn-by-turn beam position monitor data, Franchi, A., et al. (2014)
        """

        num_ele = len(self.elements)
        s = np.zeros(num_ele, dtype='float64')
        f21000 = np.zeros(num_ele, dtype='complex_')
        f30000 = np.zeros(num_ele, dtype='complex_')
        f10110 = np.zeros(num_ele, dtype='complex_')
        f10020 = np.zeros(num_ele, dtype='complex_')
        f10200 = np.zeros(num_ele, dtype='complex_')
        f31000 = np.zeros(num_ele, dtype='complex_')
        f40000 = np.zeros(num_ele, dtype='complex_')
        f20110 = np.zeros(num_ele, dtype='complex_')
        f11200 = np.zeros(num_ele, dtype='complex_')
        f20020 = np.zeros(num_ele, dtype='complex_')
        f20200 = np.zeros(num_ele, dtype='complex_')
        f00310 = np.zeros(num_ele, dtype='complex_')
        f00400 = np.zeros(num_ele, dtype='complex_')

        psix = np.zeros(num_ele)
        psiy = np.zeros(num_ele)
        for i, ele in enumerate(self.elements):
            psix[i] = ele.psix
            psiy[i] = ele.psiy

        jj = complex(0, 1)
        for i in range(num_ele-1):
            psix0 = psix[i]
            psiy0 = psiy[i]
            phix = psix[num_ele - 1]
            phiy = psiy[num_ele - 1]
            h21000 = h30000 = h10110 = h10020 = h10200 = 0
            h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
            for j, ele in enumerate(self.elements[i:]):
                ele.psix = psix[i + j] - psix0
                ele.psiy = psiy[i + j] - psiy0
                if isinstance(ele, Sextupole):  # 0        1      2       3       4       5       6       7
                    rdts = ele.driving_terms()  # h21000, h30000, h10110, h10020, h10200, h20001, h00201, h10002
                    h12000j = rdts[0].conjugate()
                    h01110j = rdts[2].conjugate()
                    h01200j = rdts[3].conjugate()
                    h01020j = rdts[4].conjugate()
                    h12000 = h21000.conjugate()
                    h01110 = h10110.conjugate()
                    h01200 = h10020.conjugate()
                    h01020 = h10200.conjugate()
                    h31000 += jj * 6 * (h30000 * h12000j - h12000 * rdts[1]) + rdts[11]
                    h40000 += jj * 3 * (h30000 * rdts[0] - h21000 * rdts[1]) + rdts[12]
                    h20110 += jj * ((h30000 * h01110j - h01110 * rdts[1]) * 3
                                    - (h21000 * rdts[2] - h10110 * rdts[0])
                                    + (h10200 * rdts[3] - h10020 * rdts[4]) * 4) + rdts[13]
                    h11200 += jj * ((h10200 * h12000j - h12000 * rdts[4]) * 2
                                    + (h21000 * h01200j - h01200 * rdts[0]) * 2
                                    + (h10200 * h01110j - h01110 * rdts[4]) * 2
                                    + (h10110 * h01200j - h01200 * rdts[2]) * (-2)) + rdts[14]
                    h20020 += jj * (-(h21000 * rdts[3] - h10020 * rdts[0])
                                    + (h30000 * h01020j - h01020 * rdts[1]) * 3
                                    + (h10110 * rdts[3] - h10020 * rdts[2]) * 2) + rdts[15]
                    h20200 += jj * ((h30000 * h01200j - h01200 * rdts[1]) * 3
                                    + (h10200 * rdts[0] - h21000 * rdts[4])
                                    + (h10110 * rdts[4] - h10200 * rdts[2]) * (-2)) + rdts[16]
                    h00310 += jj * ((h10200 * h01110j - h01110 * rdts[4])
                                    + (h10110 * h01200j - h01200 * rdts[2])) + rdts[17]
                    h00400 += jj * (h10200 * h01200j - h01200 * rdts[4]) + rdts[18]
                    h21000 = h21000 + rdts[0]
                    h30000 = h30000 + rdts[1]
                    h10110 = h10110 + rdts[2]
                    h10020 = h10020 + rdts[3]
                    h10200 = h10200 + rdts[4]
                elif isinstance(ele, Octupole):
                    rdts = ele.driving_terms()  # h22000, h11110, h00220, h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400
                    h31000 += rdts[3]
                    h40000 += rdts[4]
                    h20110 += rdts[5]
                    h11200 += rdts[6]
                    h20020 += rdts[7]
                    h20200 += rdts[8]
                    h00310 += rdts[9]
                    h00400 += rdts[10]
            for j, ele in enumerate(self.elements[:i]):
                ele.psix = psix[j] - psix0 + phix
                ele.psiy = psiy[j] - psiy0 + phiy
                if isinstance(ele, Sextupole):  # 0        1      2       3       4       5       6       7
                    rdts = ele.driving_terms()  # h21000, h30000, h10110, h10020, h10200, h20001, h00201, h10002
                    h12000j = rdts[0].conjugate()
                    h01110j = rdts[2].conjugate()
                    h01200j = rdts[3].conjugate()
                    h01020j = rdts[4].conjugate()
                    h12000 = h21000.conjugate()
                    h01110 = h10110.conjugate()
                    h01200 = h10020.conjugate()
                    h01020 = h10200.conjugate()

                    h31000 += jj * 6 * (h30000 * h12000j - h12000 * rdts[1]) + rdts[11]
                    h40000 += jj * 3 * (h30000 * rdts[0] - h21000 * rdts[1]) + rdts[12]
                    h20110 += jj * ((h30000 * h01110j - h01110 * rdts[1]) * 3
                                    - (h21000 * rdts[2] - h10110 * rdts[0])
                                    + (h10200 * rdts[3] - h10020 * rdts[4]) * 4) + rdts[13]
                    h11200 += jj * ((h10200 * h12000j - h12000 * rdts[4]) * 2
                                    + (h21000 * h01200j - h01200 * rdts[0]) * 2
                                    + (h10200 * h01110j - h01110 * rdts[4]) * 2
                                    + (h10110 * h01200j - h01200 * rdts[2]) * (-2)) + rdts[14]
                    h20020 += jj * (-(h21000 * rdts[3] - h10020 * rdts[0])
                                    + (h30000 * h01020j - h01020 * rdts[1]) * 3
                                    + (h10110 * rdts[3] - h10020 * rdts[2]) * 2) + rdts[15]
                    h20200 += jj * ((h30000 * h01200j - h01200 * rdts[1]) * 3
                                    + (h10200 * rdts[0] - h21000 * rdts[4])
                                    + (h10110 * rdts[4] - h10200 * rdts[2]) * (-2)) + rdts[16]
                    h00310 += jj * ((h10200 * h01110j - h01110 * rdts[4])
                                    + (h10110 * h01200j - h01200 * rdts[2])) + rdts[17]
                    h00400 += jj * (h10200 * h01200j - h01200 * rdts[4]) + rdts[18]
                    h21000 = h21000 + rdts[0]
                    h30000 = h30000 + rdts[1]
                    h10110 = h10110 + rdts[2]
                    h10020 = h10020 + rdts[3]
                    h10200 = h10200 + rdts[4]
                elif isinstance(ele, Octupole):
                    rdts = ele.driving_terms()  # h22000, h11110, h00220, h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400
                    h31000 += rdts[3]
                    h40000 += rdts[4]
                    h20110 += rdts[5]
                    h11200 += rdts[6]
                    h20020 += rdts[7]
                    h20200 += rdts[8]
                    h00310 += rdts[9]
                    h00400 += rdts[10]
            # calculate f_jklm
            h12000 = h21000.conjugate()
            h01110 = h10110.conjugate()
            h01200 = h10020.conjugate()
            h01020 = h10200.conjugate()
            phix = phix
            phiy = phiy
            s[i] = self.elements[i].s
            f21000[i] = h21000 / (1 - np.exp(complex(0, phix)))
            f30000[i] = h30000 / (1 - np.exp(complex(0, phix * 3)))
            f10110[i] = h10110 / (1 - np.exp(complex(0, phix)))
            f10020[i] = h10020 / (1 - np.exp(complex(0, phix - 2 * phiy)))
            f10200[i] = h10200 / (1 - np.exp(complex(0, phix + 2 * phiy)))
            f12000 = f21000[i].conjugate()
            f01200 = f10020[i].conjugate()
            f01110 = f10110[i].conjugate()
            h31000 += jj * 6 * (h30000 * f12000 - h12000 * f30000[i])
            # this part is actually not h30000, just to use one less variable name
            f31000[i] = h31000 / (1 - np.exp(complex(0, 2 * phix)))
            h40000 += jj * 3 * (h30000 * f21000[i] - h21000 * f30000[i])
            f40000[i] = h40000 / (1 - np.exp(complex(0, 4 * phix)))
            h20110 += jj * ((h30000 * f01110 - h01110 * f30000[i]) * 3
                            -(h21000 * f10110[i] - h10110 * f21000[i])
                            +(h10200 * f10020[i] - h10020 * f10200[i]) * 4)
            f20110[i] = h20110 / (1 - np.exp(complex(0, 2 * phix)))
            h11200 += jj * ((h10200 * f12000 - h12000 * f10200[i]) * 2
                            +(h21000 * f01200 - h01200 * f21000[i]) * 2
                            +(h10200 * f01110 - h01110 * f10200[i]) * 2
                            +(h10110 * f01200 - h01200 * f10110[i]) * (-2))
            f11200[i] = h11200 / (1 - np.exp(complex(0, 2 * phiy)))
            h20020 += jj * (-(h21000 * f10020[i] - h10020 * f21000[i])
                            +(h30000 * f10200[i].conjugate() - h01020 * f30000[i]) * 3
                            +(h10110 * f10020[i] - h10020 * f10110[i]) * 2)
            f20020[i] = h20020 / (1 - np.exp(complex(0, 2 * phix - 2 * phiy)))
            h20200 += jj * ((h30000 * f01200 - h01200 * f30000[i]) * 3
                            +(h10200 * f21000[i] - h21000 * f10200[i])
                            +(h10110 * f10200[i] - h10200 * f10110[i]) * (-2))
            f20200[i] = h20200 / (1 - np.exp(complex(0, 2 * phix + 2 * phiy)))
            h00310 += jj * ((h10200 * f01110 - h01110 * f10200[i])
                            +(h10110 * f01200 - h01200 * f10110[i]))
            f00310[i] = h00310 / (1 - np.exp(complex(0, 2 * phiy)))
            h00400 += jj * (h10200 * f01200 - h01200 * f10200[i])
            f00400[i] = h00400 / (1 - np.exp(complex(0, 4 * phiy)))
        for i, ele in enumerate(self.elements):
            ele.psix = psix[i]
            ele.psiy = psiy[i]
        s[-1] = self.elements[-1].s
        f21000[-1] = f21000[0]
        f30000[-1] = f30000[0]
        f10110[-1] = f10110[0]
        f10020[-1] = f10020[0]
        f10200[-1] = f10200[0]
        f31000[-1] = f31000[0]
        f40000[-1] = f40000[0]
        f20110[-1] = f20110[0]
        f11200[-1] = f11200[0]
        f20020[-1] = f20020[0]
        f20200[-1] = f20200[0]
        f00310[-1] = f00310[0]
        f00400[-1] = f00400[0]
        return {'s': s, 'f21000': f21000, 'f30000': f30000, 'f10110': f10110, 'f10020': f10020, 'f10200': f10200,
                'f31000': f31000, 'f40000': f40000, 'f20110': f20110, 'f11200': f11200, 'f20020': f20020,
                'f20200': f20200, 'f00310': f00310, 'f00400': f00400}

    def driving_terms(self, n_periods=None, printout: bool = True) -> DrivingTerms:
        """Calculate the 3rd- and 4th-order RDTs and their fluctuations.

        Return:
            DrivingTerms.

        references:
        1. The Sextupole Scheme for the SLS: An Analytic Approach, SLS Note 09/97, Johan Bengtsson
        2. Explicit formulas for 2nd-order driving terms due to sextupoles and chromatic effects of quadrupoles, Chun-xi Wang
        3. Perspectives for future light source lattices incorporating yet uncommon magnets, S. C. Leemann and A. Streun"""

        num_ele = len(self.elements)
        h21000s = np.zeros(num_ele, dtype='complex_')
        h30000s = np.zeros(num_ele, dtype='complex_')
        h10110s = np.zeros(num_ele, dtype='complex_')
        h10020s = np.zeros(num_ele, dtype='complex_')
        h10200s = np.zeros(num_ele, dtype='complex_')
        h20001s = np.zeros(num_ele, dtype='complex_')
        h00201s = np.zeros(num_ele, dtype='complex_')
        h10002s = np.zeros(num_ele, dtype='complex_')
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
        h21000 = h30000 = h10110 = h10020 = h10200 = h20001 = h00201 = h10002 = 0
        h22000 = h11110 = h00220 = h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
        geo_3rd_idx = 0
        geo_4th_idx = 0
        chr_3rd_idx = 0
        jj = complex(0, 1)
        for ele in self.elements:  # TODO: quad-sext
            if isinstance(ele, Sextupole):  # 0        1      2       3       4       5       6       7
                rdts = ele.driving_terms()  # h21000, h30000, h10110, h10020, h10200, h20001, h00201, h10002
                h12000j = rdts[0].conjugate()
                h01110j = rdts[2].conjugate()
                h01200j = rdts[3].conjugate()
                h01020j = rdts[4].conjugate()
                h12000 = h21000.conjugate()
                h01110 = h10110.conjugate()
                h01200 = h10020.conjugate()
                h01020 = h10200.conjugate()
                h22000 += jj * ((h21000 * h12000j - h12000 * rdts[0]) * 3
                                + (h30000 * rdts[1].conjugate() - h30000.conjugate() * rdts[1]) * 9) + rdts[8]
                h11110 += jj * ((h21000 * h01110j - h01110 * rdts[0]) * 2
                                - (h12000 * rdts[2] - h10110 * h12000j) * 2
                                - (h10020 * h01200j - h01200 * rdts[3]) * 4
                                + (h10200 * h01020j - h01020 * rdts[4]) * 4) + rdts[9]
                h00220 += jj * ((h10020 * h01200j - h01200 * rdts[3])
                                + (h10200 * h01020j - h01020 * rdts[4])
                                + (h10110 * h01110j - h01110 * rdts[2])) + rdts[10]
                h31000 += jj * 6 * (h30000 * h12000j - h12000 * rdts[1]) + rdts[11]
                h40000 += jj * 3 * (h30000 * rdts[0] - h21000 * rdts[1]) + rdts[12]
                h20110 += jj * ((h30000 * h01110j - h01110 * rdts[1]) * 3
                                - (h21000 * rdts[2] - h10110 * rdts[0])
                                + (h10200 * rdts[3] - h10020 * rdts[4]) * 4) + rdts[13]
                h11200 += jj * ((h10200 * h12000j - h12000 * rdts[4]) * 2
                                + (h21000 * h01200j - h01200 * rdts[0]) * 2
                                + (h10200 * h01110j - h01110 * rdts[4]) * 2
                                + (h10110 * h01200j - h01200 * rdts[2]) * (-2)) + rdts[14]
                h20020 += jj * (-(h21000 * rdts[3] - h10020 * rdts[0])
                                + (h30000 * h01020j - h01020 * rdts[1]) * 3
                                + (h10110 * rdts[3] - h10020 * rdts[2]) * 2) + rdts[15]
                h20200 += jj * ((h30000 * h01200j - h01200 * rdts[1]) * 3
                                + (h10200 * rdts[0] - h21000 * rdts[4])
                                + (h10110 * rdts[4] - h10200 * rdts[2]) * (-2)) + rdts[16]
                h00310 += jj * ((h10200 * h01110j - h01110 * rdts[4])
                                + (h10110 * h01200j - h01200 * rdts[2])) + rdts[17]
                h00400 += jj * (h10200 * h01200j - h01200 * rdts[4]) + rdts[18]
                h22000s[geo_4th_idx] = h22000
                h11110s[geo_4th_idx] = h11110
                h00220s[geo_4th_idx] = h00220
                h31000s[geo_4th_idx] = h31000
                h40000s[geo_4th_idx] = h40000
                h20110s[geo_4th_idx] = h20110
                h11200s[geo_4th_idx] = h11200
                h20020s[geo_4th_idx] = h20020
                h20200s[geo_4th_idx] = h20200
                h00310s[geo_4th_idx] = h00310
                h00400s[geo_4th_idx] = h00400
                geo_4th_idx += 1
                h21000 = h21000 + rdts[0]
                h30000 = h30000 + rdts[1]
                h10110 = h10110 + rdts[2]
                h10020 = h10020 + rdts[3]
                h10200 = h10200 + rdts[4]
                h21000s[geo_3rd_idx] = h21000
                h30000s[geo_3rd_idx] = h30000
                h10110s[geo_3rd_idx] = h10110
                h10020s[geo_3rd_idx] = h10020
                h10200s[geo_3rd_idx] = h10200
                geo_3rd_idx += 1
                h20001 += rdts[5]
                h00201 += rdts[6]
                h10002 += rdts[7]
                h20001s[chr_3rd_idx] = h20001
                h00201s[chr_3rd_idx] = h00201
                h10002s[chr_3rd_idx] = h10002
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
                h21000s[geo_3rd_idx] = h21000
                h30000s[geo_3rd_idx] = h30000
                h10110s[geo_3rd_idx] = h10110
                h10020s[geo_3rd_idx] = h10020
                h10200s[geo_3rd_idx] = h10200
                geo_3rd_idx += 1
                h22000s[geo_4th_idx] = h22000
                h11110s[geo_4th_idx] = h11110
                h00220s[geo_4th_idx] = h00220
                h31000s[geo_4th_idx] = h31000
                h40000s[geo_4th_idx] = h40000
                h20110s[geo_4th_idx] = h20110
                h11200s[geo_4th_idx] = h11200
                h20020s[geo_4th_idx] = h20020
                h20200s[geo_4th_idx] = h20200
                h00310s[geo_4th_idx] = h00310
                h00400s[geo_4th_idx] = h00400
                geo_4th_idx += 1
            elif ele.k1:
                rdts = ele.driving_terms()  # h20001, h00201, h10002
                h20001 += rdts[0]
                h00201 += rdts[1]
                h10002 += rdts[2]
                h20001s[chr_3rd_idx] = h20001
                h00201s[chr_3rd_idx] = h00201
                h10002s[chr_3rd_idx] = h10002
                chr_3rd_idx += 1

        phix = ele.psix
        phiy = ele.psiy
        R21000 = h21000 / (1 - np.exp(complex(0, phix)))
        R30000 = h30000 / (1 - np.exp(complex(0, phix * 3)))
        R10110 = h10110 / (1 - np.exp(complex(0, phix)))
        R10020 = h10020 / (1 - np.exp(complex(0, phix - 2 * phiy)))
        R10200 = h10200 / (1 - np.exp(complex(0, phix + 2 * phiy)))
        R20001 = h20001 / (1 - np.exp(complex(0, 2 * phix)))
        R00201 = h00201 / (1 - np.exp(complex(0, 2 * phiy)))
        R10002 = h10002 / (1 - np.exp(complex(0, phix)))
        R12000 = R21000.conjugate()
        R01110 = R10110.conjugate()
        R01200 = R10020.conjugate()
        R01020 = R10200.conjugate()
        h12000 = h21000.conjugate()
        h01110 = h10110.conjugate()
        h01200 = h10020.conjugate()
        h01020 = h10200.conjugate()

        h22000 = jj * ((h21000 * R12000 - h12000 * R21000) * 3
                       + (h30000 * R30000.conjugate() - h30000.conjugate() * R30000) * 9) + h22000
        h11110 = jj * ((h21000 * R01110 - h01110 * R21000) * 2
                       - (h12000 * R10110 - h10110 * R12000) * 2
                       - (h10020 * R01200 - h01200 * R10020) * 4
                       + (h10200 * R01020 - h01020 * R10200) * 4) + h11110
        h00220 = jj * ((h10020 * R01200 - h01200 * R10020)
                       + (h10200 * R01020 - h01020 * R10200)
                       + (h10110 * R01110 - h01110 * R10110)) + h00220
        h31000 = jj * 6 * (h30000 * R12000 - h12000 * R30000) + h31000
        h40000 = jj * 3 * (h30000 * R21000 - h21000 * R30000) + h40000
        h20110 = jj * ((h30000 * R01110 - h01110 * R30000) * 3
                       - (h21000 * R10110 - h10110 * R21000)
                       + (h10200 * R10020 - h10020 * R10200) * 4) + h20110
        h11200 = jj * ((h10200 * R12000 - h12000 * R10200) * 2
                       + (h21000 * R01200 - h01200 * R21000) * 2
                       + (h10200 * R01110 - h01110 * R10200) * 2
                       + (h10110 * R01200 - h01200 * R10110) * (-2)) + h11200
        h20020 = jj * (-(h21000 * R10020 - h10020 * R21000)
                       + (h30000 * R01020 - h01020 * R30000) * 3
                       + (h10110 * R10020 - h10020 * R10110) * 2) + h20020
        h20200 = jj * ((h30000 * R01200 - h01200 * R30000) * 3
                       + (h10200 * R21000 - h21000 * R10200)
                       + (h10110 * R10200 - h10200 * R10110) * (-2)) + h20200
        h00310 = jj * ((h10200 * R01110 - h01110 * R10200)
                       + (h10110 * R01200 - h01200 * R10110)) + h00310
        h00400 = jj * (h10200 * R01200 - h01200 * R10200) + h00400

        R31000 = h31000 / (1 - np.exp(complex(0, 2 * phix)))
        R40000 = h40000 / (1 - np.exp(complex(0, 4 * phix)))
        R20110 = h20110 / (1 - np.exp(complex(0, 2 * phix)))
        R11200 = h11200 / (1 - np.exp(complex(0, 2 * phiy)))
        R20020 = h20020 / (1 - np.exp(complex(0, 2 * phix - 2 * phiy)))
        R20200 = h20200 / (1 - np.exp(complex(0, 2 * phix + 2 * phiy)))
        R00310 = h00310 / (1 - np.exp(complex(0, 2 * phiy)))
        R00400 = h00400 / (1 - np.exp(complex(0, 4 * phiy)))

        n_periods = self.n_periods if n_periods is None else n_periods
        nonlinear_terms = DrivingTerms(n_periods, phix, phiy,
                                       R21000, R30000, R10110, R10020, R10200, R20001, R00201, R10002,
                                       h22000, h11110, h00220, R31000, R40000, R20110, R11200, R20020, R20200, R00310,
                                       R00400,
                                       h21000s[:geo_3rd_idx], h30000s[:geo_3rd_idx], h10110s[:geo_3rd_idx],
                                       h10020s[:geo_3rd_idx],
                                       h10200s[:geo_3rd_idx], h20001s[:chr_3rd_idx], h00201s[:chr_3rd_idx],
                                       h10002s[:chr_3rd_idx],
                                       h31000s[:geo_4th_idx], h40000s[:geo_4th_idx], h20110s[:geo_4th_idx],
                                       h11200s[:geo_4th_idx],
                                       h20020s[:geo_4th_idx], h20200s[:geo_4th_idx], h00310s[:geo_4th_idx],
                                       h00400s[:geo_4th_idx])
        if printout:
            print(nonlinear_terms)
        return nonlinear_terms

    def driving_terms_plot_data(self) -> dict:
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
        num_ele = len(self.elements) * 2
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
            if isinstance(ele, Sextupole):  # 0        1      2       3       4       5       6       7
                rdts = ele.driving_terms()  # h21000, h30000, h10110, h10020, h10200, h20001, h00201, h10002
                h12000j = rdts[0].conjugate()
                h01110j = rdts[2].conjugate()
                h01200j = rdts[3].conjugate()
                h01020j = rdts[4].conjugate()
                h12000 = h21000.conjugate()
                h01110 = h10110.conjugate()
                h01200 = h10020.conjugate()
                h01020 = h10200.conjugate()
                h22000 += jj * ((h21000 * h12000j - h12000 * rdts[0]) * 3
                                + (h30000 * rdts[1].conjugate() - h30000.conjugate() * rdts[1]) * 9) + rdts[8]
                h11110 += jj * ((h21000 * h01110j - h01110 * rdts[0]) * 2
                                - (h12000 * rdts[2] - h10110 * h12000j) * 2
                                - (h10020 * h01200j - h01200 * rdts[3]) * 4
                                + (h10200 * h01020j - h01020 * rdts[4]) * 4) + rdts[9]
                h00220 += jj * ((h10020 * h01200j - h01200 * rdts[3])
                                + (h10200 * h01020j - h01020 * rdts[4])
                                + (h10110 * h01110j - h01110 * rdts[2])) + rdts[10]
                h31000 += jj * 6 * (h30000 * h12000j - h12000 * rdts[1]) + rdts[11]
                h40000 += jj * 3 * (h30000 * rdts[0] - h21000 * rdts[1]) + rdts[12]
                h20110 += jj * ((h30000 * h01110j - h01110 * rdts[1]) * 3
                                - (h21000 * rdts[2] - h10110 * rdts[0])
                                + (h10200 * rdts[3] - h10020 * rdts[4]) * 4) + rdts[13]
                h11200 += jj * ((h10200 * h12000j - h12000 * rdts[4]) * 2
                                + (h21000 * h01200j - h01200 * rdts[0]) * 2
                                + (h10200 * h01110j - h01110 * rdts[4]) * 2
                                + (h10110 * h01200j - h01200 * rdts[2]) * (-2)) + rdts[14]
                h20020 += jj * (-(h21000 * rdts[3] - h10020 * rdts[0])
                                + (h30000 * h01020j - h01020 * rdts[1]) * 3
                                + (h10110 * rdts[3] - h10020 * rdts[2]) * 2) + rdts[15]
                h20200 += jj * ((h30000 * h01200j - h01200 * rdts[1]) * 3
                                + (h10200 * rdts[0] - h21000 * rdts[4])
                                + (h10110 * rdts[4] - h10200 * rdts[2]) * (-2)) + rdts[16]
                h00310 += jj * ((h10200 * h01110j - h01110 * rdts[4])
                                + (h10110 * h01200j - h01200 * rdts[2])) + rdts[17]
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
        RDTs_along_ring = {'s': s[:idx], 'h21000': f21000[:idx], 'h30000': f30000[:idx], 'h10110': f10110[:idx],
                           'h10020': f10020[:idx],
                           'h10200': f10200[:idx], 'h20001': f20001[:idx], 'h00201': f00201[:idx],
                           'h10002': f10002[:idx],
                           'h31000': f31000[:idx], 'h40000': f40000[:idx], 'h20110': f20110[:idx],
                           'h11200': f11200[:idx],
                           'h20020': f20020[:idx], 'h20200': f20200[:idx], 'h00310': f00310[:idx],
                           'h00400': f00400[:idx],
                           'h22000': f22000[:idx], 'h11110': f11110[:idx], 'h00220': f00220[:idx]
                           }
        return RDTs_along_ring

    def adts(self, n_periods=None, printout: bool = True) -> dict:
        """adts(self, n_periods=None, printout=True)
        compute ADTS terms.
        Return:
            {'dQxx': , 'dQxy': , 'dQyy':}

        references:
        1. The Sextupole Scheme for the SLS: An Analytic Approach, SLS Note 09/97, Johan Bengtsson"""

        n_periods = self.n_periods if n_periods is None else n_periods
        ele_list = []
        sext_index = []
        oct_index = []
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
            else:
                ele_list.append(ele)
                current_ind += 1
        Qxx = Qxy = Qyy = 0
        pi_nux = ele_list[current_ind - 1].psix / 2
        pi_nuy = ele_list[current_ind - 1].psiy / 2
        if np.sin(pi_nux) == 0 or np.sin(3 * pi_nux) == 0 or np.sin(pi_nux + 2 * pi_nuy) == 0 or np.sin(
                pi_nux - 2 * pi_nuy) == 0:
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
                        3 * np.cos(0 - pi_nux) / np.sin(pi_nux)
                        + np.cos(3 * 0 - 3 * pi_nux) / np.sin(3 * pi_nux))
                Qxy += b3l_i ** 2 / (8 * pi) * beta_xi * beta_yi * (
                        2 * beta_xi * np.cos(pi_nux) / np.sin(pi_nux)
                        - beta_yi * np.cos(pi_nux + 2 * pi_nuy) / np.sin(pi_nux + 2 * pi_nuy)
                        + beta_yi * np.cos(pi_nux - 2 * pi_nuy) / np.sin(pi_nux - 2 * pi_nuy))
                Qyy += b3l_i ** 2 / (-16 * pi) * beta_xi * beta_yi * beta_yi * (
                        4 * np.cos(pi_nux) / np.sin(pi_nux)
                        + np.cos(pi_nux + 2 * pi_nuy) / np.sin(pi_nux + 2 * pi_nuy)
                        + np.cos(pi_nux - 2 * pi_nuy) / np.sin(pi_nux - 2 * pi_nuy))
                for j in range(i):
                    b3l = ele_list[sext_index[j]].k2 * ele_list[sext_index[j]].length * b3l_i / 2
                    if b3l != 0:
                        beta_xj = (ele_list[sext_index[j]].betax + ele_list[sext_index[j] + 1].betax) / 2
                        beta_yj = (ele_list[sext_index[j]].betay + ele_list[sext_index[j] + 1].betay) / 2
                        mu_jx = (ele_list[sext_index[j]].psix + ele_list[sext_index[j] + 1].psix) / 2
                        mu_ijx = abs(mu_ix - mu_jx)
                        mu_jy = (ele_list[sext_index[j]].psiy + ele_list[sext_index[j] + 1].psiy) / 2
                        mu_ijy = abs(mu_iy - mu_jy)
                        beta_xij = beta_xj * beta_xi
                        Qxx += 2 * b3l / (-16 * pi) * pow(beta_xi * beta_xj, 1.5) * (
                                3 * np.cos(mu_ijx - pi_nux) / np.sin(pi_nux)
                                + np.cos(3 * mu_ijx - 3 * pi_nux) / np.sin(3 * pi_nux))
                        Qxy += 2 * b3l / (8 * pi) * pow(beta_xij, 0.5) * beta_yj * (
                                2 * beta_xi * np.cos(mu_ijx - pi_nux) / np.sin(pi_nux)
                                - beta_yi * np.cos(mu_ijx + 2 * mu_ijy - pi_nux - 2 * pi_nuy) / np.sin(
                            pi_nux + 2 * pi_nuy)
                                + beta_yi * np.cos(mu_ijx - 2 * mu_ijy - pi_nux + 2 * pi_nuy) / np.sin(
                            pi_nux - 2 * pi_nuy))
                        Qyy += 2 * b3l / (-16 * pi) * pow(beta_xij, 0.5) * beta_yj * beta_yi * (
                                4 * np.cos(mu_ijx - pi_nux) / np.sin(pi_nux)
                                + np.cos(mu_ijx + 2 * mu_ijy - pi_nux - 2 * pi_nuy) / np.sin(pi_nux + 2 * pi_nuy)
                                + np.cos(mu_ijx - 2 * mu_ijy - pi_nux + 2 * pi_nuy) / np.sin(pi_nux - 2 * pi_nuy))
        for i in oct_index:
            b4l = ele_list[i].k3 * ele_list[i].length / 6
            beta_xi = (ele_list[i].betax + ele_list[i + 1].betax) / 2
            beta_yi = (ele_list[i].betay + ele_list[i + 1].betay) / 2
            Qxx += 3 * b4l * beta_xi ** 2 / 8 / pi
            Qxy -= 3 * b4l * beta_xi * beta_yi / (4 * pi)
            Qyy += 3 * b4l * beta_yi ** 2 / 8 / pi

        nonlinear_terms = {'dQxx': Qxx * n_periods, 'dQxy': Qxy * n_periods, 'dQyy': Qyy * n_periods}
        if printout:
            print(f'ADTS terms, {n_periods} periods:')
            for k, b4l in nonlinear_terms.items():
                print(f'    {str(k):7}: {b4l:.2f}')
        return nonlinear_terms

    def another_method_driving_terms(self, printout=True):
        """compute resonance driving terms. return a dictionary
        nonlinear_terms = {'h21000': , 'h30000': , 'h10110': , 'h10020': ,
                           'h10200': , 'Qxx': , 'Qxy': , 'Qyy': ,
                           'h31000': , 'h40000': , 'h20110': , 'h11200': ,
                           'h20020': , 'h20200': , 'h00310': , 'h00400': }

        references:
        1. The Sextupole Scheme for the SLS: An Analytic Approach, SLS Note 09/97, Johan Bengtsson
        2. Explicit formulas for 2nd-order driving terms due to sextupoles and chromatic effects of quadrupoles, Chun-xi Wang
        3. Perspectives for future light source lattices incorporating yet uncommon magnets, S. C. Leemann and A. Streun"""

        ele_list = []
        quad_index = []
        sext_index = []
        oct_index = []
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
        Qxx = Qxy = Qyy = 0
        h20001 = h00201 = h10002 = h11001 = h00111 = 0
        h21000 = h30000 = h10110 = h10020 = h10200 = 0
        h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
        h22000 = h11110 = h00220 = 0
        pi_nux = self.elements[-1].psix / 2
        pi_nuy = self.elements[-1].psiy / 2
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
        for i in sext_index:
            b3l_i = ele_list[i].k2 * ele_list[i].length / 2  # k2 = 2 * b3, k1 = b2
            if b3l_i != 0:
                beta_xi = (ele_list[i].betax + ele_list[i + 1].betax) / 2
                eta_xi = (ele_list[i].etax + ele_list[i + 1].etax) / 2
                beta_yi = (ele_list[i].betay + ele_list[i + 1].betay) / 2
                mu_ix = (ele_list[i].psix + ele_list[i + 1].psix) / 2
                mu_iy = (ele_list[i].psiy + ele_list[i + 1].psiy) / 2
                h11001 += -b3l_i * beta_xi * eta_xi / 2
                h00111 += b3l_i * beta_yi * eta_xi / 2
                h20001 += -b3l_i * beta_xi * eta_xi / 4 * np.exp(complex(0, 2 * mu_ix))
                h00201 += b3l_i * beta_yi * eta_xi / 4 * np.exp(complex(0, 2 * mu_iy))
                h10002 += -b3l_i * beta_xi ** 0.5 * eta_xi ** 2 / 2 * np.exp(complex(0, mu_ix))
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
                        mu_jx = (ele_list[j].psix + ele_list[j + 1].psix) / 2
                        mu_ijx = abs(mu_ix - mu_jx)
                        mu_jy = (ele_list[j].psiy + ele_list[j + 1].psiy) / 2
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
                        h22000 += const * beta_xij ** 1.5 * (np.exp(complex(0, 3 * (mu_ix - mu_jx))) + 3 * np.exp(
                            complex(0, mu_ix - mu_jx))) / 64
                        h11110 += const * beta_xij ** 0.5 * beta_yi * (beta_xj * (
                                    np.exp(complex(0, mu_jx - mu_ix)) - np.exp(complex(0, mu_ix - mu_jx))) +
                                                                       beta_yj * (np.exp(
                                    complex(0, mu_ix - mu_jx + 2 * mu_iy - 2 * mu_jy)) +
                                                                                  np.exp(complex(0,
                                                                                                 -mu_ix + mu_jx + 2 * mu_iy - 2 * mu_jy)))) / 16
                        h00220 += const * beta_xij ** 0.5 * beta_yi * beta_yj * (
                                    np.exp(complex(0, mu_ix - mu_jx + 2 * mu_iy - 2 * mu_jy)) +
                                    4 * np.exp(complex(0, mu_ix - mu_jx)) -
                                    np.exp(complex(0, -mu_ix + mu_jx + 2 * mu_iy - 2 * mu_jy))) / 64
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
        for i in oct_index:
            b4l = ele_list[i].k3 * ele_list[i].length / 6
            beta_x = (ele_list[i].betax + ele_list[i + 1].betax) / 2
            beta_y = (ele_list[i].betay + ele_list[i + 1].betay) / 2
            mu_x = (ele_list[i].psix + ele_list[i + 1].psix) / 2
            mu_y = (ele_list[i].psiy + ele_list[i + 1].psiy) / 2
            Qxx += 3 * b4l * beta_x ** 2 / 8 / pi
            Qxy -= 3 * b4l * beta_x * beta_y / (4 * pi)
            Qyy += 3 * b4l * beta_y ** 2 / 8 / pi
            h31000 += -b4l * beta_x ** 2 * np.exp(complex(0, 2 * mu_x)) / 16
            h40000 += -b4l * beta_x ** 2 * np.exp(complex(0, 4 * mu_x)) / 64
            h20110 += 3 * b4l * beta_x * beta_y * np.exp(complex(0, 2 * mu_x)) / 16
            h11200 += 3 * b4l * beta_x * beta_y * np.exp(complex(0, 2 * mu_y)) / 16
            h20020 += 3 * b4l * beta_x * beta_y * np.exp(complex(0, 2 * mu_x - 2 * mu_y)) / 32
            h20200 += 3 * b4l * beta_x * beta_y * np.exp(complex(0, 2 * mu_x + 2 * mu_y)) / 32
            h00310 += -b4l * beta_y ** 2 * np.exp(complex(0, 2 * mu_y)) / 16
            h00400 += -b4l * beta_y ** 2 * np.exp(complex(0, 4 * mu_y)) / 64
            h22000 += -3 * b4l * beta_x ** 2 / 32
            h11110 += 3 * b4l * beta_x * beta_y / 8
            h00220 += -3 * b4l * beta_y ** 2 / 32
        nonlinear_terms = {'h21000': abs(h21000), 'h30000': abs(h30000), 'h10110': abs(h10110),
                           'h10020': abs(h10020),
                           'h10200': abs(h10200), 'h20001': abs(h20001), 'h00201': abs(h00201),
                           'h10002': abs(h10002),
                           'h11001': abs(h11001), 'h00111': abs(h00111), 'Qxx': Qxx, 'Qxy': Qxy, 'Qyy': Qyy,
                           'h31000': abs(h31000), 'h40000': abs(h40000), 'h20110': abs(h20110),
                           'h11200': abs(h11200),
                           'h20020': abs(h20020), 'h20200': abs(h20200), 'h00310': abs(h00310),
                           'h00400': abs(h00400),
                           'h22000': abs(h22000), 'h11110': abs(h11110), 'h00220': abs(h00220)}
        if printout:
            print('\nnonlinear terms:')
            for i, j in nonlinear_terms.items():
                print(f'    {str(i):7}: {j:.2f}')
        return nonlinear_terms

    def higher_order_chromaticity(self, delta=1e-2, matrix_precision=1e-9, resdl_limit=1e-12):
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
            """reference: SAMM: Simple Accelerator Modelling in Matlab, A. Wolski, 2013 """
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
                resdl = sum(dco ** 2) ** 0.5
                j += 1
            cos_mu = (matrix[0, 0] + matrix[1, 1]) / 2
            assert abs(cos_mu) < 1, "can not find period solution, UNSTABLE!!!"
            nux = np.arccos(cos_mu) * np.sign(matrix[0, 1]) / 2 / pi
            nuy = np.arccos((matrix[2, 2] + matrix[3, 3]) / 2) * np.sign(matrix[2, 3]) / 2 / pi
            return nux - np.floor(nux), nuy - np.floor(nuy)

        try:
            nux1, nuy1 = closed_orbit_tune(delta)
            nux_1, nuy_1 = closed_orbit_tune(-delta)
            nux2, nuy2 = closed_orbit_tune(2 * delta)
            nux_2, nuy_2 = closed_orbit_tune(-2 * delta)
        except Exception as e:
            print(e)
            raise Exception('can not find off-momentum closed orbit, try smaller delta.')
        nux0 = self.nux / self.n_periods - int(self.nux / self.n_periods)
        nuy0 = self.nuy / self.n_periods - int(self.nuy / self.n_periods)
        xi2x = self.n_periods * (nux1 + nux_1 - 2 * nux0) / 2 / delta ** 2
        xi2y = self.n_periods * (nuy1 + nuy_1 - 2 * nuy0) / 2 / delta ** 2
        xi3x = self.n_periods * (nux2 - 2 * nux1 + 2 * nux_1 - nux_2) / delta ** 3 / 12
        xi3y = self.n_periods * (nuy2 - 2 * nuy1 + 2 * nuy_1 - nuy_2) / delta ** 3 / 12
        xi4x = self.n_periods * (nux2 - 4 * nux1 + 6 * nux0 - 4 * nux_1 + nux_2) / delta ** 4 / 24
        xi4y = self.n_periods * (nuy2 - 4 * nuy1 + 6 * nuy0 - 4 * nuy_1 + nuy_2) / delta ** 4 / 24
        print(f'xi2x: {xi2x:.2f}, xi2y: {xi2y:.2f}, xi3x: {xi3x:.2f}, xi3y: {xi3y:.2f}, xi4x: {xi4x:.2f}, xi4y: {xi4y:.2f}')
        return {'xi2x': xi2x, 'xi2y': xi2y, 'xi3x': xi3x, 'xi3y': xi3y, 'xi4x': xi4x, 'xi4y': xi4y}

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
        for ele in self.elements:
            file1.write(f'{ele.s:.6e} {ele.name:10} {ele.betax:.6e}  {ele.alphax:.6e}  {ele.psix / 2 / pi:.6e}  '
                        f'{ele.betay:.6e}  {ele.alphay:.6e}  {ele.psiy / 2 / pi:.6e}  {ele.etax:.6e}  {ele.etaxp:.6e}\n')
        file1.close()

    def __add__(self, other):
        assert isinstance(other, CSLattice), 'can only add CSLattice.'
        newlattice = CSLattice(self.elements * self.n_periods + other.elements * other.n_periods)
        return newlattice

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

