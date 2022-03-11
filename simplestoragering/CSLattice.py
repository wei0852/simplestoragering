# -*- coding: utf-8 -*-
import numpy as np
import copy
from .components import LineEnd, Mark
from .RFCavity import RFCavity
from .constants import pi, c, Cq, Cr, LENGTH_PRECISION
from .particles import RefParticle
from .HBend import HBend
from .Drift import Drift
from .Quadrupole import Quadrupole
from .Sextupole import Sextupole


class CSLattice(object):
    """lattice object, solve by Courant-Snyder method"""

    def __init__(self, ele_list: list, periods_number: int = 1, coupling: float = 0.00):
        self.length = 0
        self.periods_number = periods_number
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
            self.length = round(self.length + ele.length, LENGTH_PRECISION)
            if isinstance(ele, HBend):
                self.angle += ele.theta
                self.abs_angle += abs(ele.theta)
            current_identifier += 1
            current_s = round(current_s + ele.length, LENGTH_PRECISION)
        last_ele = LineEnd(s=self.length, identifier=current_identifier)
        self.elements.append(last_ele)
        self.angle = self.angle * 180 / pi * periods_number
        self.abs_angle = self.abs_angle * 180 / pi * periods_number
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

    def linear_optics(self):
        """calculate optical functions and ring parameters."""

        self.__solve_along()
        self.__global_parameters()

    def find_the_periodic_solution(self):
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

    def set_initial_twiss(self, twiss_x0, twiss_y0, eta_x0):
        self.twiss_x0 = copy.deepcopy(twiss_x0)
        self.twiss_y0 = copy.deepcopy(twiss_y0)
        self.eta_x0 = copy.deepcopy(eta_x0)

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
        self.I1 = integral1 * self.periods_number
        self.I2 = integral2 * self.periods_number
        self.I3 = integral3 * self.periods_number
        self.I4 = integral4 * self.periods_number
        self.I5 = integral5 * self.periods_number
        self.natural_xi_x = natural_xi_x * self.periods_number
        self.natural_xi_y = natural_xi_y * self.periods_number
        self.xi_x = (natural_xi_x + sextupole_part_xi_x) * self.periods_number
        self.xi_y = (natural_xi_y + sextupole_part_xi_y) * self.periods_number
        self.nux = self.elements[-1].nux * self.periods_number
        self.nuy = self.elements[-1].nuy * self.periods_number

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

    def compute_nonlinear_term(self):
        """compute resonance driving terms. return a dictionary
        nonlinear_terms = {'h21000': , 'h30000': , 'h10110': , 'h10020': ,
                           'h10200': , 'Qxx': , 'Qxy': , 'Qyy': ,
                           'h31000': , 'h40000': , 'h20110': , 'h11200': ,
                           'h20020': , 'h20200': , 'h00310': , 'h00400': }

        references:
        1. The Sextupole Scheme for the SLS: An Analytic Approach, SLS Note 09/97, Johan Bengtsson
        2. Explicit formulas for 2nd-order driving terms due to sextupoles and chromatic effects of quadrupoles, Chun-xi Wang"""

        ele_list = []
        for ele in self.elements:
            if isinstance(ele, Sextupole):
                ele_list += ele.slice(ele.n_slices)
            else:
                ele_list.append(ele)
        # if list_data:
        #     driving_term = {}
        #     driving_term['h21000'] = []
        #     driving_term['h30000'] = []
        #     driving_term['h10110'] = []
        #     driving_term['h10020'] = []
        #     driving_term['h10200'] = []
        Qxx = Qxy = Qyy = 0
        h21000 = h30000 = h10110 = h10020 = h10200 = 0
        h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
        num = len(ele_list)
        pi_nux = self.elements[-1].psix / 2
        pi_nuy = self.elements[-1].psiy / 2
        for i in range(num - 1):
            b3l_i = ele_list[i].k2 * ele_list[i].length / 2  # k2 = 2 * b3, k1 = b2
            if b3l_i != 0:
                beta_xi = (ele_list[i].betax + ele_list[i + 1].betax) / 2
                beta_yi = (ele_list[i].betay + ele_list[i + 1].betay) / 2
                mu_ix = (ele_list[i].psix + ele_list[i + 1].psix) / 2
                mu_iy = (ele_list[i].psiy + ele_list[i + 1].psiy) / 2
                h21000 += - b3l_i * beta_xi ** 1.5 * np.exp(complex(0, mu_ix)) / 8
                h30000 += - b3l_i * beta_xi ** 1.5 * np.exp(complex(0, 3 * mu_ix)) / 24
                h10110 += b3l_i * beta_xi ** 0.5 * beta_yi * np.exp(complex(0, mu_ix)) / 4
                h10020 += b3l_i * beta_xi ** 0.5 * beta_yi * np.exp(complex(0, mu_ix - 2 * mu_iy)) / 8
                h10200 += b3l_i * beta_xi ** 0.5 * beta_yi * np.exp(complex(0, mu_ix + 2 * mu_iy)) / 8
                for j in range(num - 1):
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
                        h31000 += const * beta_xij ** 1.5 * np.exp(complex(0, 3 * mu_ix - mu_jx)) / 32
                        h40000 += const * beta_xij ** 1.5 * np.exp(complex(0, 3 * mu_ix + mu_jx)) / 64
                        h20110 += const * beta_xij ** 0.5 * beta_yi * (
                                beta_xj * (np.exp(complex(0, 3 * mu_jx - mu_ix)) - np.exp(complex(0, mu_ix + mu_jx))) +
                                2 * beta_yj * np.exp(complex(0, mu_ix + mu_jx + 2 * mu_iy - 2 * mu_jy))) / 32
                        h11200 += const * beta_xij ** 0.5 * beta_yi * (
                                beta_xj * (np.exp(complex(0, -mu_ix + mu_jx + 2 * mu_iy)) - np.exp(complex(0, mu_ix - mu_jx + 2 * mu_iy))) +
                                2 * beta_yj * (np.exp(complex(0, mu_ix - mu_jx + 2 * mu_iy)) + np.exp(complex(0, - mu_ix + mu_jx + 2 * mu_iy)))) / 32
                        h20020 += const * beta_xij ** 0.5 * beta_yi * (
                                beta_xj * np.exp(complex(0, -mu_ix + 3 * mu_jx - 2 * mu_iy)) -
                                (beta_xj + 4 * beta_yj) * np.exp(complex(0, mu_ix + mu_jx - 2 * mu_iy))) / 64
                        h20200 += const * beta_xij ** 0.5 * beta_yi * (
                                beta_xj * np.exp(complex(0, -mu_ix + 3 * mu_jx + 2 * mu_iy))
                                - (beta_xj - 4 * beta_yj) * np.exp((complex(0, mu_ix + mu_jx + 2 * mu_iy)))) / 64
                        h00310 += const * beta_xij ** 0.5 * beta_yi * beta_yj * (
                                np.exp(complex(0, mu_ix - mu_jx + 2 * mu_iy)) -
                                np.exp(complex(0, -mu_ix + mu_jx + 2 * mu_iy))) / 32
                        h00400 += const * beta_xij ** 0.5 * beta_yi * beta_yj * np.exp(complex(0, mu_ix - mu_jx + 2 * mu_iy + 2 * mu_jy)) / 64
        nonlinear_terms = {'h21000': abs(h21000), 'h30000': abs(h30000), 'h10110': abs(h10110), 'h10020': abs(h10020),
                           'h10200': abs(h10200), 'Qxx': Qxx / 2, 'Qxy': Qxy / 2, 'Qyy': Qyy / 2,
                           'h31000': abs(h31000), 'h40000': abs(h40000), 'h20110': abs(h20110), 'h11200': abs(h11200),
                           'h20020': abs(h20020), 'h20200': abs(h20200), 'h00310': abs(h00310), 'h00400': abs(h00400)}
        print('\nnonlinear terms:')
        for i, j in nonlinear_terms.items():
            print(f'    {str(i):7}: {j:.4f}')
        return nonlinear_terms

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
        except Exception:
            print('!!!!!!!\ncan not find off-momentum closed orbit, try smaller delta.\n    !!!! you may need to change matrix_precision, too.')
            return {'xi2x': 1e9, 'xi2y': 1e9, 'xi3x': 1e9, 'xi3y': 1e9}
        xi2x = (nux1 + nux_1 - 2 * (self.nux - int(self.nux))) / 2 / delta ** 2
        xi2y = (nuy1 + nuy_1 - 2 * (self.nuy - int(self.nuy))) / 2 / delta ** 2
        xi3x = (nux3 - nux_3 + 3 * nux_1 - 3 * nux1) / (delta * 2) ** 3 / 6
        xi3y = (nuy3 - nuy_3 + 3 * nuy_1 - 3 * nuy1) / (delta * 2) ** 3 / 6
        print(f'xi2x: {xi2x:.2f}, xi2y: {xi2y:.2f}, xi3x: {xi3x:.2f}, xi3y: {xi3y:.2f}')
        return {'xi2x': xi2x, 'xi2y': xi2y, 'xi3x': xi3x, 'xi3y': xi3y}

    def __global_parameters(self):
        self.Jx = 1 - self.I4 / self.I2
        self.Jy = 1
        self.Js = 2 + self.I4 / self.I2
        self.sigma_e = RefParticle.gamma * np.sqrt(Cq * self.I3 / (self.Js * self.I2))
        self.emittance = Cq * RefParticle.gamma * RefParticle.gamma * self.I5 / (self.Jx * self.I2)
        self.U0 = Cr * RefParticle.energy ** 4 * self.I2 / (2 * pi)
        self.f_c = c * RefParticle.beta / (self.length * self.periods_number)
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
            self.nuz = (self.rf_cavity.voltage * self.rf_cavity.omega_rf * abs(np.cos(self.rf_cavity.phase) * self.etap)
                        * self.length / RefParticle.energy / c) ** 0.5 / 2 / pi
            self.sigma_z = self.sigma_e * abs(self.etap) * self.length / (2 * pi * self.nuz)

    def output_matrix(self, file_name: str = 'matrix.txt'):
        """output uncoupled matrix for each element and contained matrix"""

        matrix = np.identity(6)
        file = open(file_name, 'w')
        location = 0.0
        for ele in self.elements:
            file.write(f'{ele.type()} {ele.name} at s={location},  {ele.magnets_data()}\n')
            location = round(location + ele.length, LENGTH_PRECISION)
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

    def __str__(self):
        val = ""
        val += f'{str("angle ="):11} {self.angle:9.3f}'
        val += f'\n{str("abs_angle ="):11} {self.abs_angle:9.3f}'
        val += f'\n{str("nux ="):11} {self.nux:9.4f}'
        val += f'\n{str("nuy ="):11} {self.nuy:9.4f}'
        val += f'\n{str("I1 ="):11} {self.I1:9.5e}'
        val += f'\n{str("I2 ="):11} {self.I2:9.5e}'
        val += f'\n{str("I3 ="):11} {self.I3:9.5e}'
        val += f'\n{str("I4 ="):11} {self.I4:9.5e}'
        val += f'\n{str("I5 ="):11} {self.I5:9.5e}'
        val += f'\n{str("Js ="):11} {self.Js:9.4f}'
        val += f'\n{str("Jx ="):11} {self.Jx:9.4f}'
        val += f'\n{str("energy ="):11} {RefParticle.energy:9.2e} MeV'
        # val += f'\n{str("gamma ="):11} {RefParticle.gamma:9.2f}'
        val += f'\n{str("sigma_e ="):11} {self.sigma_e:9.3e}'
        val += f'\n{str("emittance ="):11} {self.emittance:9.3e} m*rad'
        val += f'\n{str("Length ="):11} {self.length * self.periods_number:9.3f} m'
        val += f'\n{str("U0 ="):11} {self.U0 * 1000:9.2f} keV'
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
