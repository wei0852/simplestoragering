# -*- coding: utf-8 -*-
import numpy as np
import copy
from .components import LineEnd, Mark
from .rfcavity import RFCavity
from .constants import pi, c, Cq, Cr, LENGTH_PRECISION
from .particles import RefParticle
from .hbend import HBend


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
        for ele in ele_list:
            if isinstance(ele, RFCavity):
                self.rf_cavity = ele
            self.elements.append(ele)
            self.length = round(self.length + ele.length, LENGTH_PRECISION)
            if isinstance(ele, HBend):
                self.angle += ele.theta
                self.abs_angle += abs(ele.theta)
        self.angle = self.angle * 180 / pi * periods_number
        self.abs_angle = self.abs_angle * 180 / pi * periods_number
        self.ele_slices = None
        self.__slice()
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

    def compute(self):
        """calculate optical functions and ring parameters."""

        self.__solve_along()
        self.__radiation_integrals()
        self.__global_parameters()

    def __slice(self):
        self.ele_slices = []
        current_s = 0
        current_identifier = 0
        for ele in self.elements:
            [new_list, current_s] = ele.slice(current_s, current_identifier)
            self.ele_slices += new_list
            if isinstance(new_list[0], Mark):
                if new_list[0].name in self.mark:
                    self.mark[new_list[0].name].append(new_list[0])
                else:
                    self.mark[new_list[0].name] = [new_list[0]]
            current_identifier += 1
        last_ele = LineEnd(s=self.length, identifier=current_identifier)
        self.ele_slices.append(last_ele)

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
        psix = 0
        psiy = 0
        for ele in self.ele_slices:
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
            ele.curl_H = ele.gammax * ele.etax ** 2 + 2 * ele.alphax * ele.etax * ele.etaxp + ele.betax * ele.etaxp ** 2
            [betax, alphax, gammax] = ele.next_twiss('x')
            [betay, alphay, gammay] = ele.next_twiss('y')
            [etax, etaxp] = ele.next_eta_bag('x')
            [etay, etayp] = ele.next_eta_bag('y')
            psix, psiy = ele.next_phase()
        self.nux = psix * self.periods_number / 2 / pi
        self.nuy = psiy * self.periods_number / 2 / pi

    def __radiation_integrals(self):
        integral1 = 0
        integral2 = 0
        integral3 = 0
        integral4 = 0
        integral5 = 0
        natural_xi_x = 0
        sextupole_part_xi_x = 0
        natural_xi_y = 0
        sextupole_part_xi_y = 0
        for ele in self.ele_slices:
            i1, i2, i3, i4, i5, xix, xiy = ele.radiation_integrals()
            integral1 += i1
            integral2 += i2
            integral3 += i3
            integral4 += i4
            integral5 += i5
            if ele.type == 'Sextupole':
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

    def compute_adts(self):
        sext_list = []
        Qxx = 0
        Qyy = 0
        Qxy = 0
        pi_nux = pi * self.nux
        pi_nuy = pi * self.nuy
        for ele in self.ele_slices:
            if ele.type == 'Sextupole':
                sext_list.append(ele)
        for j in range(len(sext_list)):
            k2_l_j = sext_list[j].k2 * sext_list[j].length
            beta_xj = sext_list[j].betax
            beta_yj = sext_list[j].betay
            phi_xj = sext_list[j].psix
            phi_yj = sext_list[j].psiy
            for k in range(len(sext_list)):
                k2l = sext_list[k].k2 * sext_list[k].length * k2_l_j
                beta_xjk = sext_list[k].betax * beta_xj
                beta_yk = sext_list[k].betay
                phi_jk_x = abs(sext_list[k].psix - phi_xj)
                phi_jk_y = abs(sext_list[k].psiy - phi_yj)
                Qxx += k2l * beta_xjk ** 1.5 * (3 * np.cos(phi_jk_x - pi_nux) / np.sin(pi_nux) + np.cos(3 * phi_jk_x - 3 * pi_nux) / np.sin(3 * pi_nux))
                # Qxy += k2l * beta_xjk ** 0.5 * beta_yj * ()
        Qxx = -Qxx / 64 / pi
        print(f'Qxx = {Qxx}')

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
        for ele in self.ele_slices:
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
        val += f'\n{str("I1 ="):11} {self.I1:9.3e}'
        val += f'\n{str("I2 ="):11} {self.I2:9.3e}'
        val += f'\n{str("I3 ="):11} {self.I3:9.3e}'
        val += f'\n{str("I4 ="):11} {self.I4:9.3e}'
        val += f'\n{str("I5 ="):11} {self.I5:9.3e}'
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
