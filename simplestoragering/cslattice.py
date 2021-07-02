# -*- coding: utf-8 -*-
import numpy as np
from .components import LineEnd
from .rfcavity import RFCavity
from .constants import pi, c, Cq, Cr, LENGTH_PRECISION
from .particles import RefParticle


class CSLattice(object):
    """lattice object, solve by Courant-Snyder method"""

    def __init__(self, ele_list: list, periods_number: int, coupling=0.00):
        self.length = 0
        self.periods_number = periods_number
        self.coup = coupling
        self.elements = []
        self.rf_cavity = None
        for ele in ele_list:
            if isinstance(ele, RFCavity):
                self.rf_cavity = ele
            self.elements.append(ele)
            self.length = round(self.length + ele.length, LENGTH_PRECISION)
        self.ele_slices = None
        self.__slice()
        # initialize twiss
        self.twiss_x0 = None
        self.twiss_y0 = None
        self.eta_x0 = None
        self.eta_y0 = None
        self.__solve_initial_twiss()
        # solve twiss
        self.nux = None
        self.nuy = None
        self.nuz = None
        self.__solve_along()
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
        self.radiation_integrals()
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
        self.global_parameters()

    # def compute(self):
    #     """calculate optical functions and ring parameters."""
    #
    #     self.__solve_initial_twiss()
    #     self.__solve_along()
    #     self.radiation_integrals()
    #     self.global_parameters()

    def __slice(self):
        self.ele_slices = []
        current_s = 0
        current_identifier = 0
        for ele in self.elements:
            [new_list, current_s] = ele.slice(current_s, current_identifier)
            self.ele_slices += new_list
            current_identifier += 1
        last_ele = LineEnd(s=self.length, identifier=current_identifier)
        self.ele_slices.append(last_ele)

    def __solve_initial_twiss(self):
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

    def __solve_along(self):
        [betax, alphax, gammax] = self.twiss_x0
        [betay, alphay, gammay] = self.twiss_y0
        [etax, etaxp] = self.eta_x0
        [etay, etayp] = self.eta_y0
        nux = 0
        nuy = 0
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
            ele.nux = nux
            ele.nuy = nuy
            ele.curl_H = ele.gammax * ele.etax ** 2 + 2 * ele.alphax * ele.etax * ele.etaxp + ele.betax * ele.etaxp ** 2
            [betax, alphax, gammax] = ele.next_twiss('x')
            [betay, alphay, gammay] = ele.next_twiss('y')
            [etax, etaxp] = ele.next_eta_bag('x')
            [etay, etayp] = ele.next_eta_bag('y')
            nux, nuy = ele.next_phase()
        self.nux = nux * self.periods_number
        self.nuy = nuy * self.periods_number

    def radiation_integrals(self):
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
            integral1 = integral1 + ele.length * ele.etax * ele.h
            integral2 = integral2 + ele.length * ele.h ** 2
            integral3 = integral3 + ele.length * abs(ele.h) ** 3
            integral4 = integral4 + ele.length * (ele.h ** 2 + 2 * ele.k1) * ele.etax * ele.h
            integral5 = integral5 + ele.length * ele.curl_H * abs(ele.h) ** 3
            natural_xi_x = natural_xi_x - (ele.k1 + ele.h ** 2) * ele.length * ele.betax
            sextupole_part_xi_x = sextupole_part_xi_x + ele.etax * ele.k2 * ele.length * ele.betax
            natural_xi_y = natural_xi_y + ele.k1 * ele.length * ele.betay
            sextupole_part_xi_y = sextupole_part_xi_y - ele.etax * ele.k2 * ele.length * ele.betay
            if 200 <= ele.symbol < 300:
                integral4 = integral4 + (ele.h ** 2 * ele.etax * np.tan(ele.theta_in)
                                         - ele.h ** 2 * ele.etax * np.tan(ele.theta_out))
                natural_xi_x = natural_xi_x + ele.h * (np.tan(ele.theta_in) + np.tan(ele.theta_out)) * ele.betax
                natural_xi_y = natural_xi_y - ele.h * (np.tan(ele.theta_in) + np.tan(ele.theta_out)) * ele.betay
        self.I1 = integral1 * self.periods_number
        self.I2 = integral2 * self.periods_number
        self.I3 = integral3 * self.periods_number
        self.I4 = integral4 * self.periods_number
        self.I5 = integral5 * self.periods_number
        self.natural_xi_x = natural_xi_x * self.periods_number / (4 * pi)
        self.natural_xi_y = natural_xi_y * self.periods_number / (4 * pi)
        self.xi_x = (natural_xi_x + sextupole_part_xi_x) * self.periods_number / (4 * pi)
        self.xi_y = (natural_xi_y + sextupole_part_xi_y) * self.periods_number / (4 * pi)

    def global_parameters(self):
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

    def matrix_output(self, file_name: str = 'matrix.txt'):
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
        """output s, ElementName, betax, alphax, nux, betay, alphay, nuy, etax, etaxp"""

        file1 = open(file_name, 'w')
        file1.write('& s, ElementName, betax, alphax, nux, betay, alphay, nuy, etax, etaxp\n')
        last_identifier = 123465
        for ele in self.ele_slices:
            if ele.identifier != last_identifier:
                file1.write(f'{ele.s:.6e} {ele.name:10} {ele.betax:.6e}  {ele.alphax:.6e}  {ele.nux:.6e}  '
                            f'{ele.betay:.6e}  {ele.alphay:.6e}  {ele.nuy:.6e}  {ele.etax:.6e}  {ele.etaxp:.6e}\n')
                last_identifier = ele.identifier
        file1.close()

    def __str__(self):
        val = ""
        val += ("nux =       " + str(self.nux))
        val += ("\nnuy =       " + str(self.nuy))
        val += ("\ncurl_H =    " + str(self.ele_slices[0].curl_H))
        val += ("\nI1 =        " + str(self.I1))
        val += ("\nI2 =        " + str(self.I2))
        val += ("\nI3 =        " + str(self.I3))
        val += ("\nI4 =        " + str(self.I4))
        val += ("\nI5 =        " + str(self.I5))
        val += ("\nJs =        " + str(self.Js))
        val += ("\nJx =        " + str(self.Jx))
        val += ("\nJy =        " + str(self.Jy))
        val += ("\nenergy =    " + str(RefParticle.energy) + "MeV")
        val += ("\ngamma =     " + str(RefParticle.gamma))
        val += ("\nsigma_e =   " + str(self.sigma_e))
        val += ("\nemittance = " + str(self.emittance * 1e9) + " nm*rad")
        val += ("\nLength =    " + str(self.length * self.periods_number) + " m")
        val += ("\nU0 =        " + str(self.U0 * 1000) + "  keV")
        val += ("\nTperiod =   " + str(1 / self.f_c * 1e9) + " nsec")
        val += ("\nalpha =     " + str(self.alpha))
        val += ("\neta_p =     " + str(self.etap))
        val += ("\ntau0 =      " + str(self.tau0 * 1e3) + " msec")
        val += ("\ntau_e =     " + str(self.tau_s * 1e3) + " msec")
        val += ("\ntau_x =     " + str(self.tau_x * 1e3) + " msec")
        val += ("\ntau_y =     " + str(self.tau_y * 1e3) + " msec")
        val += ("\nnatural_xi_x =" + str(self.natural_xi_x))
        val += ("\nnatural_xi_y =" + str(self.natural_xi_y))
        val += ("\nxi_x =      " + str(self.xi_x))
        val += ("\nxi_y =      " + str(self.xi_y))
        if self.sigma_z is not None:
            val += ("\nnuz =       " + str(self.nuz))
            val += ("\nsigma_z =   " + str(self.sigma_z))
        return val
