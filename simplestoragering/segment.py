"""this is almost the same as cslattice, but I don't want to change the code in cslattice.py because my holiday is coming,
so I wrote a new class. Maybe someday I will merge these two files."""
import copy

from .constants import LENGTH_PRECISION, pi, Cq, Cr, c
from .components import LineEnd, Mark
from .particles import RefParticle
import numpy as np


class Segment(object):
    """half cell"""

    def __init__(self, ele_list: list):
        self.length = 0
        self.elements = []
        self.mark = []
        self.matrix = np.identity(6)
        for ele in ele_list:
            self.matrix = ele.matrix.dot(self.matrix)
            self.elements.append(ele)
            self.length = round(self.length + ele.length, LENGTH_PRECISION)
        self.ele_slices = None
        self.ring_length = self.length
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

    def __slice(self):
        self.ele_slices = []
        current_s = 0
        current_identifier = 0
        for ele in self.elements:
            [new_list, current_s] = ele.slice(current_s, current_identifier)
            self.ele_slices += new_list
            if isinstance(new_list[0], Mark):
                self.mark.append(new_list[0])
            current_identifier += 1
        last_ele = LineEnd(s=self.length, identifier=current_identifier)
        self.ele_slices.append(last_ele)

    def initialize_twiss_as_cell(self):
        matrix = np.identity(6)
        for ele in self.ele_slices:
            matrix = ele.matrix.dot(matrix)
        betax2 = -matrix[0, 1] * matrix[1, 1] / matrix[0, 0] / matrix[1, 0]
        betay2 = -matrix[2, 3] * matrix[3, 3] / matrix[2, 2] / matrix[3, 2]
        if betax2 <= 0 or betay2 <= 0 or matrix[1, 0] == 0:
            # print(self.__str__())
            raise Exception
        betax = (betax2) ** 0.5
        alphax = 0
        gammax = 1 / betax
        betay = (betay2) ** 0.5
        alphay = 0
        gammay = 1 / betay
        etaxp = 0
        etax = - matrix[1, 5] / matrix[1, 0]
        self.twiss_x0 = np.array([betax, alphax, gammax])
        self.twiss_y0 = np.array([betay, alphay, gammay])
        self.eta_x0 = np.array([etax, etaxp])

    def initialize_twiss_as_end_cell(self, twiss_x0, twiss_y0, eta_x0):
        matrix = np.identity(6)
        for ele in self.elements:
            matrix = ele.matrix.dot(matrix)
        alpha_xf = matrix[0, 0] * matrix[1, 0] * twiss_x0[0] + matrix[0, 1] * matrix[1, 1] * twiss_x0[2]
        alpha_yf = matrix[2, 2] * matrix[3, 2] * twiss_y0[0] + matrix[2, 3] * matrix[3, 3] * twiss_y0[2]
        eta_f = matrix[1, 0] * eta_x0[0] + matrix[1, 5]
        if abs(alpha_xf) > 1e-5 or abs(alpha_yf) > 1e-5 or abs(eta_f) > 1e-5:
            raise Exception
        self.twiss_x0 = copy.deepcopy(twiss_x0)
        self.twiss_y0 = copy.deepcopy(twiss_y0)
        self.eta_x0 = copy.deepcopy(eta_x0)

    def set_twiss(self, twiss_x0, twiss_y0, eta_x0):
        self.twiss_x0 = copy.deepcopy(twiss_x0)
        self.twiss_y0 = copy.deepcopy(twiss_y0)
        self.eta_x0 = copy.deepcopy(eta_x0)

    def compute(self, straight=0):
        [betax, alphax, gammax] = self.twiss_x0
        [betay, alphay, gammay] = self.twiss_y0
        [etax, etaxp] = self.eta_x0
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
            ele.etaxp = etaxp
            ele.nux = nux
            ele.nuy = nuy
            ele.curl_H = ele.gammax * ele.etax ** 2 + 2 * ele.alphax * ele.etax * ele.etaxp + ele.betax * ele.etaxp ** 2
            [betax, alphax, gammax] = ele.next_twiss('x')
            [betay, alphay, gammay] = ele.next_twiss('y')
            [etax, etaxp] = ele.next_eta_bag('x')
            nux, nuy = ele.next_phase()
        self.nux = nux
        self.nuy = nuy
        if not straight:
            self.radiation_integrals()
            self.global_parameters()

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
        self.I1 = integral1
        self.I2 = integral2
        self.I3 = integral3
        self.I4 = integral4
        self.I5 = integral5
        self.natural_xi_x = natural_xi_x / (4 * pi)
        self.natural_xi_y = natural_xi_y / (4 * pi)
        self.xi_x = (natural_xi_x + sextupole_part_xi_x) / (4 * pi)
        self.xi_y = (natural_xi_y + sextupole_part_xi_y) / (4 * pi)

    def global_parameters(self):
        self.Jx = 1 - self.I4 / self.I2
        self.Jy = 1
        self.Js = 2 + self.I4 / self.I2
        # self.sigma_e = RefParticle.gamma * np.sqrt(Cq * self.I3 / (self.Js * self.I2))
        self.emittance = Cq * RefParticle.gamma * RefParticle.gamma * self.I5 / (self.Jx * self.I2)
        # self.U0 = Cr * RefParticle.energy ** 4 * self.I2 / (2 * pi)
        # self.f_c = c * RefParticle.beta / self.ring_length
        # self.tau0 = 2 * RefParticle.energy / self.U0 / self.f_c
        # self.tau_s = self.tau0 / self.Js
        # self.tau_x = self.tau0 / self.Jx
        # self.tau_y = self.tau0 / self.Jy
        # self.alpha = self.I1 * 98 * self.f_c / c  # momentum compaction factor
        # self.etap = self.alpha - 1 / RefParticle.gamma ** 2  # phase slip factor

    def __str__(self):
        return f'nux: {self.nux: .6f}\nnuy: {self.nuy: .6f}\nlength = {self.length: .6f}\nxi_x = {self.xi_x: .6f}\nxi_y = {self.xi_y: .6f}\n' \
               f'Jx = {self.Jx: .6f}\nemittance = {self.emittance * 1e9: .6f} nm rad\n' \
               f'    {self.elements[0]}\n    {self.elements[1]}\n    {self.elements[2]}\n    {self.elements[3]}'

