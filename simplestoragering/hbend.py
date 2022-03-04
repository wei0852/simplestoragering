# -*- coding: utf-8 -*-
from .components import Element
from .constants import Cr, LENGTH_PRECISION, pi
from .particles import RefParticle, Beam7, calculate_beta
from copy import deepcopy
import numpy as np


class HBend(Element):
    """horizontal Bend"""

    def __init__(self, name: str = None, length: float = 0, theta: float = 0, theta_in: float = 0, theta_out: float = 0,
                 k1: float = 0, n_slices: int = 3):
        self.name = name
        self.length = length
        self.h = theta / self.length
        self.theta_in = theta_in
        self.theta_out = theta_out
        self.n_slices = n_slices
        self.k1 = k1
        self.cal_matrix()

    @property
    def theta(self):
        return self.h * self.length

    def set_slices(self, n_slices):
        self.n_slices = n_slices

    def cal_matrix(self):
        h_beta = self.h / RefParticle.beta
        inlet_edge = np.array([[1, 0, 0, 0, 0, 0],
                               [np.tan(self.theta_in) * self.h, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, -np.tan(self.theta_in) * self.h, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]])
        fx = self.k1 + self.h ** 2
        [cx, sx, dx] = self.__calculate_csd(fx)
        if fx != 0:
            m56 = self.length / RefParticle.gamma ** 2 / RefParticle.beta ** 2 - self.h ** 2 * (self.length - sx) / fx
        else:
            m56 = self.length / RefParticle.gamma ** 2 / RefParticle.beta ** 2 - self.h ** 2 * self.length ** 3 / 6
        fy = - self.k1
        [cy, sy, dy] = self.__calculate_csd(fy)
        middle_section = np.array([[cx, sx, 0, 0, 0, h_beta * dx],
                                   [-fx * sx, cx, 0, 0, 0, h_beta * sx],
                                   [0, 0, cy, sy, 0, 0],
                                   [0, 0, -fy * sy, cy, 0, 0],
                                   [h_beta * sx, h_beta * dx, 0, 0, 1, - m56],
                                   [0, 0, 0, 0, 0, 1]])
        outlet_edge = np.array([[1, 0, 0, 0, 0, 0],
                                [np.tan(self.theta_out) * self.h, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, -np.tan(self.theta_out) * self.h, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])
        self.matrix = outlet_edge.dot(middle_section).dot(inlet_edge)

    def __calculate_csd(self, fu):
        if fu > 0:
            sqrt_fu_z = np.sqrt(fu) * self.length
            cu = np.cos(sqrt_fu_z)
            su = np.sin(sqrt_fu_z) / np.sqrt(fu)
            du = (1 - cu) / fu
        elif fu < 0:
            sqrt_fu_z = np.sqrt(-fu) * self.length
            cu = np.cosh(sqrt_fu_z)
            su = np.sinh(sqrt_fu_z) / np.sqrt(-fu)
            du = (1 - cu) / fu
        else:
            cu = 1
            su = self.length
            du = self.length ** 2 / 2
        return [cu, su, du]

    @property
    def damping_matrix(self):
        m66 = 1 - Cr * RefParticle.energy ** 3 * self.length * self.h ** 2 / pi
        # delta_delta = - Cr * RefParticle.energy ** 3 * self.length * self.h ** 2 / pi / 2
        matrix = self.matrix
        matrix[5, 5] = m66
        # matrix[1, 5] = matrix[1, 5] * (1 + delta_delta / self.closed_orbit[5] / 2)
        return matrix

    @property
    def closed_orbit_matrix(self):
        # m67 = -(self.closed_orbit[5] + 1) ** 2 * Cr * RefParticle.energy ** 3 * self.theta ** 2 / 2 / pi / self.length
        m67 = -Cr * RefParticle.energy ** 3 * self.theta ** 2 / 2 / pi / self.length
        matrix7 = np.identity(7)
        matrix7[0:6, 0:6] = self.matrix  # Bend doesn't use thin len approximation, so
        matrix7[5, 6] = m67
        matrix7[1, 6] = self.h * self.length * m67 / 2
        return matrix7

    def symplectic_track(self, beam: Beam7):
        p = beam.get_particle()
        p[4] = - p[4]
        p = self.matrix.dot(p)
        p[4] = - p[4]
        beam.set_particle(p)
        return beam

        # [x0, px0, y0, py0, z0, delta0] = beam.get_particle()
        # d1 = np.sqrt(1 + 2 * delta0 / RefParticle.beta + delta0 ** 2)
        # ds = self.length
        # # entrance
        # px1 = px0 + self.h * np.tan(self.theta_in) * x0
        # py1 = py0 - self.h * np.tan(self.theta_in) * y0
        # # drift
        # h = self.h
        # k1 = self.k1
        # a1 = h - self.h / d1
        #
        # wx = np.sqrt((h * self.h + k1) / d1)
        # xc = np.cos(wx * ds)
        # xs = np.sin(wx * ds) / wx
        # xs2 = np.sin(2 * wx * ds) / wx
        #
        # wy = np.sqrt(k1 / d1)
        # yc = np.cosh(wy * ds)
        # ys = ds
        # ys2 = 2 * ds
        #
        # if wy.all():
        #     ys = np.sinh(wy * ds) / wy
        #     ys2 = np.sinh(2 * wy * ds) / wy
        #
        # x2 = x0 * xc + px1 * xs / d1 + a1 * (1 - xc) / wx / wx
        # px2 = -d1 * wx * wx * x0 * xs + px1 * xc + a1 * xs * d1
        #
        # y2 = y0 * yc + py1 * ys / d1
        # py2 = d1 * wy * wy * y0 * ys + py1 * yc
        #
        # d0 = 1 / RefParticle.beta + delta0
        #
        # c0 = (1 / RefParticle.beta - d0 / d1) * ds - d0 * a1 * (h * (ds - xs) + a1 * (2 * ds - xs2) / 8) / wx / wx / d1
        #
        # c1 = -d0 * (h * xs - a1 * (2 * ds - xs2) / 4) / d1
        #
        # c2 = -d0 * (h * (1 - xc) / wx / wx + a1 * xs * xs / 2) / d1 / d1
        #
        # c11 = -d0 * wx * wx * (2 * ds - xs2) / d1 / 8
        # c12 = d0 * wx * wx * xs * xs / d1 / d1 / 2
        # c22 = -d0 * (2 * ds + xs2) / d1 / d1 / d1 / 8
        #
        # c33 = -d0 * wy * wy * (2 * ds - ys2) / d1 / 8
        # c34 = -d0 * wy * wy * ys * ys / d1 / d1 / 2
        # c44 = -d0 * (2 * ds + ys2) / d1 / d1 / d1 / 8
        #
        # z1 = (z0 + c0 + c1 * x0 + c2 * px1 + c11 * x0 * x0 + c12 * x0 * px1 + c22 * px1 * px1 + c33 * y0 * y0 +
        #       c34 * y0 * py1 + c44 * py1 * py1)
        # # exit
        # px3 = px2 + self.h * np.tan(self.theta_out) * x2
        # py3 = py2 - self.h * np.tan(self.theta_out) * y2
        # beam.set_particle([x2, px3, y2, py3, z1, delta0])
        # return beam

    def real_track(self, beam: Beam7) -> Beam7:
        [x0, px0, y0, py0, z0, delta0] = beam.get_particle()
        # delta1 = delta0 - (delta0 + 1) ** 2 * Cr * RefParticle.energy ** 3 * self.theta ** 2 / 2 / pi / self.length
        # delta1 = (delta0 - (delta0 * RefParticle.beta + 1) ** 2 * Cr * RefParticle.energy ** 3 * self.theta ** 2 /
        #           2 / pi / self.length / RefParticle.beta)
        # use average energy
        delta01 = (delta0)
        d1 = np.sqrt(1 + 2 * delta01 / RefParticle.beta + delta01 ** 2)
        ds = self.length
        # entrance
        px1 = px0 + self.h * np.tan(self.theta_in) * x0
        py1 = py0 - self.h * np.tan(self.theta_in) * y0
        # drift
        h = self.h
        k1 = self.k1
        a1 = h - self.h / d1

        wx = np.sqrt((h * self.h + k1) / d1)
        xc = np.cos(wx * ds)
        xs = np.sin(wx * ds) / wx
        xs2 = np.sin(2 * wx * ds) / wx

        wy = np.sqrt(k1 / d1)
        yc = np.cosh(wy * ds)
        ys = ds
        ys2 = 2 * ds

        if wy.all():
            ys = np.sinh(wy * ds) / wy
            ys2 = np.sinh(2 * wy * ds) / wy

        x2 = x0 * xc + px1 * xs / d1 + a1 * (1 - xc) / wx / wx
        px2 = -d1 * wx * wx * x0 * xs + px1 * xc + a1 * xs * d1

        y2 = y0 * yc + py1 * ys / d1
        py2 = d1 * wy * wy * y0 * ys + py1 * yc

        d0 = 1 / RefParticle.beta + delta0

        c0 = (1 / RefParticle.beta - d0 / d1) * ds - d0 * a1 * (h * (ds - xs) + a1 * (2 * ds - xs2) / 8) / wx / wx / d1

        c1 = -d0 * (h * xs - a1 * (2 * ds - xs2) / 4) / d1

        c2 = -d0 * (h * (1 - xc) / wx / wx + a1 * xs * xs / 2) / d1 / d1

        c11 = -d0 * wx * wx * (2 * ds - xs2) / d1 / 8
        c12 = d0 * wx * wx * xs * xs / d1 / d1 / 2
        c22 = -d0 * (2 * ds + xs2) / d1 / d1 / d1 / 8

        c33 = -d0 * wy * wy * (2 * ds - ys2) / d1 / 8
        c34 = -d0 * wy * wy * ys * ys / d1 / d1 / 2
        c44 = -d0 * (2 * ds + ys2) / d1 / d1 / d1 / 8

        z1 = (z0 + c0 + c1 * x0 + c2 * px1 + c11 * x0 * x0 + c12 * x0 * px1 + c22 * px1 * px1 + c33 * y0 * y0 +
              c34 * y0 * py1 + c44 * py1 * py1)
        # damping
        delta_ct = (self.length / RefParticle.beta - (z1 - z0))
        current_beta = calculate_beta(delta0)
        e_loss_div_e0 = ((delta0 * RefParticle.beta + 1) ** 2 * Cr * RefParticle.energy ** 3 * h ** 2 *
                         RefParticle.beta ** 2 * current_beta ** 2 / 2 / pi) * delta_ct
        delta1 = (delta0 - e_loss_div_e0 / RefParticle.beta)
        # e1_div_e0 = (delta1 + 1) / (delta0 + 1)  # approx
        e1_div_e0 = np.sqrt(((1 + delta1 * RefParticle.beta) ** 2 - 1 / RefParticle.gamma ** 2) /
                            ((1 + delta0 * RefParticle.beta) ** 2 - 1 / RefParticle.gamma ** 2))
        px2 = px2 * e1_div_e0
        py2 = py2 * e1_div_e0
        # exit
        px3 = px2 + self.h * np.tan(self.theta_out) * x2
        py3 = py2 - self.h * np.tan(self.theta_out) * y2
        beam.set_particle([x2, px3, y2, py3, z1, delta1])
        return beam

    def radiation_integrals(self):
        integral1 = self.length * self.etax * self.h
        integral2 = self.length * self.h ** 2
        integral3 = self.length * abs(self.h) ** 3
        integral4 = self.length * (
                    self.h ** 2 + 2 * self.k1) * self.etax * self.h + 2 * self.h ** 2 * self.etax * np.tan(
            self.theta_in)
        if self.theta_out != 0:
            eta = self.matrix[:2, :2].dot(np.array([self.etax, self.etaxp])) + np.array(
                [self.matrix[0, 5], self.matrix[1, 5]])
            integral4 -= 2 * self.h ** 2 * eta[0] * np.tan(self.theta_out)
        curl_H = self.gammax * self.etax ** 2 + 2 * self.alphax * self.etax * self.etaxp + self.betax * self.etaxp ** 2
        integral5 = self.length * curl_H * abs(self.h) ** 3
        xi_x = (- (self.k1 + self.h ** 2) * self.length * self.betax + self.h * (
                np.tan(self.theta_in) + np.tan(self.theta_out)) * self.betax) / 4 / pi
        xi_y = (self.k1 * self.length * self.betay - self.h * (
                np.tan(self.theta_in) + np.tan(self.theta_out)) * self.betay) / 4 / pi
        xi_x += - 2 * self.k1 * (np.tan(self.theta_in) + np.tan(self.theta_out)) * self.etax * self.betax / 4 / pi
        xi_y += 2 * self.k1 * (np.tan(self.theta_in) + np.tan(self.theta_out)) * self.etax * self.betay / 4 / pi
        return integral1, integral2, integral3, integral4, integral5, xi_x, xi_y

    def slice(self, initial_s, identifier):
        """slice component to element list, return [ele_list, final_z]

        this method is rewritten because of the edge angles."""

        ele_list = []
        current_s = initial_s
        if self.n_slices == 1:
            ele = HBend(self.name, self.length, self.theta, self.theta_in, self.theta_out, self.k1, 1)
            ele.s = current_s
            ele.identifier = identifier
            current_s = round(current_s + self.length, LENGTH_PRECISION)
            return [[ele], current_s]
        length = round(self.length / self.n_slices, LENGTH_PRECISION)
        ele = HBend(self.name, length, self.h * length, self.theta_in, 0, self.k1, 1)
        ele.s = current_s
        ele.identifier = identifier
        ele_list.append(ele)
        current_s = round(current_s + ele.length, LENGTH_PRECISION)
        for i in range(self.n_slices - 2):
            ele = HBend(self.name, length, self.h * length, 0, 0, self.k1, 1)
            ele.s = current_s
            ele.identifier = identifier
            ele_list.append(ele)
            current_s = round(current_s + ele.length, LENGTH_PRECISION)
        length = round(self.length + initial_s - current_s, LENGTH_PRECISION)
        ele = HBend(self.name, length, self.h * length, 0, self.theta_out, self.k1, 1)
        ele.s = current_s
        ele.identifier = identifier
        ele_list.append(ele)
        current_s = round(current_s + ele.length, LENGTH_PRECISION)
        return [ele_list, current_s]

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
