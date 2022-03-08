# -*- coding: utf-8 -*-
from .components import Element, assin_twiss
from .constants import Cr, LENGTH_PRECISION, pi
from .particles import RefParticle, Beam7, calculate_beta
from .functions import next_twiss
from copy import deepcopy
import numpy as np


class HBend(Element):
    """horizontal Bend"""

    def __init__(self, name: str = None, length: float = 0, theta: float = 0, theta_in: float = 0, theta_out: float = 0,
                 k1: float = 0):
        self.name = name
        self.length = length
        self.h = theta / self.length
        self.theta_in = theta_in
        self.theta_out = theta_out
        self.k1 = k1

    @property
    def theta(self):
        return self.h * self.length

    def set_slices(self, n_slices):
        self.n_slices = n_slices

    @property
    def matrix(self):
        return _hbend_matrix(self.length, self.h, self.theta_in, self.theta_out, self.k1)

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

    def symplectic_track(self, particle):
        # [x0, px0, y0, py0, ct0, dp0] = beam.get_particle()
        [x0, px0, y0, py0, ct0, dp0] = particle

        beta0 = RefParticle.beta

        ds = self.length
        k0 = self.h
        d1 = np.sqrt(1 + 2 * dp0 / beta0 + dp0 ** 2)

        r10 = k0 * np.tan(self.theta_in)
        r32 = -k0 * np.tan(self.theta_in)

        px1 = px0 + r10 * x0
        py1 = py0 + r32 * y0

        # % Then, apply a map for the body of the dipole
        h = self.h
        k1 = self.k1
        a1 = h - k0 / d1

        wx = np.sqrt(complex(h * k0 + k1, 0) / d1)
        wx_square = np.real(wx ** 2)                    # 转换为float
        xc = np.real(np.cos(wx * ds))
        xs = np.real(np.sin(wx * ds) / wx)
        xs2 = np.real(np.sin(2 * wx * ds) / wx)

        wy = np.sqrt(complex(k1, 0) / d1)
        wy_square = np.real(wy ** 2)
        yc = np.real(np.cosh(wy * ds))
        ys = ds
        ys2 = 2 * ds

        if wy.all():
            ys = np.real(np.sinh(wy * ds) / wy)
            ys2 = np.real(np.sinh(2 * wy * ds) / wy)

        x2 = x0 * xc + px1 * xs / d1 + a1 * (1 - xc) / wx_square
        px2 = -d1 * wx_square * x0 * xs + px1 * xc + a1 * xs * d1

        y2 = y0 * yc + py1 * ys / d1
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

        ct1 = ct0 + c0 + c1 * x0 + c2 * px1 + c11 * x0 * x0 + c12 * x0 * px1 + c22 * px1 * px1 + c33 * y0 * y0 + c34 * y0 * py1 + c44 * py1 * py1

        r10 = k0 * np.tan(self.theta_out)
        r32 = -k0 * np.tan(self.theta_out)

        px3 = px2 + r10 * x2
        py3 = py2 + r32 * y2

        # beam.set_particle([x2, px3, y2, py3, ct1, dp0])
        # return beam
        return np.array([x2, px3, y2, py3, ct1, dp0])

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

    def __radiation_integrals(self, length, integrals, twiss0, twiss1):
        betax = (twiss0[0] + twiss1[0]) / 2
        alphax = (twiss0[1] + twiss1[1]) / 2
        gammax = (twiss0[2] + twiss1[2]) / 2
        betay = (twiss0[3] + twiss1[3]) / 2
        etax = (twiss0[6] + twiss1[6]) / 2
        etaxp = (twiss0[7] + twiss1[7]) / 2
        integrals[0] += length * etax * self.h
        integrals[1] += length * self.h ** 2
        integrals[2] += length * abs(self.h) ** 3
        integrals[3] += length * (self.h ** 2 + 2 * self.k1) * etax * self.h
        curl_H = gammax * etax ** 2 + 2 * alphax * etax * etaxp + betax * etaxp ** 2
        integrals[4] += length * curl_H * abs(self.h) ** 3
        integrals[5] += (- (self.k1 + self.h ** 2) * length * betax) / 4 / pi
        integrals[6] += (self.k1 * length * betay) / 4 / pi

    def copy(self):
        return HBend(self.name, self.length, self.theta, self.theta_in, self.theta_out, self.k1)

    def linear_optics(self):
        twiss0 = np.array([self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy])
        integrals = np.zeros(7)
        current_s = 0
        length = 0.01
        # inlet
        integrals[3] += 2 * self.h ** 2 * self.etax * np.tan(self.theta_in)
        integrals[5] += (self.h * np.tan(self.theta_in) * self.betax - 2 * self.k1 * np.tan(self.theta_in) * self.etax * self.betax) / 4 / pi
        integrals[6] += (- self.h * np.tan(self.theta_in) * self.betay + 2 * self.k1 * np.tan(self.theta_in) * self.etax * self.betay) / 4 / pi
        matrix = _hbend_matrix(length, self.h, self.theta_in, 0, self.k1)
        twiss1 = next_twiss(matrix, twiss0)
        self.__radiation_integrals(length, integrals, twiss0, twiss1)
        current_s = round(current_s + length, LENGTH_PRECISION)
        for i in range(len(twiss0)):
            twiss0[i] = twiss1[i]
        while current_s < self.length - length:
            matrix = _hbend_matrix(length, self.h, 0, 0, self.k1)
            twiss1 = next_twiss(matrix, twiss0)
            self.__radiation_integrals(length, integrals, twiss0, twiss1)
            current_s = round(current_s + length, LENGTH_PRECISION)
            for j in range(len(twiss0)):
                twiss0[j] = twiss1[j]
        length = round(self.length - current_s, LENGTH_PRECISION)
        matrix = _hbend_matrix(length, self.h, 0, self.theta_out, self.k1)
        twiss1 = next_twiss(matrix, twiss0)
        self.__radiation_integrals(length, integrals, twiss0, twiss1)
        integrals[3] += - 2 * self.h ** 2 * twiss1[6] * np.tan(self.theta_out)
        integrals[5] += (self.h * np.tan(self.theta_out) * twiss1[0] - 2 * self.k1 * np.tan(self.theta_out) * twiss1[6] * twiss1[0]) / 4 / pi
        integrals[6] += (- self.h * np.tan(self.theta_out) * twiss1[3] + 2 * self.k1 * np.tan(self.theta_out) * twiss1[6] * twiss1[3]) / 4 / pi
        return integrals, twiss1

    def slice(self, n_slices: int) -> list:
        """slice component to element list, return [ele_list, final_z]

        this method is rewritten because of the edge angles."""

        ele_list = []
        current_s = self.s
        twiss0 = np.array([self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy])
        if n_slices == 1:
            ele = HBend(self.name, self.length, self.theta, self.theta_in, self.theta_out, self.k1)
            ele.s = current_s
            assin_twiss(ele, twiss0)
            ele.identifier = self.identifier
            return [ele]
        length = round(self.length / n_slices, LENGTH_PRECISION)
        ele = HBend(self.name, length, self.h * length, self.theta_in, 0, self.k1)
        ele.s = current_s
        assin_twiss(ele, twiss0)
        twiss0 = next_twiss(ele.matrix, twiss0)
        ele_list.append(ele)
        current_s = round(current_s + ele.length, LENGTH_PRECISION)
        for i in range(n_slices - 2):
            ele = HBend(self.name, length, self.h * length, 0, 0, self.k1)
            ele.s = current_s
            assin_twiss(ele, twiss0)
            twiss0 = next_twiss(ele.matrix, twiss0)
            ele_list.append(ele)
            current_s = round(current_s + ele.length, LENGTH_PRECISION)
        length = round(self.length + self.s - current_s, LENGTH_PRECISION)
        ele = HBend(self.name, length, self.h * length, 0, self.theta_out, self.k1)
        ele.identifier = self.identifier
        ele.s = current_s
        assin_twiss(ele, twiss0)
        ele_list.append(ele)
        return ele_list

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


def _hbend_matrix(length, h, theta_in, theta_out, k1):
    h_beta = h / RefParticle.beta
    inlet_edge = np.array([[1, 0, 0, 0, 0, 0],
                            [np.tan(theta_in) * h, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, -np.tan(theta_in) * h, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])
    fx = k1 + h ** 2
    [cx, sx, dx] = __calculate_csd(length, fx)
    if fx != 0:
        m56 = length / RefParticle.gamma ** 2 / RefParticle.beta ** 2 - h ** 2 * (length - sx) / fx
    else:
        m56 = length / RefParticle.gamma ** 2 / RefParticle.beta ** 2 - h ** 2 * length ** 3 / 6
    fy = - k1
    [cy, sy, dy] = __calculate_csd(length, fy)
    middle_section = np.array([[cx, sx, 0, 0, 0, h_beta * dx],
                                   [-fx * sx, cx, 0, 0, 0, h_beta * sx],
                                   [0, 0, cy, sy, 0, 0],
                                   [0, 0, -fy * sy, cy, 0, 0],
                                   [h_beta * sx, h_beta * dx, 0, 0, 1, - m56],
                                   [0, 0, 0, 0, 0, 1]])
    outlet_edge = np.array([[1, 0, 0, 0, 0, 0],
                                [np.tan(theta_out) * h, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, -np.tan(theta_out) * h, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])
    return outlet_edge.dot(middle_section).dot(inlet_edge)


def __calculate_csd(length, fu):
    if fu > 0:
        sqrt_fu_z = np.sqrt(fu) * length
        cu = np.cos(sqrt_fu_z)
        su = np.sin(sqrt_fu_z) / np.sqrt(fu)
        du = (1 - cu) / fu
    elif fu < 0:
        sqrt_fu_z = np.sqrt(-fu) * length
        cu = np.cosh(sqrt_fu_z)
        su = np.sinh(sqrt_fu_z) / np.sqrt(-fu)
        du = (1 - cu) / fu
    else:
        cu = 1
        su = length
        du = length ** 2 / 2
    return [cu, su, du]
