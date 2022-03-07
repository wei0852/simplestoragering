# -*- coding: utf-8 -*-
from .components import Element, assin_twiss
from .constants import Cr, LENGTH_PRECISION
from .particles import RefParticle, Beam7
from .drift import drift_matrix
from .exceptions import ParticleLost
from .functions import next_twiss
import numpy as np


class Sextupole(Element):
    """sextupole"""
    # symbol = 400

    def __init__(self, name: str = None, length: float = 0, k2: float = 0, n_slices: int = 1):
        self.name = name
        self.length = length
        self.k2 = k2
        self.n_slices = n_slices

    def slice(self, n_slices: int) -> list:
        """slice component to element list, return [ele_list, final_z], the identifier identifies different magnet"""
        ele_list = []
        current_s = self.s
        length = round(self.length / n_slices, LENGTH_PRECISION)
        twiss0 = np.array([self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy])
        for i in range(n_slices - 1):
            ele = Sextupole(self.name, length, self.k2)
            assin_twiss(ele, twiss0)
            ele.identifier = self.identifier
            ele.s = current_s
            twiss0 = next_twiss(ele.matrix, twiss0)
            ele_list.append(ele)
            current_s = round(current_s + ele.length, LENGTH_PRECISION)
        length = round(self.length + self.s - current_s, LENGTH_PRECISION)
        ele = Sextupole(self.name, length, self.k2)
        ele.identifier = self.identifier
        ele.s = current_s
        assin_twiss(ele, twiss0)
        ele_list.append(ele)
        return ele_list

    @property
    def matrix(self):
        return sext_matrix(self.length, self.k2, self.closed_orbit)

    @property
    def damping_matrix(self):
        """I think damping is not in thin len approximation"""
        k2l = self.k2 * self.length
        x0 = self.closed_orbit[0]
        y0 = self.closed_orbit[2]
        lambda_s = Cr * RefParticle.energy ** 3 * k2l ** 2 * (x0 ** 2 + y0 ** 2) / np.pi / self.length
        matrix = self.matrix
        matrix[5, 0] = - lambda_s * x0
        matrix[5, 2] = - lambda_s * y0
        matrix[5, 5] = 1 - lambda_s * (x0 ** 2 + y0 ** 2) / 2
        return matrix

    @property
    def closed_orbit_matrix(self):
        """it's different from its transform matrix, x is replaced by closed orbit x0"""
        m67 = - (Cr * RefParticle.energy ** 3 * self.k2 ** 2 * self.length *
                 (self.closed_orbit[0] ** 2 + self.closed_orbit[2] ** 2) ** 2 / 8 / np.pi)
        # m67 = m67 * (1 + self.closed_orbit[5]) ** 2
        matrix7 = np.identity(7)
        # drift = Drift(length=self.length / 2).matrix
        drift = drift_matrix(self.length / 2)
        matrix = np.identity(6)
        matrix[1, 5] = self.k2 * self.length * (self.closed_orbit[0] ** 2 - self.closed_orbit[2] ** 2)
        matrix[4, 0] = - matrix[1, 5]
        matrix[4, 2] = self.k2 * self.length * self.closed_orbit[0] * self.closed_orbit[2]
        matrix[3, 5] = - matrix[4, 2]
        matrix7[0: 6, 0: 6] = drift.dot(matrix).dot(drift)
        matrix7[5, 6] = m67
        matrix7[1, 6] = matrix7[4, 0]
        matrix7[3, 6] = matrix7[4, 2]
        return matrix7

    def symplectic_track(self, beam):
        [x0, px0, y0, py0, z0, delta0] = beam.get_particle()
        ds = self.length / 2
        k2 = self.k2
        # drift
        d1 = np.sqrt(1 - px0 ** 2 - py0 ** 2 + 2 * delta0 / RefParticle.beta + delta0 ** 2)
        x1 = x0 + ds * px0 / d1
        y1 = y0 + ds * py0 / d1
        z1 = z0 + ds * (1 - (1 + RefParticle.beta * delta0) / d1) / RefParticle.beta
        # kick
        px1 = px0 - (x1 * x1 - y1 * y1) * k2 * ds
        py1 = py0 + x1 * y1 * k2 * ds * 2
        # drift
        np.seterr(all='raise')
        try:
            d2 = np.sqrt(1 - px1 ** 2 - py1 ** 2 + 2 * delta0 / RefParticle.beta + delta0 ** 2)
        except FloatingPointError:
            raise ParticleLost(self.s)
        x2 = x1 + ds * px1 / d2
        y2 = y1 + ds * py1 / d2
        z2 = z1 + ds * (1 - (1 + RefParticle.beta * delta0) / d2) / RefParticle.beta
        beam.set_particle([x2, px1, y2, py1, z2, delta0])
        return beam

    def real_track(self, beam: Beam7) -> Beam7:
        [x0, px0, y0, py0, z0, delta0] = beam.get_particle()
        ds = self.length / 2
        k2 = self.k2
        # drift
        try:
            d1 = np.sqrt(1 - px0 ** 2 - py0 ** 2 + 2 * delta0 / RefParticle.beta + delta0 ** 2)
        except Exception:
            raise ParticleLost(self.s)
        x1 = x0 + ds * px0 / d1
        y1 = y0 + ds * py0 / d1
        z1 = z0 + ds * (1 - (1 + RefParticle.beta * delta0) / d1) / RefParticle.beta
        # kick
        px1 = px0 - (x1 * x1 - y1 * y1) * k2 * ds
        py1 = py0 + x1 * y1 * k2 * ds * 2
        # damping, when beta approx 1
        # delta1 = delta0 - (delta0 + 1) ** 2 * (Cr * RefParticle.energy ** 3 * self.k2 ** 2 * self.length *
        #                                        (x1 ** 2 + y1 ** 2) ** 2 / 8 / pi)     # beta0 \approx 1
        delta1 = (delta0 - (delta0 * RefParticle.beta + 1) ** 2 * Cr * RefParticle.energy ** 3 * self.k2 ** 2 *
                  self.length * (x1 ** 2 + y1 ** 2) ** 2 / 8 / np.pi / RefParticle.beta)
        # e1_div_e0 = (delta1 + 1) / (delta0 + 1)  # approximation
        e1_div_e0 = np.sqrt(((1 + delta1 * RefParticle.beta) ** 2 - 1 / RefParticle.gamma ** 2) /
                            ((1 + delta0 * RefParticle.beta) ** 2 - 1 / RefParticle.gamma ** 2))
        px1 = px1 * e1_div_e0
        py1 = py1 * e1_div_e0
        # drift
        try:
            d2 = np.sqrt(1 - px1 ** 2 - py1 ** 2 + 2 * delta1 / RefParticle.beta + delta1 ** 2)
        except Exception:
            raise ParticleLost(self.s)
        x2 = x1 + ds * px1 / d2
        y2 = y1 + ds * py1 / d2
        z2 = z1 + ds * (1 - (1 + RefParticle.beta * delta1) / d2) / RefParticle.beta

        beam.set_particle([x2, px1, y2, py1, z2, delta1])
        return beam

    def copy(self):
        return Sextupole(self.name, self.length, self.k2)

    def linear_optics(self):
        twiss0 = np.array([self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy])
        integrals = np.zeros(7)
        current_s = 0
        length = 0.01
        orbit = self.closed_orbit
        while current_s < self.length - length:
            matrix = sext_matrix(length, self.k2, orbit)
            twiss1 = next_twiss(matrix, twiss0)
            orbit = matrix.dot(orbit)
            betax = (twiss0[0] + twiss1[0]) / 2
            betay = (twiss0[3] + twiss1[3]) / 2
            etax = (twiss0[6] + twiss1[6]) / 2
            integrals[5] += etax * self.k2 * length * betax / 4 / np.pi
            integrals[6] += - etax * self.k2 * length * betay / 4 / np.pi
            current_s = round(current_s + length, LENGTH_PRECISION)
            for i in range(len(twiss0)):
                twiss0[i] = twiss1[i]
        length = round(self.length - current_s, LENGTH_PRECISION)
        matrix = sext_matrix(length, self.k2, orbit)
        twiss1 = next_twiss(matrix, twiss0)
        betax = (twiss0[0] + twiss1[0]) / 2
        betay = (twiss0[3] + twiss1[3]) / 2
        etax = (twiss0[6] + twiss1[6]) / 2
        integrals[5] += etax * self.k2 * length * betax / 4 / np.pi
        integrals[6] += - etax * self.k2 * length * betay / 4 / np.pi
        return integrals, twiss1


def sext_matrix(length, k2, closed_orbit):
    k2l = k2 * length
    x0 = closed_orbit[0]
    y0 = closed_orbit[2]
    x02_y02_2 = (x0 ** 2 - y0 ** 2) / 2  # (x0 ** 2 - y0 ** 2) / 2
    matrix = np.array([[1, 0, 0, 0, 0, 0],
                       [- k2l * x0, 1, k2l * y0, 0, 0, k2l * x02_y02_2],
                       [0, 0, 1, 0, 0, 0],
                       [k2l * y0, 0, k2l * x0, 1, 0, - k2l * x0 * y0],
                       [- k2l * x02_y02_2, 0, k2l * x0 * y0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])
    drift = drift_matrix(length=length / 2)
    return drift.dot(matrix).dot(drift)
