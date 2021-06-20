from .components import Element
from .constants import Cr, pi
from .particles import RefParticle, Beam7
from .exceptions import ParticleLost
import numpy as np


class Quadrupole(Element):
    """normal Quadrupole"""
    symbol = 300

    def __init__(self, name: str = None, length: float = 0, k1: float = 0, n_slices: int = 1):
        self.name = name
        self.length = length
        self.k1 = k1
        self.n_slices = n_slices
        if k1 > 0:
            self.symbol = 320
        else:
            self.symbol = 310

    @property
    def matrix(self):
        if self.k1 > 0:
            sqk = np.sqrt(self.k1)
            sqkl = sqk * self.length
            return np.array([[np.cos(sqkl), np.sin(sqkl) / sqk, 0, 0, 0, 0],
                             [- sqk * np.sin(sqkl), np.cos(sqkl), 0, 0, 0, 0],
                             [0, 0, np.cosh(sqkl), np.sinh(sqkl) / sqk, 0, 0],
                             [0, 0, sqk * np.sinh(sqkl), np.cosh(sqkl), 0, 0],
                             [0, 0, 0, 0, 1, self.length / RefParticle.gamma ** 2],
                             [0, 0, 0, 0, 0, 1]])
        else:
            sqk = np.sqrt(-self.k1)
            sqkl = sqk * self.length
            return np.array([[np.cosh(sqkl), np.sinh(sqkl) / sqk, 0, 0, 0, 0],
                             [sqk * np.sinh(sqkl), np.cosh(sqkl), 0, 0, 0, 0],
                             [0, 0, np.cos(sqkl), np.sin(sqkl) / sqk, 0, 0],
                             [0, 0, - sqk * np.sin(sqkl), np.cos(sqkl), 0, 0],
                             [0, 0, 0, 0, 1, self.length / RefParticle.gamma ** 2],
                             [0, 0, 0, 0, 0, 1]])

    @property
    def damping_matrix(self):
        lambda_q = Cr * RefParticle.energy ** 3 * self.k1 ** 2 * self.length / pi
        matrix = self.matrix
        matrix[5, 5] = 1 - lambda_q * (self.closed_orbit[0] ** 2 + self.closed_orbit[2] ** 2)
        matrix[5, 0] = - lambda_q * self.closed_orbit[0]
        matrix[5, 2] = - lambda_q * self.closed_orbit[2]
        return matrix

    @property
    def closed_orbit_matrix(self):
        m67 = - Cr * RefParticle.energy ** 3 * self.k1 ** 2 * self.length * (self.closed_orbit[0] ** 2 +
                                                                             self.closed_orbit[2] ** 2) / 2 / pi
        # m67 = m67 * (1 + self.closed_orbit[5]) ** 2
        matrix7 = np.identity(7)
        matrix7[0:6, 0:6] = self.matrix
        matrix7[5, 6] = m67
        return matrix7

    def symplectic_track(self, beam):
        [x0, px0, y0, py0, z0, delta0] = beam.get_particle()
        # drift
        ds = self.length / 2
        try:
            d1 = np.sqrt(1 - px0 ** 2 - py0 ** 2 + 2 * delta0 / RefParticle.beta + delta0 ** 2)
        except Exception:
            raise ParticleLost(self.s)
        x1 = x0 + ds * px0 / d1
        y1 = y0 + ds * py0 / d1
        z1 = z0 + ds * (1 - (1 + RefParticle.beta * delta0) / d1) / RefParticle.beta
        # kick
        px1 = px0 - self.k1 * x1 * self.length
        py1 = py0 + self.k1 * y1 * self.length
        # drift
        try:
            d2 = np.sqrt(1 - px1 ** 2 - py1 ** 2 + 2 * delta0 / RefParticle.beta + delta0 ** 2)
        except Exception:
            raise ParticleLost(self.s)
        x2 = x1 + ds * px1 / d2
        y2 = y1 + ds * py1 / d2
        z2 = z1 + ds * (1 - (1 + RefParticle.beta * delta0) / d2) / RefParticle.beta
        beam.set_particle([x2, px1, y2, py1, z2, delta0])
        return beam

    def real_track(self, beam: Beam7) -> Beam7:
        [x0, px0, y0, py0, z0, delta0] = beam.get_particle()
        # drift
        ds = self.length / 2
        try:
            d1 = np.sqrt(1 - px0 ** 2 - py0 ** 2 + 2 * delta0 / RefParticle.beta + delta0 ** 2)
        except Exception:
            raise ParticleLost(self.s)
        x1 = x0 + ds * px0 / d1
        y1 = y0 + ds * py0 / d1
        z1 = z0 + ds * (1 - (1 + RefParticle.beta * delta0) / d1) / RefParticle.beta
        # kick
        px1 = px0 - self.k1 * x1 * self.length
        py1 = py0 + self.k1 * y1 * self.length
        # damping
        delta1 = (delta0 - (1 + delta0 * RefParticle.beta) ** 2 * Cr * RefParticle.energy ** 3 * self.k1 ** 2 *
                  self.length * (x1 ** 2 + y1 ** 2) / 2 / pi / RefParticle.beta)
        # e1_div_e0 = (delta1 + 1) / (delta0 + 1)
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
