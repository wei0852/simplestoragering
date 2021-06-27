from .components import Element
from .constants import Cr, pi
from .particles import RefParticle, Beam7
from .drift import Drift
from .exceptions import ParticleLost
import numpy as np


class Sextupole(Element):
    """sextupole"""
    symbol = 400

    def __init__(self, name: str = None, length: float = 0, k2: float = 0, n_slices: int = 1):
        self.name = name
        self.length = length
        self.k2 = k2
        self.n_slices = n_slices
        if k2 > 0:
            self.symbol = 420
        else:
            self.symbol = 410

    @property
    def matrix(self):
        k2l = self.k2 * self.length
        x0 = self.closed_orbit[0]
        y0 = self.closed_orbit[2]
        x02_y02_2 = (x0 ** 2 - y0 ** 2) / 2  # (x0 ** 2 - y0 ** 2) / 2
        matrix = np.array([[1, 0, 0, 0, 0, 0],
                           [- k2l * x0, 1, k2l * y0, 0, 0, k2l * x02_y02_2],
                           [0, 0, 1, 0, 0, 0],
                           [k2l * y0, 0, k2l * x0, 1, 0, - k2l * x0 * y0],
                           [- k2l * x02_y02_2, 0, k2l * x0 * y0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        drift = Drift(length=self.length / 2).matrix
        total = drift.dot(matrix).dot(drift)
        return total

    @property
    def damping_matrix(self):
        """I think damping is not in thin len approximation"""
        k2l = self.k2 * self.length
        x0 = self.closed_orbit[0]
        y0 = self.closed_orbit[2]
        lambda_s = Cr * RefParticle.energy ** 3 * k2l ** 2 * (x0 ** 2 + y0 ** 2) / pi / self.length
        matrix = self.matrix
        matrix[5, 0] = - lambda_s * x0
        matrix[5, 2] = - lambda_s * y0
        matrix[5, 5] = 1 - lambda_s * (x0 ** 2 + y0 ** 2) / 2
        return matrix

    @property
    def closed_orbit_matrix(self):
        """it's different from its transform matrix, x is replaced by closed orbit x0"""
        m67 = - (Cr * RefParticle.energy ** 3 * self.k2 ** 2 * self.length *
                 (self.closed_orbit[0] ** 2 + self.closed_orbit[2] ** 2) ** 2 / 8 / pi)
        # m67 = m67 * (1 + self.closed_orbit[5]) ** 2
        matrix7 = np.identity(7)
        drift = Drift(length=self.length / 2).matrix
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
                  self.length * (x1 ** 2 + y1 ** 2) ** 2 / 8 / pi / RefParticle.beta)
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
