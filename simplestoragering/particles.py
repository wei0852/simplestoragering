# -*- coding: utf-8 -*-
from .constants import me, c
import numpy as np


class RefParticle(object):
    """set particle's type and energy"""
    energy = None
    gamma = None
    beta = None
    rigidity = None

    @classmethod
    def set_energy(cls, energy):
        """MeV"""
        cls.energy = energy
        cls.gamma = cls.energy / me + 1
        cls.beta = np.sqrt(1 - 1 / cls.gamma ** 2)
        cls.rigidity = cls.gamma * me * cls.beta * 1e6 / c

    @classmethod
    def __str__(cls):
        return f"Electron, gamma = {cls.gamma: .3f}"


class Beam7(object):
    """beam with 7 particles, track beam7 to solve transfer matrix"""

    def __init__(self, particle=None):
        self.matrix = None
        self.precision = 1e-9  # the precision must be small
        if particle is not None:
            self.init_particle(particle)

    def init_particle(self, particle):
        assert len(particle) == 6
        self.matrix = np.eye(6, 7) * self.precision
        for i in range(6):
            self.matrix[i, :] = self.matrix[i, :] + particle[i]

    def set_particle(self, particle):
        particle[4] = -particle[4]
        for i in range(6):
            self.matrix[i, :] = particle[i]

    def get_particle(self) -> list:
        x = []
        for i in range(6):
            x.append(self.matrix[i, :])
        x[4] = -x[4]
        return [x[i] for i in range(6)]

    def get_particle_array(self) -> np.ndarray:
        p = np.zeros(6)
        for i in range(6):
            p[i] = self.matrix[i, 6]
        p[4] = - p[4]
        return p

    def set_dp(self, dp):
        self.matrix[5, :] = dp

    def solve_transfer_matrix(self) -> np.ndarray:
        """solve transfer matrix according to result of tracking"""
        matrix = np.zeros([6, 6])
        for i in range(6):
            matrix[:, i] = (self.matrix[:, i] - self.matrix[:, 6]) / self.precision
        return matrix


def calculate_beta(delta):
    """calculate beta of particle"""

    gamma = RefParticle.gamma * (delta * RefParticle.beta + 1)
    beta = np.sqrt(1 - 1 / gamma ** 2)
    return beta
