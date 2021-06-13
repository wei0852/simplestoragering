from scipy.constants import physical_constants
import numpy as np


class Particle(object):
    """set particle's type and energy"""
    energy = None
    gamma = None
    beta = None

    @classmethod
    def set_energy(cls, energy):
        cls.energy = energy
        text = "electron mass energy equivalent in MeV"
        mass = physical_constants[text][0]
        cls.gamma = cls.energy / mass
        cls.beta = np.sqrt(1 - 1 / cls.gamma ** 2)

    @classmethod
    def __str__(cls):
        return "mass: %s MeV, gamma = %s" % (cls.energy / cls.gamma, cls.gamma)


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
        return p

    def set_dp(self, dp):
        self.matrix[5, :] = dp

    def solve_transfer_matrix(self) -> np.ndarray:
        """solve transfer matrix according to result of tracking"""
        matrix = np.zeros([6, 6])
        for i in range(6):
            matrix[:, i] = (self.matrix[:, i] - self.matrix[:, 6]) / self.precision
        return matrix
