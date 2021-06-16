from .components import Element
from .constants import pi, c
from .particles import RefParticle, Beam7
from .drift import Drift
import numpy as np


class RFCavity(Element):
    """thin len approximation, don't have length. The unit of voltage should be the same as RefParticle.energy, MeV"""
    symbol = 500
    length = 0

    def __init__(self, name: str = None, voltage_in_MeV: float = 0, frequency: float = 0, phase: float = 0):
        self.name = name
        self.voltage = voltage_in_MeV
        self.f_rf = frequency
        self.phase = phase
        self.omega_rf = 2 * pi * self.f_rf
        self.f_c = 0

    @property
    def harmonic_number(self):
        return self.f_rf / self.f_c

    @property
    def matrix(self):
        temp_val = 2 * pi * self.f_rf / RefParticle.beta / c  # h / R
        matrix = np.identity(6)
        matrix[5, 4] = (self.voltage * temp_val * np.cos(self.phase) / RefParticle.energy)
        drift = Drift(length=self.length / 2).matrix
        total = drift.dot(matrix).dot(drift)
        return total

    @property
    def damping_matrix(self):
        temp_val = 2 * pi * self.f_rf / RefParticle.beta / c  # h / R
        z0 = self.closed_orbit[4]
        matrix = self.matrix
        matrix[1, 1] = 1 - self.voltage * (np.sin(self.phase) + temp_val * z0 * np.cos(self.phase)) / RefParticle.energy
        matrix[3, 3] = matrix[1, 1]
        return matrix

    @property
    def closed_orbit_matrix(self):
        m67 = self.voltage * np.sin(self.phase) / RefParticle.energy
        # m67 = self.voltage * np.sin(self.phase + self.omega_rf * self.closed_orbit[4] / c) / RefParticle.energy
        matrix7 = np.identity(7)
        matrix7[0:6, 0:6] = self.matrix
        matrix7[5, 6] = m67
        return matrix7

    def symplectic_track(self, beam):
        """rf cavity tracking is simplified by thin len approximation"""
        [x0, px0, y0, py0, z0, dp0] = beam.get_particle()
        beta0 = RefParticle.beta
        ds = self.length
        # First        apply        a        drift        through        ds / 2
        d1 = np.sqrt(1 - px0 * px0 - py0 * py0 + 2 * dp0 / beta0 + dp0 * dp0)
        x1 = x0 + ds * px0 / d1 / 2
        y1 = y0 + ds * py0 / d1 / 2
        z1 = z0 + ds * (1 - (1 + beta0 * dp0) / d1) / beta0 / 2
        # Next, apply        an        rf        'kick'
        vnorm = self.voltage / RefParticle.energy
        dp1 = dp0 + vnorm * np.sin(self.phase - self.omega_rf * z1 / c)
        # Finally, apply        a        second        drift        through        ds / 2
        d1 = np.sqrt(1 - px0 * px0 - py0 * py0 + 2 * dp1 / beta0 + dp1 * dp1)
        x2 = x1 + ds * px0 / d1 / 2
        y2 = y1 + ds * py0 / d1 / 2
        z2 = z1 + ds * (1 - (1 + beta0 * dp1) / d1) / beta0 / 2
        beam.set_particle([x2, px0, y2, py0, z2, dp1])
        return beam

    def real_track(self, beam: Beam7) -> Beam7:
        [x0, px0, y0, py0, z0, delta0] = beam.get_particle()
        beta0 = RefParticle.beta
        ds = self.length
        # First        apply        a        drift        through        ds / 2
        d1 = np.sqrt(1 - px0 * px0 - py0 * py0 + 2 * delta0 / beta0 + delta0 * delta0)
        x1 = x0 + ds * px0 / d1 / 2
        y1 = y0 + ds * py0 / d1 / 2
        z1 = z0 + ds * (1 - (1 + beta0 * delta0) / d1) / beta0 / 2
        # Next, apply        an        rf        'kick'
        vnorm = self.voltage / RefParticle.energy
        delta1 = delta0 + vnorm * np.sin(self.phase - self.omega_rf * z1 / c)
        # px0 = px0 - vnorm * self.omega_rf * np.cos(self.phase) * x1 / 2 / c
        # py0 = py0 - vnorm * self.omega_rf * np.cos(self.phase) * y1 / 2 / c
        # Finally, apply        a        second        drift        through        ds / 2
        d1 = np.sqrt(1 - px0 * px0 - py0 * py0 + 2 * delta1 / beta0 + delta1 * delta1)
        x2 = x1 + ds * px0 / d1 / 2
        y2 = y1 + ds * py0 / d1 / 2
        z2 = z1 + ds * (1 - (1 + beta0 * delta1) / d1) / beta0 / 2
        # damping
        # dp0_div_dp1 = (delta0 * RefParticle.beta + 1) / (delta1 * RefParticle.beta + 1)
        # px0 = px0 * dp0_div_dp1
        # py0 = py0 * dp0_div_dp1
        beam.set_particle([x2, px0, y2, py0, z2, delta1])
        return beam
