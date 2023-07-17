# -*- coding: utf-8 -*-
from .components import Element
from .globalvars import pi, c, RefParticle
import numpy as np


class RFCavity(Element):
    """thin len approximation, the length is 0. The unit of voltage should be the same as RefParticle.energy, MeV"""
    length = 0

    def __init__(self, name: str = None, voltage_in_MeV: float = 0, frequency: float = 0, phase: float = 0):
        self.name = name
        self.voltage = voltage_in_MeV
        self.f_rf = frequency
        self.phase = phase
        self.omega_rf = 2 * pi * self.f_rf

    @property
    def matrix(self):
        temp_val = 2 * pi * self.f_rf / RefParticle.beta / c  # h / R
        matrix = np.identity(6)
        matrix[5, 4] = (self.voltage * temp_val * np.cos(self.phase) / RefParticle.energy)
        return matrix

    def copy(self):
        return self

    def slice(self, n_slices: int) -> list:
        return [self]

    def symplectic_track(self, beam):
        """rf cavity tracking is simplified by thin len approximation"""
        [x0, px0, y0, py0, z0, dp0] = beam
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
        return np.array([x2, px0, y2, py0, z2, dp1])

    def radiation_integrals(self):
        return 0, 0, 0, 0, 0, 0, 0
