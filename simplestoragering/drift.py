# -*- coding: utf-8 -*-
from .components import Element
from .particles import RefParticle, Beam7
import numpy as np


class Drift(Element):
    """drift class"""
    symbol = 100

    def __init__(self, name: str = None, length: float = 0.0, n_slices: int = 1):
        self.name = name
        self.length = length
        self.n_slices = n_slices

    @property
    def matrix(self):
        return np.array([[1, self.length, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, self.length, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, self.length / RefParticle.gamma ** 2],
                         [0, 0, 0, 0, 0, 1]])

    @property
    def damping_matrix(self):
        return self.matrix

    @property
    def closed_orbit_matrix(self):
        matrix = self.matrix
        matrix7 = np.identity(7)
        matrix7[0:6, 0:6] = matrix
        return matrix7

    def symplectic_track(self, beam):
        assert isinstance(beam, Beam7)
        [x0, px0, y0, py0, z0, delta0] = beam.get_particle()
        ds = self.length
        d1 = np.sqrt(1 - px0 ** 2 - py0 ** 2 + 2 * delta0 / RefParticle.beta + delta0 ** 2)
        x1 = x0 + ds * px0 / d1
        y1 = y0 + ds * py0 / d1
        z1 = z0 + ds * (1 - (1 + RefParticle.beta * delta0) / d1) / RefParticle.beta
        beam.set_particle([x1, px0, y1, py0, z1, delta0])
        return beam

    def real_track(self, beam: Beam7) -> Beam7:
        return self.symplectic_track(beam)

    def radiation_integrals(self):
        return 0, 0, 0, 0, 0, 0, 0
