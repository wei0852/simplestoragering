# -*- coding: utf-8 -*-
from .components import Element, assin_twiss, next_twiss
from .exceptions import ParticleLost
from .globalvars import RefParticle
import numpy as np


class Drift(Element):
    """drift class"""

    def __init__(self, name: str = None, length: float = 0.0):
        self.name = name
        self.length = length

    def slice(self, n_slices: int) -> list:
        """slice component to element list, return ele_list"""
        ele_list = []
        current_s = self.s
        length = self.length / n_slices
        twiss0 = np.array([self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy])
        for i in range(n_slices):
            ele = Drift(self.name, length)
            ele.s = current_s
            assin_twiss(ele, twiss0)
            twiss0 = next_twiss(ele.matrix, twiss0)
            ele_list.append(ele)
            current_s = current_s + ele.length
        return ele_list

    def linear_optics(self):
        twiss0 = np.array([self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy])
        twiss1 = next_twiss(self.matrix, twiss0)
        return np.zeros(7), twiss1

    @property
    def matrix(self):
        return drift_matrix(self.length)

    def symplectic_track(self, beam):
        [x0, px0, y0, py0, z0, delta0] = beam
        ds = self.length
        np.seterr(all='raise')
        try:
            d1 = np.sqrt(1 - px0 ** 2 - py0 ** 2 + 2 * delta0 / RefParticle.beta + delta0 ** 2)
        except FloatingPointError:
            raise ParticleLost(self.s)
        x1 = x0 + ds * px0 / d1
        y1 = y0 + ds * py0 / d1
        z1 = z0 + ds * (1 - (1 + RefParticle.beta * delta0) / d1) / RefParticle.beta
        return np.array([x1, px0, y1, py0, z1, delta0])

    def copy(self):
        return Drift(self.name, self.length)

    def __repr__(self):
        return f"Drift('{self.name}', length = {self.length})"


def drift_matrix(length):
    return np.array([[1, length, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, length, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, length / (RefParticle.gamma * RefParticle.beta) ** 2],
                     [0, 0, 0, 0, 0, 1]])
