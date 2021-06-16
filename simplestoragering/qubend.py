from .components import Element
from .constants import Cr, pi
from .particles import RefParticle, Beam7
from .drift import Drift
from .exceptions import UnfinishedWork
import numpy as np


class QuBend(Element):
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
        raise UnfinishedWork()
        # if self.k1 > 0:
        #     sqk = np.sqrt(self.k1)
        #     sqkl = sqk * self.length
        #     return np.array([[np.cos(sqkl), np.sin(sqkl) / sqk, 0, 0, 0, 0],
        #                      [- sqk * np.sin(sqkl), np.cos(sqkl), 0, 0, 0, 0],
        #                      [0, 0, np.cosh(sqkl), np.sinh(sqkl) / sqk, 0, 0],
        #                      [0, 0, sqk * np.sinh(sqkl), np.cosh(sqkl), 0, 0],
        #                      [0, 0, 0, 0, 1, self.length / RefParticle.gamma ** 2],
        #                      [0, 0, 0, 0, 0, 1]])
        # else:
        #     sqk = np.sqrt(-self.k1)
        #     sqkl = sqk * self.length
        #     return np.array([[np.cosh(sqkl), np.sinh(sqkl) / sqk, 0, 0, 0, 0],
        #                      [sqk * np.sinh(sqkl), np.cosh(sqkl), 0, 0, 0, 0],
        #                      [0, 0, np.cos(sqkl), np.sin(sqkl) / sqk, 0, 0],
        #                      [0, 0, - sqk * np.sin(sqkl), np.cos(sqkl), 0, 0],
        #                      [0, 0, 0, 0, 1, self.length / RefParticle.gamma ** 2],
        #                      [0, 0, 0, 0, 0, 1]])

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
        if self.k1 > 0:
            return self.__track_fq(beam)
        elif self.k1 < 0:
            return self.__track_dq(beam)
        else:
            return Drift(length=self.length).symplectic_track(beam)

    def __track_fq(self, beam):
        [x0, px0, y0, py0, ct0, dp0] = beam.get_particle()

        beta0 = RefParticle.beta

        ds = self.length
        k1 = self.k1

        d1 = np.sqrt(1 + 2 * dp0 / beta0 + dp0 * dp0)
        w = np.sqrt(k1 / d1)

        xs = np.sin(w * ds)
        xc = np.cos(w * ds)
        ys = np.sinh(w * ds)
        yc = np.cosh(w * ds)
        xs2 = np.sin(2 * w * ds)
        ys2 = np.sinh(2 * w * ds)

        x1 = x0 * xc + px0 * xs * w / k1
        px1 = -k1 * x0 * xs / w + px0 * xc
        y1 = y0 * yc + py0 * ys * w / k1
        py1 = k1 * y0 * ys / w + py0 * yc

        d0 = 1 / beta0 + dp0
        d2 = -d0 / d1 / d1 / d1 / 2

        c0 = (1 / beta0 - d0 / d1) * ds
        c11 = k1 * k1 * d2 * (xs2 / w - 2 * ds) / w / w / 4
        c12 = -k1 * d2 * xs * xs / w / w
        c22 = d2 * (xs2 / w + 2 * ds) / 4
        c33 = k1 * k1 * d2 * (ys2 / w - 2 * ds) / w / w / 4
        c34 = k1 * d2 * ys * ys / w / w
        c44 = d2 * (ys2 / w + 2 * ds) / 4

        ct1 = (ct0 + c0 + c11 * x0 * x0 + c12 * x0 * px0 + c22 * px0 * px0 + c33 * y0 * y0 + c34 * y0 * py0 +
               c44 * py0 * py0)

        beam.set_particle([x1, px1, y1, py1, ct1, dp0])
        return beam

    def __track_dq(self, beam):
        [x0, px0, y0, py0, ct0, dp0] = beam.get_particle()

        beta0 = RefParticle.beta

        ds = self.length
        k1 = self.k1

        d1 = np.sqrt(1 + 2 * dp0 / beta0 + dp0 * dp0)
        w = np.sqrt(abs(k1) / d1)

        xs = np.sinh(w * ds)
        xc = np.cosh(w * ds)
        ys = np.sin(w * ds)
        yc = np.cos(w * ds)
        xs2 = np.sinh(2 * w * ds)

        ys2 = np.sin(2 * w * ds)

        x1 = x0 * xc + px0 * xs * w / abs(k1)
        px1 = -k1 * x0 * xs / w + px0 * xc
        y1 = y0 * yc + py0 * ys * w / abs(k1)
        py1 = k1 * y0 * ys / w + py0 * yc

        d0 = 1 / beta0 + dp0
        d2 = -d0 / d1 / d1 / d1 / 2
        c0 = (1 / beta0 - d0 / d1) * ds
        c11 = k1 * k1 * d2 * (xs2 / w - 2 * ds) / w / w / 4
        c12 = -k1 * d2 * xs * xs / w / w
        c22 = d2 * (xs2 / w + 2 * ds) / 4
        c33 = k1 * k1 * d2 * (ys2 / w - 2 * ds) / w / w / 4
        c34 = k1 * d2 * ys * ys / w / w
        c44 = d2 * (ys2 / w + 2 * ds) / 4
        ct1 = (ct0 + c0 + c11 * x0 * x0 + c12 * x0 * px0 + c22 * px0 * px0 + c33 * y0 * y0 + c34 * y0 * py0 +
               c44 * py0 * py0)
        beam.set_particle([x1, px1, y1, py1, ct1, dp0])
        return beam

    def real_track(self, beam: Beam7) -> Beam7:
        [x0, px0, y0, py0, z0, delta0] = beam.get_particle()
        delta1 = (delta0 - (delta0 + 1) ** 2 * Cr * RefParticle.energy ** 3 * self.k1 ** 2 * self.length *
                  (x0 ** 2 + y0 ** 2) / 2 / pi)
        beam.set_particle([x0, px0, y0, py0, z0, (delta0 + delta1) / 2])
        beam = self.symplectic_track(beam)
        [x0, px0, y0, py0, z0, delta01] = beam.get_particle()
        dp1_div_dp0 = (delta1 * RefParticle.beta + 1) / (delta0 * RefParticle.beta + 1)
        px0 = px0 * dp1_div_dp0
        py0 = py0 * dp1_div_dp0
        beam.set_particle([x0, px0, y0, py0, z0, delta1])
        return beam
