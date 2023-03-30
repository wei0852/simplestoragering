# -*- coding: utf-8 -*-
from .components import Element, assin_twiss, next_twiss
from .globalvars import Cr, RefParticle
from .Drift import drift_matrix
from .exceptions import ParticleLost
import numpy as np


class Octupole(Element):
    """sextupole"""
    # symbol = 400

    def __init__(self, name: str = None, length: float = 0, k3: float = 0, n_slices: int = 1):
        self.name = name
        self.length = length
        self.k3 = k3
        self.n_slices = n_slices

    def slice(self, n_slices: int) -> list:
        """slice component to element list, return ele_list"""
        ele_list = []
        current_s = self.s
        length = self.length / n_slices
        twiss0 = np.array([self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy])
        for i in range(n_slices):
            ele = Octupole(self.name, length, self.k3)
            assin_twiss(ele, twiss0)
            ele.s = current_s
            twiss0 = next_twiss(ele.matrix, twiss0)
            ele_list.append(ele)
            current_s = current_s + ele.length
        return ele_list

    @property
    def matrix(self):
        return oct_matrix(self.length, self.k3, self.closed_orbit)

    @property
    def damping_matrix(self):
        """I think damping is not in thin len approximation"""
        raise Exception('Unfinished, Octupole damping matrix.')

    @property
    def closed_orbit_matrix(self):
        """it's different from its transform matrix, x is replaced by closed orbit x0"""
        raise Exception('Unfinished, Octupole closed orbit matrix')

    def symplectic_track(self, beam):
        [x0, px0, y0, py0, ct0, dp0] = beam

        beta0 = RefParticle.beta

        ds = self.length / self.n_slices
        k3 = self.k3
        for i in range(self.n_slices):
            d1 = np.sqrt(1 - px0 * px0 - py0 * py0 + 2 * dp0 / beta0 + dp0 * dp0)

            x1 = x0 + ds * px0 / d1 / 2
            y1 = y0 + ds * py0 / d1 / 2
            ct1 = ct0 + ds * (1 - (1 + beta0 * dp0) / d1) / beta0 / 2

            px1 = px0 - (x1 ** 3 / 3 - x1 * y1 ** 2) * k3 * ds / 2
            py1 = py0 - (y1 ** 3 / 3 - y1 * x1 ** 2) * k3 * ds / 2
            try:
                d1 = np.sqrt(1 - px1 * px1 - py1 * py1 + 2 * dp0 / beta0 + dp0 * dp0)
            except FloatingPointError:
                print(f'particle lost in {self.name} at {self.s + i * ds}\n')
                raise ParticleLost(' just lost')
            x2 = x1 + ds * px1 / d1 / 2
            y2 = y1 + ds * py1 / d1 / 2
            ct2 = ct1 + ds * (1 - (1 + beta0 * dp0) / d1) / beta0 / 2
            x0 = x2
            px0 = px1
            y0 = y2
            py0 = py1
            ct0 = ct2
        return np.array([x2, px1, y2, py1, ct2, dp0])

    def real_track(self, beam):
        raise Exception('Unfinished, Octupole real track')
        # [x0, px0, y0, py0, z0, delta0] = beam
        #
        # ds = self.length / self.n_slices
        # k3 = self.k3
        # for i in range(self.n_slices):
        # # drift
        #     try:
        #         d1 = np.sqrt(1 - px0 ** 2 - py0 ** 2 + 2 * delta0 / RefParticle.beta + delta0 ** 2)
        #     except Exception:
        #         raise ParticleLost(self.s)
        #     x1 = x0 + ds * px0 / d1 / 2
        #     y1 = y0 + ds * py0 / d1 / 2
        #     z1 = z0 + ds * (1 - (1 + RefParticle.beta * delta0) / d1) / RefParticle.beta / 2
        # # kick
        #     px1 = px0 - (x1 * x1 - y1 * y1) * k3 * ds / 2
        #     py1 = py0 + x1 * y1 * k3 * ds
        # # damping, when beta approx 1
        # # delta1 = delta0 - (delta0 + 1) ** 2 * (Cr * RefParticle.energy ** 3 * self.k3 ** 2 * self.length *
        # #                                        (x1 ** 2 + y1 ** 2) ** 2 / 8 / pi)     # beta0 \approx 1
        #     delta1 = (delta0 - (delta0 * RefParticle.beta + 1) ** 2 * Cr * RefParticle.energy ** 3 * self.k3 ** 2 *
        #               self.length * (x1 ** 2 + y1 ** 2) ** 2 / 8 / np.pi / RefParticle.beta)
        # # e1_div_e0 = (delta1 + 1) / (delta0 + 1)  # approximation
        #     e1_div_e0 = np.sqrt(((1 + delta1 * RefParticle.beta) ** 2 - 1 / RefParticle.gamma ** 2) /
        #                         ((1 + delta0 * RefParticle.beta) ** 2 - 1 / RefParticle.gamma ** 2))
        #     px1 = px1 * e1_div_e0
        #     py1 = py1 * e1_div_e0
        # # drift
        #     try:
        #         d2 = np.sqrt(1 - px1 ** 2 - py1 ** 2 + 2 * delta1 / RefParticle.beta + delta1 ** 2)
        #     except Exception:
        #         raise ParticleLost(self.s)
        #     x2 = x1 + ds * px1 / d2 / 2
        #     y2 = y1 + ds * py1 / d2 / 2
        #     z2 = z1 + ds * (1 - (1 + RefParticle.beta * delta1) / d2) / RefParticle.beta / 2
        #     x0 = x2
        #     px0 = px1
        #     y0 = y2
        #     py0 = py1
        #     z0 = z2
        # # beam.set_particle()
        # return np.array([x2, px1, y2, py1, z2, delta1])

    def copy(self):
        return Octupole(self.name, self.length, self.k3, self.n_slices)

    def linear_optics(self):
        twiss0 = np.array([self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy])
        # integrals = np.zeros(7)
        # current_s = 0
        # length = 0.01
        # orbit = self.closed_orbit
        # while current_s < self.length - length:
        #     matrix = oct_matrix(length, self.k3, orbit)
        #     twiss1 = next_twiss(matrix, twiss0)
        #     orbit = matrix.dot(orbit)
        #     current_s = current_s + length
        #     for i in range(len(twiss0)):
        #         twiss0[i] = twiss1[i]
        # length = self.length - current_s
        # matrix = oct_matrix(length, self.k3, orbit)
        # twiss1 = next_twiss(matrix, twiss0)
        twiss1 = next_twiss(self.matrix, twiss0)
        return np.zeros(7), twiss1

    def driving_terms(self):
        twiss0 = np.array([self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy])
        h22000 = h11110 = h00220 = 0
        h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
        length = self.length / self.n_slices
        b4l = length * self.k3 / 6
        for i in range(self.n_slices):
            matrix = drift_matrix(length)
            twiss1 = next_twiss(matrix, twiss0)
            betax = (twiss0[0] + twiss1[0]) / 2
            betay = (twiss0[3] + twiss1[3]) / 2
            psix = (twiss0[10] + twiss1[10]) / 2
            psiy = (twiss0[11] + twiss1[11]) / 2

            h22000 += betax ** 2
            h11110 += betax * betay
            h00220 += betay ** 2

            h31000 += betax ** 2 * np.exp(complex(0, 2 * psix))
            h40000 += betax ** 2 * np.exp(complex(0, 4 * psix))
            h20110 += betax * betay * np.exp(complex(0, 2 * psix))
            h11200 += betax * betay * np.exp(complex(0, 2 * psiy))
            h20020 += betax * betay * np.exp(complex(0, 2 * psix - 2 * psiy))
            h20200 += betax * betay * np.exp(complex(0, 2 * psix + 2 * psiy))
            h00310 += betay ** 2 * np.exp(complex(0, 2 * psiy))
            h00400 += betay ** 2 * np.exp(complex(0, 4 * psiy))
            twiss0 = twiss1
        h31000 =  -b4l * h31000 / 16
        h40000 =  -b4l * h40000 / 64
        h20110 =  3 * b4l * h20110 / 16
        h11200 =  3 * b4l * h11200 / 16
        h20020 =  3 * b4l * h20020 / 32
        h20200 =  3 * b4l * h20200 / 32
        h00310 =  -b4l * h00310 / 16
        h00400 =  -b4l * h00400 / 64
        h22000 = -3 * b4l * h22000 / 32
        h11110 =  3 * b4l * h11110 / 8
        h00220 = -3 * b4l * h00220 / 32
        return np.array([h22000, h11110, h00220, h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400])

    def __repr__(self):
        return f"Sextupole('{self.name}', length = {self.length}, k3 = {self.k3}, n_slices = {self.n_slices})"


def oct_matrix(length, k3, closed_orbit):
    # k3l = k3 * length
    # x0 = closed_orbit[0]
    # y0 = closed_orbit[2]
    # x02_y02_2 = (x0 ** 2 - y0 ** 2) / 2  # (x0 ** 2 - y0 ** 2) / 2
    # matrix = np.array([[1, 0, 0, 0, 0, 0],
    #                    [- k3l * x0, 1, k3l * y0, 0, 0, k3l * x02_y02_2],
    #                    [0, 0, 1, 0, 0, 0],
    #                    [k3l * y0, 0, k3l * x0, 1, 0, - k3l * x0 * y0],
    #                    [- k3l * x02_y02_2, 0, k3l * x0 * y0, 0, 1, 0],
    #                    [0, 0, 0, 0, 0, 1]])
    drift = drift_matrix(length=length)
    return drift
