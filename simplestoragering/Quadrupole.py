# -*- coding: utf-8 -*-
from .components import Element, assin_twiss, next_twiss
from .globalvars import Cr, RefParticle
from .exceptions import ParticleLost
import numpy as np


class Quadrupole(Element):
    """normal Quadrupole"""

    def __init__(self, name: str = None, length: float = 0, k1: float = 0, n_slices: int = 4):
        self.name = name
        self.length = length
        self.k1 = k1
        self.n_slices = n_slices

    def slice(self, n_slices: int) -> list:
        """slice component to element list, return [ele_list, final_z], the identifier identifies different magnet"""
        ele_list = []
        current_s = self.s
        length = self.length / n_slices
        twiss0 = np.array(
            [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp,
             self.etay, self.etayp, self.psix, self.psiy])
        for i in range(n_slices):
            ele = Quadrupole(self.name, length, self.k1)
            ele.s = current_s
            assin_twiss(ele, twiss0)
            twiss0 = next_twiss(ele.matrix, twiss0)
            ele_list.append(ele)
            current_s = current_s + ele.length
        return ele_list

    @property
    def matrix(self):
        return quad_matrix(self.length, self.k1)

    @property
    def damping_matrix(self):
        lambda_q = Cr * RefParticle.energy ** 3 * self.k1 ** 2 * self.length / np.pi
        matrix = self.matrix
        matrix[5, 5] = 1 - lambda_q * (self.closed_orbit[0] ** 2 + self.closed_orbit[2] ** 2)
        matrix[5, 0] = - lambda_q * self.closed_orbit[0]
        matrix[5, 2] = - lambda_q * self.closed_orbit[2]
        return matrix

    @property
    def closed_orbit_matrix(self):
        m67 = - Cr * RefParticle.energy ** 3 * self.k1 ** 2 * self.length * (self.closed_orbit[0] ** 2 +
                                                                             self.closed_orbit[2] ** 2) / 2 / np.pi
        # m67 = m67 * (1 + self.closed_orbit[5]) ** 2
        matrix7 = np.identity(7)
        matrix7[0:6, 0:6] = self.matrix
        matrix7[5, 6] = m67
        return matrix7

    def symplectic_track(self, beam):
        # [x0, px0, y0, py0, ct0, dp0] = beam.get_particle()
        [x0, px0, y0, py0, ct0, dp0] = beam

        beta0 = RefParticle.beta

        ds = self.length
        k1 = self.k1
        try:
            d1 = np.sqrt(1 + 2 * dp0 / beta0 + dp0 * dp0)
        except FloatingPointError:
            print(f'particle lost in {self.name} at {self.s}\n')
            raise ParticleLost(' just lost')
        w = np.sqrt(complex(k1, 0) / d1)
        w_2 = np.real(w ** 2)

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
        c11 = k1 * k1 * d2 * (xs2 / w - 2 * ds) / w_2 / 4
        c12 = -k1 * d2 * xs * xs / w_2
        c22 = d2 * (xs2 / w + 2 * ds) / 4
        c33 = k1 * k1 * d2 * (ys2 / w - 2 * ds) / w_2 / 4
        c34 = k1 * d2 * ys * ys / w / w
        c44 = d2 * (ys2 / w + 2 * ds) / 4

        ct1 = ct0 + c0 + c11 * x0 * x0 + c12 * x0 * px0 + c22 * px0 * px0 + c33 * y0 * y0 + c34 * y0 * py0 + c44 * py0 * py0

        # beam.set_particle([abs(x1), abs(px1), abs(y1), abs(py1), abs(ct1), abs(dp0)])
        # return beam
        return np.real(np.array([x1, px1, y1, py1, ct1, dp0]))

    def real_track(self, beam):
        n = int(self.length / 0.01)
        ds = self.length / n
        [x0, px0, y0, py0, z0, delta0] = beam
        for i in range(n):
            # drift
            # ds = self.length / 2
            try:
                d1 = np.sqrt(1 - px0 ** 2 - py0 ** 2 + 2 * delta0 / RefParticle.beta + delta0 ** 2)
            except Exception:
                raise ParticleLost(self.s)
            x1 = x0 + ds * px0 / d1 / 2
            y1 = y0 + ds * py0 / d1 / 2
            z1 = z0 + (ds / 2) * (1 - (1 + RefParticle.beta * delta0) / d1) / RefParticle.beta
            # kick
            px1 = px0 - self.k1 * x1 * ds
            py1 = py0 + self.k1 * y1 * ds
            # damping
            delta1 = (delta0 - (1 + delta0 * RefParticle.beta) ** 2 * Cr * RefParticle.energy ** 3 * self.k1 ** 2 *
                      ds * (x1 ** 2 + y1 ** 2) / 2 / np.pi / RefParticle.beta)
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
            x2 = x1 + (ds / 2) * px1 / d2
            y2 = y1 + (ds / 2) * py1 / d2
            z2 = z1 + (ds / 2) * (1 - (1 + RefParticle.beta * delta1) / d2) / RefParticle.beta
            x0 = x2
            px0 = px1
            y0 = y2
            py0 = py1
            z0 = z2
            delta0 = delta1
        # beam.set_particle([x2, px1, y2, py1, z2, delta1])
        return np.array([x2, px1, y2, py1, z2, delta1])

    def copy(self):
        return Quadrupole(name=self.name, length=self.length, k1=self.k1, n_slices=self.n_slices)

    def linear_optics(self):
        twiss0 = np.array(
            [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp,
             self.etay, self.etayp, self.psix, self.psiy])
        integrals = np.zeros(7)
        current_s = 0
        length = 0.01
        while current_s < self.length - length:
            matrix = quad_matrix(length, self.k1)
            twiss1 = next_twiss(matrix, twiss0)
            betax = (twiss0[0] + twiss1[0]) / 2
            betay = (twiss0[3] + twiss1[3]) / 2
            integrals[5] += - self.k1 * length * betax / 4 / np.pi
            integrals[6] += self.k1 * length * betay / 4 / np.pi
            current_s = current_s + length
            for i in range(len(twiss0)):
                twiss0[i] = twiss1[i]
        length = self.length - current_s
        matrix = quad_matrix(length, self.k1)
        twiss1 = next_twiss(matrix, twiss0)
        betax = (twiss0[0] + twiss1[0]) / 2
        betay = (twiss0[3] + twiss1[3]) / 2
        integrals[5] += - self.k1 * length * betax / 4 / np.pi
        integrals[6] += self.k1 * length * betay / 4 / np.pi
        return integrals, twiss1

    def driving_terms(self):
        twiss0 = np.array([self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy])
        h20001 = h00201 = h10002 = 0
        length = self.length / self.n_slices
        for i in range(self.n_slices):
            matrix = quad_matrix(length, self.k1)
            twiss1 = next_twiss(matrix, twiss0)
            betax = (twiss0[0] + twiss1[0]) / 2
            betay = (twiss0[3] + twiss1[3]) / 2
            etax = (twiss0[6] + twiss1[6]) / 2
            psix = (twiss0[10] + twiss1[10]) / 2
            psiy = (twiss0[11] + twiss1[11]) / 2

            h20001 += betax * np.exp(complex(0, 2 * psix))
            h00201 += betay * np.exp(complex(0, 2 * psiy))
            h10002 += betax ** 0.5 * etax * np.exp(complex(0, psix))

            twiss0 = twiss1
        h20001 = h20001 * length * self.k1 / 8
        h00201 = -h00201 * length * self.k1 / 8
        h10002 = h10002 * length * self.k1 / 2
        return np.array([h20001, h00201, h10002])

    def __repr__(self):
        return f"Quadrupole('{self.name}', length = {self.length}, k1 = {self.k1})"


def quad_matrix(length, k1):
    if k1 > 0:
        sqk = np.sqrt(k1)
        sqkl = sqk * length
        return np.array([[np.cos(sqkl), np.sin(sqkl) / sqk, 0, 0, 0, 0],
                         [- sqk * np.sin(sqkl), np.cos(sqkl), 0, 0, 0, 0],
                         [0, 0, np.cosh(sqkl), np.sinh(sqkl) / sqk, 0, 0],
                         [0, 0, sqk * np.sinh(sqkl), np.cosh(sqkl), 0, 0],
                         [0, 0, 0, 0, 1, length / RefParticle.gamma ** 2],
                         [0, 0, 0, 0, 0, 1]])
    elif k1 < 0:
        sqk = np.sqrt(-k1)
        sqkl = sqk * length
        return np.array([[np.cosh(sqkl), np.sinh(sqkl) / sqk, 0, 0, 0, 0],
                         [sqk * np.sinh(sqkl), np.cosh(sqkl), 0, 0, 0, 0],
                         [0, 0, np.cos(sqkl), np.sin(sqkl) / sqk, 0, 0],
                         [0, 0, - sqk * np.sin(sqkl), np.cos(sqkl), 0, 0],
                         [0, 0, 0, 0, 1, length / RefParticle.gamma ** 2],
                         [0, 0, 0, 0, 0, 1]])

    else:
        return np.array([[1, length, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, length, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, length / (RefParticle.gamma * RefParticle.beta) ** 2],
                         [0, 0, 0, 0, 0, 1]])
