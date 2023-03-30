# -*- coding: utf-8 -*-
from .components import Element, assin_twiss, next_twiss
from .globalvars import Cr, RefParticle
from .Drift import drift_matrix
from .exceptions import ParticleLost
import numpy as np


class Sextupole(Element):
    """sextupole"""
    # symbol = 400

    def __init__(self, name: str = None, length: float = 0, k2: float = 0, n_slices: int = 1):
        self.name = name
        self.length = length
        self.k2 = k2
        self.n_slices = n_slices

    def slice(self, n_slices: int) -> list:
        """slice component to element list, return [ele_list, final_z], the identifier identifies different magnet"""
        ele_list = []
        current_s = self.s
        length = self.length / n_slices
        twiss0 = np.array([self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy])
        for i in range(n_slices - 1):
            ele = Sextupole(self.name, length, self.k2)
            assin_twiss(ele, twiss0)
            ele.s = current_s
            twiss0 = next_twiss(ele.matrix, twiss0)
            ele_list.append(ele)
            current_s = current_s + ele.length
        length = self.length + self.s - current_s
        ele = Sextupole(self.name, length, self.k2)
        ele.s = current_s
        assin_twiss(ele, twiss0)
        ele_list.append(ele)
        return ele_list

    @property
    def matrix(self):
        return sext_matrix(self.length, self.k2, self.closed_orbit)

    @property
    def damping_matrix(self):
        """I think damping is not in thin len approximation"""
        k2l = self.k2 * self.length
        x0 = self.closed_orbit[0]
        y0 = self.closed_orbit[2]
        lambda_s = Cr * RefParticle.energy ** 3 * k2l ** 2 * (x0 ** 2 + y0 ** 2) / np.pi / self.length
        matrix = self.matrix
        matrix[5, 0] = - lambda_s * x0
        matrix[5, 2] = - lambda_s * y0
        matrix[5, 5] = 1 - lambda_s * (x0 ** 2 + y0 ** 2) / 2
        return matrix

    @property
    def closed_orbit_matrix(self):
        """it's different from its transform matrix, x is replaced by closed orbit x0"""
        m67 = - (Cr * RefParticle.energy ** 3 * self.k2 ** 2 * self.length *
                 (self.closed_orbit[0] ** 2 + self.closed_orbit[2] ** 2) ** 2 / 8 / np.pi)
        # m67 = m67 * (1 + self.closed_orbit[5]) ** 2
        matrix7 = np.identity(7)
        # drift = Drift(length=self.length / 2).matrix
        drift = drift_matrix(self.length / 2)
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
        # [x0, px0, y0, py0, ct0, dp0] = beam.get_particle()
        [x0, px0, y0, py0, ct0, dp0] = beam

        beta0 = RefParticle.beta

        ds = self.length / self.n_slices
        k2 = self.k2
        for i in range(self.n_slices):
            d1 = np.sqrt(1 - px0 * px0 - py0 * py0 + 2 * dp0 / beta0 + dp0 * dp0)

            x1 = x0 + ds * px0 / d1 / 2
            y1 = y0 + ds * py0 / d1 / 2
            ct1 = ct0 + ds * (1 - (1 + beta0 * dp0) / d1) / beta0 / 2

            px1 = px0 - (x1 * x1 - y1 * y1) * k2 * ds / 2
            py1 = py0 + x1 * y1 * k2 * ds
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
        # beam.set_particle([x2, px1, y2, py1, ct2, dp0])
        # return beam
        return np.array([x2, px1, y2, py1, ct2, dp0])

    def real_track(self, beam):
        [x0, px0, y0, py0, z0, delta0] = beam

        ds = self.length / self.n_slices
        k2 = self.k2
        for i in range(self.n_slices):
        # drift
            try:
                d1 = np.sqrt(1 - px0 ** 2 - py0 ** 2 + 2 * delta0 / RefParticle.beta + delta0 ** 2)
            except Exception:
                raise ParticleLost(self.s)
            x1 = x0 + ds * px0 / d1 / 2
            y1 = y0 + ds * py0 / d1 / 2
            z1 = z0 + ds * (1 - (1 + RefParticle.beta * delta0) / d1) / RefParticle.beta / 2
        # kick
            px1 = px0 - (x1 * x1 - y1 * y1) * k2 * ds / 2
            py1 = py0 + x1 * y1 * k2 * ds
        # damping, when beta approx 1
        # delta1 = delta0 - (delta0 + 1) ** 2 * (Cr * RefParticle.energy ** 3 * self.k2 ** 2 * self.length *
        #                                        (x1 ** 2 + y1 ** 2) ** 2 / 8 / pi)     # beta0 \approx 1
            delta1 = (delta0 - (delta0 * RefParticle.beta + 1) ** 2 * Cr * RefParticle.energy ** 3 * self.k2 ** 2 *
                      self.length * (x1 ** 2 + y1 ** 2) ** 2 / 8 / np.pi / RefParticle.beta)
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
            x2 = x1 + ds * px1 / d2 / 2
            y2 = y1 + ds * py1 / d2 / 2
            z2 = z1 + ds * (1 - (1 + RefParticle.beta * delta1) / d2) / RefParticle.beta / 2
            x0 = x2
            px0 = px1
            y0 = y2
            py0 = py1
            z0 = z2
        # beam.set_particle()
        return np.array([x2, px1, y2, py1, z2, delta1])

    def copy(self):
        return Sextupole(self.name, self.length, self.k2, self.n_slices)

    def linear_optics(self):
        twiss0 = np.array([self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy])
        integrals = np.zeros(7)
        current_s = 0
        length = 0.01
        orbit = self.closed_orbit
        while current_s < self.length - length:
            matrix = sext_matrix(length, self.k2, orbit)
            twiss1 = next_twiss(matrix, twiss0)
            orbit = matrix.dot(orbit)
            betax = (twiss0[0] + twiss1[0]) / 2
            betay = (twiss0[3] + twiss1[3]) / 2
            etax = (twiss0[6] + twiss1[6]) / 2
            integrals[5] += etax * self.k2 * length * betax / 4 / np.pi
            integrals[6] += - etax * self.k2 * length * betay / 4 / np.pi
            current_s = current_s + length
            for i in range(len(twiss0)):
                twiss0[i] = twiss1[i]
        length = self.length - current_s
        matrix = sext_matrix(length, self.k2, orbit)
        twiss1 = next_twiss(matrix, twiss0)
        betax = (twiss0[0] + twiss1[0]) / 2
        betay = (twiss0[3] + twiss1[3]) / 2
        etax = (twiss0[6] + twiss1[6]) / 2
        integrals[5] += etax * self.k2 * length * betax / 4 / np.pi
        integrals[6] += - etax * self.k2 * length * betay / 4 / np.pi
        return integrals, twiss1

    def driving_terms(self):
        twiss0 = np.array([self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp, self.etay, self.etayp, self.psix, self.psiy])
        jj = complex(0, 1)
        h20001 = h00201 = h10002 = h21000 = h30000 = h10110 = h10020 = h10200 = 0
        h22000 = h11110 = h00220 = h31000 = h40000 = h20110 = h11200 = h20020 = h20200 = h00310 = h00400 = 0
        length = self.length / self.n_slices
        b3l = length * self.k2 / 2
        for i in range(self.n_slices):
            matrix = drift_matrix(length)
            twiss1 = next_twiss(matrix, twiss0)
            betax = (twiss0[0] + twiss1[0]) / 2
            betay = (twiss0[3] + twiss1[3]) / 2
            etax = (twiss0[6] + twiss1[6]) / 2
            psix = (twiss0[10] + twiss1[10]) / 2
            psiy = (twiss0[11] + twiss1[11]) / 2

            h20001 += betax * etax * np.exp((complex(0, 2 * psix)))
            h00201 += betay * etax * np.exp(complex(0, 2 * psiy))
            h10002 += betax ** 0.5 * etax ** 2 * np.exp(complex(0, psix))

            h21000j = betax ** 1.5 * np.exp(complex(0, psix))
            h30000j = betax ** 1.5 * np.exp(complex(0, 3 * psix))
            h10110j = betax ** 0.5 * betay * np.exp(complex(0, psix))
            h10020j = betax ** 0.5 * betay * np.exp(complex(0, psix - 2 * psiy))
            h10200j = betax ** 0.5 * betay * np.exp(complex(0, psix + 2 * psiy))

            h12000j = h21000j.conjugate()
            h01110j = h10110j.conjugate()
            h01200j = h10020j.conjugate()
            h22000 += ((h21000 * h12000j - h21000.conjugate() * h21000j) * 3
                        +(h30000 * h30000j.conjugate() - h30000.conjugate() * h30000j))
            h11110 += (-(h21000 * h01110j - h10110.conjugate() * h21000j)
                        +(h21000.conjugate() * h10110j - h10110 * h12000j)
                        -(h10020 * h01200j - h10020.conjugate() * h10020j)
                        +(h10200 * h10200j.conjugate() - h10200.conjugate() * h10200j))
            h00220 += ((h10020 * h01200j - h10020.conjugate() * h10020j)
                        +(h10200 * h10200j.conjugate() - h10200.conjugate() * h10200j)
                        +(h10110 * h01110j - h10110.conjugate() * h10110j) * 4)
            h31000 += (h30000 * h12000j - h21000.conjugate() * h30000j)
            h40000 += (h30000 * h21000j - h21000 * h30000j)
            h20110 += (-(h30000 * h01110j - h10110.conjugate() * h30000j)
                        +(h21000 * h10110j - h10110 * h21000j)
                            +(h10200 * h10020j - h10020 * h10200j) * 2)
            h11200 += (-(h10200 * h12000j - h21000.conjugate() * h10200j)
                            -(h21000 * h01200j - h10020.conjugate() * h21000j)
                            +(h10200 * h01110j - h10110.conjugate() * h10200j) * 2
                            -(h10110 * h01200j - h10020.conjugate() * h10110j) * 2)
            h20020 += ((h21000 * h10020j - h10020 * h21000j)
                        -(h30000 * h10200j.conjugate() - h10200.conjugate() * h30000j)
                        +(h10110 * h10020j - h10020 * h10110j) * 4)
            h20200 += (-(h30000 * h01200j - h10020.conjugate() * h30000j)
                        -(h10200 * h21000j - h21000 * h10200j)
                        -(h10110 * h10200j - h10200 * h10110j) * 4)
            h00310 += ((h10200 * h01110j - h10110.conjugate() * h10200j)
                        +(h10110 * h01200j - h10020.conjugate() * h10110j))
            h00400 += (h10200 * h01200j - h10020.conjugate() * h10200j)
            h21000 += h21000j
            h30000 += h30000j
            h10110 += h10110j
            h10020 += h10020j
            h10200 += h10200j
            twiss0 = twiss1
        h20001 = -h20001 * b3l/ 4
        h00201 = h00201 * b3l / 4
        h10002 = -h10002 * b3l / 2
        h21000 = - h21000 * b3l / 8
        h30000 = - h30000 * b3l / 24
        h10110 = h10110 * b3l / 4
        h10020 = h10020 * b3l / 8
        h10200 = h10200 * b3l / 8
        h22000 = jj * b3l ** 2  * h22000 / 64
        h11110 = jj * b3l ** 2  * h11110 / 16
        h00220 = jj * b3l ** 2  * h00220 / 64
        h31000 = jj * b3l ** 2  * h31000 / 32
        h40000 = jj * b3l ** 2  * h40000 / 64
        h20110 = jj * b3l ** 2  * h20110 / 32
        h11200 = jj * b3l ** 2  * h11200 / 32
        h20020 = jj * b3l ** 2  * h20020 / 64
        h20200 = jj * b3l ** 2  * h20200 / 64
        h00310 = jj * b3l ** 2  * h00310 / 32
        h00400 = jj * b3l ** 2  * h00400 / 64
        return np.array([h21000, h30000, h10110, h10020, h10200, h20001, h00201, h10002,
                         h22000, h11110, h00220, h31000, h40000, h20110, h11200, h20020, h20200, h00310, h00400])

    def __repr__(self):
        return f"Sextupole('{self.name}', length = {self.length}, k2 = {self.k2}, n_slices = {self.n_slices})"


def sext_matrix(length, k2, closed_orbit):
    k2l = k2 * length
    x0 = closed_orbit[0]
    y0 = closed_orbit[2]
    x02_y02_2 = (x0 ** 2 - y0 ** 2) / 2  # (x0 ** 2 - y0 ** 2) / 2
    matrix = np.array([[1, 0, 0, 0, 0, 0],
                       [- k2l * x0, 1, k2l * y0, 0, 0, k2l * x02_y02_2],
                       [0, 0, 1, 0, 0, 0],
                       [k2l * y0, 0, k2l * x0, 1, 0, - k2l * x0 * y0],
                       [- k2l * x02_y02_2, 0, k2l * x0 * y0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])
    drift = drift_matrix(length=length / 2)
    return drift.dot(matrix).dot(drift)
