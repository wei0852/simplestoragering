from .components import Element
from .constants import Cr, LENGTH_PRECISION, pi
from .particles import RefParticle, Beam7
from .exceptions import ParticleLost
from copy import deepcopy
import numpy as np


class HBend(Element):
    """horizontal Bend"""
    symbol = 200

    def __init__(self, name: str = None, length: float = 0, theta: float = 0, theta_in: float = 0, theta_out: float = 0,
                 n_slices: int = 3):
        self.name = name
        self.length = length
        self.h = theta / self.length
        self.theta_in = theta_in
        self.theta_out = theta_out
        self.n_slices = n_slices

    @property
    def theta(self):
        return self.h * self.length

    def set_slices(self, n_slices):
        self.n_slices = n_slices

    @property
    def matrix(self):
        cx = np.cos(self.theta)
        h_beta = self.h * RefParticle.beta
        sin_theta = np.sin(self.theta)
        inlet_edge = np.array([[1, 0, 0, 0, 0, 0],
                               [np.tan(self.theta_in) * self.h, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, -np.tan(self.theta_in) * self.h, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]])
        # middle_section = np.array([[cx, sin_theta / self.h, 0, 0, 0, (1 - cx) / h_beta],
        #                            [-sin_theta * self.h, cx, 0, 0, 0, sin_theta / RefParticle.beta],
        #                            [0, 0, 1, self.length, 0, 0],
        #                            [0, 0, 0, 1, 0, 0],
        #                            [- np.sin(self.theta), - (1 - cx) / h_beta, 0, 0, 1,
        #                             - self.length + sin_theta / h_beta / RefParticle.beta],
        #                            [0, 0, 0, 0, 0, 1]])
        # according to elegant result
        middle_section = np.array([[cx, sin_theta / self.h, 0, 0, 0, (1 - cx) / h_beta],
                                   [-sin_theta * self.h, cx, 0, 0, 0, sin_theta / RefParticle.beta],
                                   [0, 0, 1, self.length, 0, 0],
                                   [0, 0, 0, 1, 0, 0],
                                   [np.sin(self.theta), (1 - cx) / h_beta, 0, 0, 1,
                                    self.length - sin_theta / h_beta / RefParticle.beta],
                                   [0, 0, 0, 0, 0, 1]])
        outlet_edge = np.array([[1, 0, 0, 0, 0, 0],
                                [np.tan(self.theta_out) * self.h, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, -np.tan(self.theta_out) * self.h, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])
        return outlet_edge.dot(middle_section).dot(inlet_edge)

    @property
    def damping_matrix(self):
        m66 = 1 - Cr * RefParticle.energy ** 3 * self.length * self.h ** 2 / pi
        # delta_delta = - Cr * RefParticle.energy ** 3 * self.length * self.h ** 2 / pi / 2
        matrix = self.matrix
        matrix[5, 5] = m66
        # matrix[1, 5] = matrix[1, 5] * (1 + delta_delta / self.closed_orbit[5] / 2)
        return matrix

    @property
    def closed_orbit_matrix(self):
        # m67 = -(self.closed_orbit[5] + 1) ** 2 * Cr * RefParticle.energy ** 3 * self.theta ** 2 / 2 / pi / self.length
        m67 = -Cr * RefParticle.energy ** 3 * self.theta ** 2 / 2 / pi / self.length
        matrix7 = np.identity(7)
        matrix7[0:6, 0:6] = self.matrix  # Bend doesn't use thin len approximation, so
        matrix7[5, 6] = m67
        matrix7[1, 6] = self.h * self.length * m67 / 2
        return matrix7

    def symplectic_track(self, beam):
        [x0, px0, y0, py0, z0, delta0] = beam.get_particle()
        d1 = np.sqrt(1 + 2 * delta0 / RefParticle.beta + delta0 ** 2)
        ds = self.length
        # entrance
        px1 = px0 + self.h * np.tan(self.theta_in) * x0
        py1 = py0 - self.h * np.tan(self.theta_in) * y0
        # drift
        h = self.h
        k1 = self.k1
        a1 = h - self.h / d1

        wx = np.sqrt((h * self.h + k1) / d1)
        xc = np.cos(wx * ds)
        xs = np.sin(wx * ds) / wx
        xs2 = np.sin(2 * wx * ds) / wx

        wy = np.sqrt(k1 / d1)
        yc = np.cosh(wy * ds)
        ys = ds
        ys2 = 2 * ds

        if wy.any():
            ys = np.sinh(wy * ds) / wy
            ys2 = np.sinh(2 * wy * ds) / wy

        x2 = x0 * xc + px1 * xs / d1 + a1 * (1 - xc) / wx / wx
        px2 = -d1 * wx * wx * x0 * xs + px1 * xc + a1 * xs * d1

        y2 = y0 * yc + py1 * ys / d1
        py2 = d1 * wy * wy * y0 * ys + py1 * yc

        d0 = 1 / RefParticle.beta + delta0

        c0 = (1 / RefParticle.beta - d0 / d1) * ds - d0 * a1 * (h * (ds - xs) + a1 * (2 * ds - xs2) / 8) / wx / wx / d1

        c1 = -d0 * (h * xs - a1 * (2 * ds - xs2) / 4) / d1

        c2 = -d0 * (h * (1 - xc) / wx / wx + a1 * xs * xs / 2) / d1 / d1

        c11 = -d0 * wx * wx * (2 * ds - xs2) / d1 / 8
        c12 = d0 * wx * wx * xs * xs / d1 / d1 / 2
        c22 = -d0 * (2 * ds + xs2) / d1 / d1 / d1 / 8

        c33 = -d0 * wy * wy * (2 * ds - ys2) / d1 / 8
        c34 = -d0 * wy * wy * ys * ys / d1 / d1 / 2
        c44 = -d0 * (2 * ds + ys2) / d1 / d1 / d1 / 8

        z1 = (z0 + c0 + c1 * x0 + c2 * px1 + c11 * x0 * x0 + c12 * x0 * px1 + c22 * px1 * px1 + c33 * y0 * y0 +
              c34 * y0 * py1 + c44 * py1 * py1)
        # exit
        px3 = px2 + self.h * np.tan(self.theta_out) * x2
        py3 = py2 - self.h * np.tan(self.theta_out) * y2
        beam.set_particle([x2, px3, y2, py3, z1, delta0])
        return beam

    def real_track(self, beam: Beam7) -> Beam7:
        [x0, px0, y0, py0, z0, delta0] = beam.get_particle()
        delta1 = delta0 - (delta0 + 1) ** 2 * Cr * RefParticle.energy ** 3 * self.theta ** 2 / 2 / pi / self.length
        # use average energy
        delta01 = (delta0 + delta1) / 2
        d1 = np.sqrt(1 + 2 * delta01 / RefParticle.beta + delta01 ** 2)
        ds = self.length
        # entrance
        px1 = px0 + self.h * np.tan(self.theta_in) * x0
        py1 = py0 - self.h * np.tan(self.theta_in) * y0
        # drift
        h = self.h
        k1 = self.k1
        a1 = h - self.h / d1

        wx = np.sqrt((h * self.h + k1) / d1)
        xc = np.cos(wx * ds)
        xs = np.sin(wx * ds) / wx
        xs2 = np.sin(2 * wx * ds) / wx

        wy = np.sqrt(k1 / d1)
        yc = np.cosh(wy * ds)
        ys = ds
        ys2 = 2 * ds

        if wy.any():
            ys = np.sinh(wy * ds) / wy
            ys2 = np.sinh(2 * wy * ds) / wy

        x2 = x0 * xc + px1 * xs / d1 + a1 * (1 - xc) / wx / wx
        px2 = -d1 * wx * wx * x0 * xs + px1 * xc + a1 * xs * d1

        y2 = y0 * yc + py1 * ys / d1
        py2 = d1 * wy * wy * y0 * ys + py1 * yc

        d0 = 1 / RefParticle.beta + delta0

        c0 = (1 / RefParticle.beta - d0 / d1) * ds - d0 * a1 * (h * (ds - xs) + a1 * (2 * ds - xs2) / 8) / wx / wx / d1

        c1 = -d0 * (h * xs - a1 * (2 * ds - xs2) / 4) / d1

        c2 = -d0 * (h * (1 - xc) / wx / wx + a1 * xs * xs / 2) / d1 / d1

        c11 = -d0 * wx * wx * (2 * ds - xs2) / d1 / 8
        c12 = d0 * wx * wx * xs * xs / d1 / d1 / 2
        c22 = -d0 * (2 * ds + xs2) / d1 / d1 / d1 / 8

        c33 = -d0 * wy * wy * (2 * ds - ys2) / d1 / 8
        c34 = -d0 * wy * wy * ys * ys / d1 / d1 / 2
        c44 = -d0 * (2 * ds + ys2) / d1 / d1 / d1 / 8

        z1 = (z0 + c0 + c1 * x0 + c2 * px1 + c11 * x0 * x0 + c12 * x0 * px1 + c22 * px1 * px1 + c33 * y0 * y0 +
              c34 * y0 * py1 + c44 * py1 * py1)
        # damping
        e1_div_e0 = (delta1 + 1) / (delta0 + 1)
        px2 = px2 * e1_div_e0
        py2 = py2 * e1_div_e0
        # exit
        px3 = px2 + self.h * np.tan(self.theta_out) * x2
        py3 = py2 - self.h * np.tan(self.theta_out) * y2
        beam.set_particle([x2, px3, y2, py3, z1, delta1])
        return beam

    def slice(self, initial_s, identifier):
        """slice component to element list, return [ele_list, final_z]"""
        ele_list = []
        current_s = initial_s
        ele = deepcopy(self)
        ele.identifier = identifier
        ele.theta_out = 0
        ele.s = current_s
        ele.length = round(self.length / self.n_slices, LENGTH_PRECISION)
        ele_list.append(deepcopy(ele))
        current_s = round(current_s + ele.length, LENGTH_PRECISION)
        if self.n_slices == 1:
            ele_list[0].theta_out = self.theta_out
            return [ele_list, current_s]
        for i in range(self.n_slices - 2):
            ele.s = current_s
            ele.theta_in = 0
            ele.theta_out = 0
            ele.length = round(self.length / self.n_slices, LENGTH_PRECISION)
            ele_list.append(deepcopy(ele))
            current_s = round(current_s + ele.length, LENGTH_PRECISION)
        ele.s = current_s
        ele.theta_in = 0
        ele.theta_out = self.theta_out
        ele.length = round(self.length + initial_s - current_s, LENGTH_PRECISION)
        ele_list.append(deepcopy(ele))
        current_s = round(current_s + ele.length, LENGTH_PRECISION)
        return [ele_list, current_s]

    def __str__(self):
        text = str(self.name)
        text += (' ' * max(0, 6 - len(self.name)))
        text += (': ' + str(self.type()))
        text += (':   s = ' + str(self.s))
        text += (',   length = ' + str(self.length))
        theta = self.theta * 180 / pi
        text += ',   theta = ' + str(theta)
        text += ',   theta_in = ' + str(self.theta_in)
        text += ',   theta_out = ' + str(self.theta_out)
        return text
