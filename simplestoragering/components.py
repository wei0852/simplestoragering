"""Components:
Drift:
........."""

from .particles import Particle, Beam7
import numpy as np
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from .constants import c, pi, Cr, LENGTH_PRECISION
from .exceptions import UnfinishedWork


def match_symbol(symbol):
    """:return the element type of the symbol

    1: drift, 2: dipole, 3: quadrupole, 4: sextupole,
    21: sector dipole, 22: rectangle dipole, 24: dipole edge
    """
    symbol_dict = {-1: 'end', 1: 'Drift     ', 2: 'Bend      ', 3: 'Quadrupole', 4: 'Sextupole ', 5: 'rf cavity'}
    try:
        return symbol_dict[int(symbol / 100)]
    except KeyError:
        raise UnfinishedWork('symbol key error')


class Element(metaclass=ABCMeta):
    """parent class of magnet components

    has 3 kind of matrices, 7x7 matrix to solve closed orbit, matrix for slim method,
    matrix for Courant-Snyder method


    Methods:
        matrix: transport matrix for Courant-Snyder method
        matrix: coupled matrix
        damping_matrix: transport matrix with damping
        closed_orbit_matrix: transport matrix for solving closed orbit
    """
    name = None
    symbol = None
    length = 0
    s = None
    h = 0
    k1 = 0
    k2 = 0
    n_slices = 1
    identifier = None
    closed_orbit = np.zeros(6)
    tune = None
    beam = None
    betax = None
    alphax = None
    gammax = None
    psix = None
    betay = None
    alphay = None
    gammay = None
    psiy = None
    etax = None
    etaxp = None
    etay = None
    etayp = None
    curl_H = None

    @property
    @abstractmethod
    def damping_matrix(self):
        """matrix with coupled effects and damping rates to calculate tune and damping rate"""
        pass

    @property
    @abstractmethod
    def matrix(self):
        """matrix with coupled effects to calculate tune and bunch distribution"""

    @property
    def next_closed_orbit(self):
        """usually """
        x07 = np.append(self.closed_orbit, 1)
        x0 = self.closed_orbit_matrix.dot(x07)
        x0 = np.delete(x0, [6])
        return x0

    @property
    @abstractmethod
    def closed_orbit_matrix(self):
        """:return a 7x7 matrix to solve the closed orbit"""
        pass

    @abstractmethod
    def symplectic_track(self, beam: Beam7) -> Beam7:
        """assuming that the energy is constant, the result is symplectic."""
        pass

    @abstractmethod
    def real_track(self, beam: Beam7) -> Beam7:
        """tracking with energy loss, the result is not symplectic"""
        pass

    def type(self):
        """return the type of the component"""

        return match_symbol(self.symbol)

    def slice(self, initial_s, identifier):
        """slice component to element list, return [ele_list, final_z], the identifier identifies different magnet"""
        ele_list = []
        current_s = initial_s
        ele = deepcopy(self)
        ele.identifier = identifier
        for i in range(self.n_slices - 1):
            ele.s = current_s
            ele.length = round(self.length / self.n_slices, LENGTH_PRECISION)
            ele_list.append(deepcopy(ele))
            current_s = round(current_s + ele.length, LENGTH_PRECISION)
        ele.s = current_s
        ele.length = round(self.length + initial_s - current_s, LENGTH_PRECISION)
        ele_list.append(deepcopy(ele))
        current_s = round(current_s + ele.length, LENGTH_PRECISION)
        return [ele_list, current_s]

    def __sub_matrix(self, direction):
        """return sub_matrix of x or y direction"""
        if direction == 'x':
            return np.array([[self.matrix[0, 0], self.matrix[0, 1]],
                             [self.matrix[1, 0], self.matrix[1, 1]]])
        elif direction == 'y':
            return np.array([[self.matrix[2, 2], self.matrix[2, 3]],
                             [self.matrix[3, 2], self.matrix[3, 3]]])
        else:
            raise Exception("direction must be 'x' or 'y' !!!")

    def __get_twiss(self, direction):
        if direction == 'x':
            return np.array([self.betax, self.alphax, self.gammax])
        elif direction == 'y':
            return np.array([self.betay, self.alphay, self.gammay])

    def next_twiss(self, direction):
        """calculate twiss parameters at the element's exit according to the data at the element's entrance"""
        sub = self.__sub_matrix(direction)
        matrix_cal = np.array([[sub[0, 0] ** 2, -2 * sub[0, 0] * sub[0, 1], sub[0, 1] ** 2],
                               [-sub[0, 0] * sub[1, 0], 2 * sub[0, 1] * sub[1, 0] + 1, -sub[0, 1] * sub[1, 1]],
                               [sub[1, 0] ** 2, -2 * sub[1, 0] * sub[1, 1], sub[1, 1] ** 2]])
        return matrix_cal.dot(self.__get_twiss(direction))

    def next_eta_bag(self, direction):
        """calculate  parameters at the element's exit according to the data at the element's entrance"""
        if direction == 'x':
            eta_bag = np.array([self.etax, self.etaxp])
            return self.__sub_matrix('x').dot(eta_bag) + np.array([self.matrix[0, 5], self.matrix[1, 5]])
        elif direction == 'y':
            eta_bag = np.array([self.etay, self.etayp])
            return self.__sub_matrix('y').dot(eta_bag) + np.array([self.matrix[2, 5], self.matrix[3, 5]])
        else:
            raise Exception("direction must be 'x' or 'y' !!!")

    def next_phase(self):
        """:return dpsix, dpsiy"""
        dpsix = np.arctan(self.matrix[0, 1] / (self.matrix[0, 0] * self.betax - self.matrix[0, 1] * self.alphax))
        psix = self.psix + dpsix
        dpsiy = np.arctan(self.matrix[2, 3] / (self.matrix[2, 2] * self.betay - self.matrix[2, 3] * self.alphay))
        psiy = self.psiy + dpsiy
        return psix, psiy

    def __repr__(self):
        return self.name

    def __str__(self):
        text = str(self.name)
        text += (' ' * max(0, 6 - len(str(self.name))))
        text += (': ' + str(match_symbol(self.symbol)))
        text += (':   s = ' + str(self.s))
        text += (',   length = ' + str(self.length))
        if self.h != 0:
            theta = self.length * self.h * 180 / pi
            text += ',   theta = ' + str(theta)
        if self.k1 != 0:
            text += ',   k1 = ' + str(self.k1)
        if self.k2 != 0:
            text += ',   k2 = ' + str(self.k2)
        return text

    def magnets_data(self):
        """:return length, theta, k1, k2"""
        text = ''
        text += ('length = ' + str(self.length))
        if self.h != 0:
            theta = self.length * self.h * 180 / pi
            text += ',   theta = ' + str(theta)
        if self.k1 != 0:
            text += ',   k1 = ' + str(self.k1)
        if self.k2 != 0:
            text += ',   k2 = ' + str(self.k2)
        return text


class Mark(Element):
    """mark class, only for marking"""
    symbol = 0

    def __init__(self, name: str):
        self.name = name
        self.n_slices = 1

    @property
    def damping_matrix(self):
        return np.identity(6)

    @property
    def matrix(self):
        return np.identity(6)

    @property
    def closed_orbit_matrix(self):
        """mark element don't need this matrix, closed orbit won't change in Mark element"""
        return np.identity(7)

    @property
    def next_closed_orbit(self):
        return self.closed_orbit

    def symplectic_track(self, beam: Beam7) -> Beam7:
        assert isinstance(beam, Beam7)
        return beam

    def real_track(self, beam: Beam7) -> Beam7:
        return beam


class LineEnd(Element):
    """mark the end of a line, store the data at the end."""
    symbol = -100

    def __init__(self, s, identifier, name='_END_'):
        self.name = name
        self.s = s
        self.identifier = identifier
        self.n_slices = 1

    @property
    def damping_matrix(self):
        return np.identity(6)

    @property
    def matrix(self):
        return np.identity(6)

    @property
    def closed_orbit_matrix(self):
        """mark element don't need this matrix, closed orbit won't change in Mark element"""
        return np.identity(7)

    @property
    def next_closed_orbit(self):
        return self.closed_orbit

    def symplectic_track(self, beam):
        return beam

    def real_track(self, beam: Beam7) -> Beam7:
        return beam


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
                         [0, 0, 0, 0, 1, self.length / Particle.gamma ** 2],
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
        [x0, px0, y0, py0, z0, dp0] = beam.get_particle()
        ds = self.length
        d1 = np.sqrt(1 - px0 ** 2 - py0 ** 2 + 2 * dp0 / Particle.beta + dp0 ** 2)
        x1 = x0 + ds * px0 / d1
        y1 = y0 + ds * py0 / d1
        z1 = z0 + ds * (1 - (1 + Particle.beta * dp0) / d1) / Particle.beta
        beam.set_particle([x1, px0, y1, py0, z1, dp0])
        return beam

    def real_track(self, beam: Beam7) -> Beam7:
        return self.symplectic_track(beam)


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
        h_beta = self.h * Particle.beta
        sin_theta = np.sin(self.theta)
        inlet_edge = np.array([[1, 0, 0, 0, 0, 0],
                               [np.tan(self.theta_in) * self.h, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, -np.tan(self.theta_in) * self.h, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]])
        # middle_section = np.array([[cx, sin_theta / self.h, 0, 0, 0, (1 - cx) / h_beta],
        #                            [-sin_theta * self.h, cx, 0, 0, 0, sin_theta / Particle.beta],
        #                            [0, 0, 1, self.length, 0, 0],
        #                            [0, 0, 0, 1, 0, 0],
        #                            [- np.sin(self.theta), - (1 - cx) / h_beta, 0, 0, 1,
        #                             - self.length + sin_theta / h_beta / Particle.beta],
        #                            [0, 0, 0, 0, 0, 1]])
        # according to elegant result
        middle_section = np.array([[cx, sin_theta / self.h, 0, 0, 0, (1 - cx) / h_beta],
                                   [-sin_theta * self.h, cx, 0, 0, 0, sin_theta / Particle.beta],
                                   [0, 0, 1, self.length, 0, 0],
                                   [0, 0, 0, 1, 0, 0],
                                   [np.sin(self.theta), (1 - cx) / h_beta, 0, 0, 1,
                                    self.length - sin_theta / h_beta / Particle.beta],
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
        m66 = 1 - Cr * Particle.energy ** 3 * self.length * self.h ** 2 / pi
        # delta_delta = - Cr * Particle.energy ** 3 * self.length * self.h ** 2 / pi / 2
        matrix = self.matrix
        matrix[5, 5] = m66
        # matrix[1, 5] = matrix[1, 5] * (1 + delta_delta / self.closed_orbit[5] / 2)
        return matrix

    @property
    def closed_orbit_matrix(self):
        # m67 = -(self.closed_orbit[5] + 1) ** 2 * Cr * Particle.energy ** 3 * self.theta ** 2 / 2 / pi / self.length
        m67 = -Cr * Particle.energy ** 3 * self.theta ** 2 / 2 / pi / self.length
        matrix7 = np.identity(7)
        matrix7[0:6, 0:6] = self.matrix  # Bend doesn't use thin len approximation, so
        matrix7[5, 6] = m67
        matrix7[1, 6] = self.h * self.length * m67 / 2
        return matrix7

    def symplectic_track(self, beam):
        [x0, px0, y0, py0, z0, dp0] = beam.get_particle()
        d1 = np.sqrt(1 + 2 * dp0 / Particle.beta + dp0 ** 2)
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

        d0 = 1 / Particle.beta + dp0

        c0 = (1 / Particle.beta - d0 / d1) * ds - d0 * a1 * (h * (ds - xs) + a1 * (2 * ds - xs2) / 8) / wx / wx / d1

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
        beam.set_particle([x2, px3, y2, py3, z1, dp0])
        return beam

    def real_track(self, beam: Beam7) -> Beam7:
        [x0, px0, y0, py0, z0, dp0] = beam.get_particle()
        dp1 = dp0 - (dp0 + 1) ** 2 * Cr * Particle.energy ** 3 * self.theta ** 2 / 2 / pi / self.length
        # use average energy
        dp0 = (dp0 + dp1) / 2
        d1 = np.sqrt(1 + 2 * dp0 / Particle.beta + dp0 ** 2)
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

        if np.all(wy):
            ys = np.sinh(wy * ds) / wy
            ys2 = np.sinh(2 * wy * ds) / wy

        x2 = x0 * xc + px1 * xs / d1 + a1 * (1 - xc) / wx / wx
        px2 = -d1 * wx * wx * x0 * xs + px1 * xc + a1 * xs * d1

        y2 = y0 * yc + py1 * ys / d1
        py2 = d1 * wy * wy * y0 * ys + py1 * yc

        d0 = 1 / Particle.beta + dp0

        c0 = (1 / Particle.beta - d0 / d1) * ds - d0 * a1 * (h * (ds - xs) + a1 * (2 * ds - xs2) / 8) / wx / wx / d1

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
        '''exit'''
        px3 = px2 + self.h * np.tan(self.theta_out) * x2
        py3 = py2 - self.h * np.tan(self.theta_out) * y2
        beam.set_particle([x2, px3, y2, py3, z1, dp1])
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
        text += (': ' + str(match_symbol(self.symbol)))
        text += (':   s = ' + str(self.s))
        text += (',   length = ' + str(self.length))
        theta = self.theta * 180 / pi
        text += ',   theta = ' + str(theta)
        text += ',   theta_in = ' + str(self.theta_in)
        text += ',   theta_out = ' + str(self.theta_out)
        return text


class VBend(Element):
    """vertical Bend"""
    symbol = 250

    def __init__(self, name: str = None, length: float = 0, theta: float = 0, theta_in: float = 0, theta_out: float = 0,
                 n_slices: int = 3):
        self.name = name
        self.length = length
        self.theta = theta
        self.h = self.theta / self.length
        self.theta_in = theta_in
        self.theta_out = theta_out
        self.n_slices = max(n_slices, 3)

    @property
    def damping_matrix(self):
        raise UnfinishedWork()

    @property
    def matrix(self):
        raise UnfinishedWork()

    @property
    def closed_orbit_matrix(self):
        m67 = - Cr * Particle.energy ** 3 * self.theta ** 2 / 2 / pi / self.length
        matrix7 = np.identity(7)
        matrix7[0:6, 0:6] = self.matrix
        matrix7[5, 6] = m67
        matrix7[3, 6] = - m67 * self.theta / 2
        return matrix7

    def symplectic_track(self, beam):
        pass

    def real_track(self, beam: Beam7) -> Beam7:
        pass


class Quadrupole(Element):
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
        if self.k1 > 0:
            sqk = np.sqrt(self.k1)
            sqkl = sqk * self.length
            return np.array([[np.cos(sqkl), np.sin(sqkl) / sqk, 0, 0, 0, 0],
                             [- sqk * np.sin(sqkl), np.cos(sqkl), 0, 0, 0, 0],
                             [0, 0, np.cosh(sqkl), np.sinh(sqkl) / sqk, 0, 0],
                             [0, 0, sqk * np.sinh(sqkl), np.cosh(sqkl), 0, 0],
                             [0, 0, 0, 0, 1, self.length / Particle.gamma ** 2],
                             [0, 0, 0, 0, 0, 1]])
        else:
            sqk = np.sqrt(-self.k1)
            sqkl = sqk * self.length
            return np.array([[np.cosh(sqkl), np.sinh(sqkl) / sqk, 0, 0, 0, 0],
                             [sqk * np.sinh(sqkl), np.cosh(sqkl), 0, 0, 0, 0],
                             [0, 0, np.cos(sqkl), np.sin(sqkl) / sqk, 0, 0],
                             [0, 0, - sqk * np.sin(sqkl), np.cos(sqkl), 0, 0],
                             [0, 0, 0, 0, 1, self.length / Particle.gamma ** 2],
                             [0, 0, 0, 0, 0, 1]])

    @property
    def thin_len(self):
        drift = Drift(length=self.length / 2).matrix
        matrix = np.identity(6)
        matrix[1, 0] = - self.k1 * self.length
        matrix[3, 2] = self.k1 * self.length
        return drift.dot(matrix).dot(drift)

    @property
    def damping_matrix(self):
        lambda_q = Cr * Particle.energy ** 3 * self.k1 ** 2 * self.length / pi
        matrix = self.matrix
        matrix[5, 5] = 1 - lambda_q * (self.closed_orbit[0] ** 2 + self.closed_orbit[2] ** 2)
        matrix[5, 0] = - lambda_q * self.closed_orbit[0]
        matrix[5, 2] = - lambda_q * self.closed_orbit[2]
        return matrix

    @property
    def closed_orbit_matrix(self):
        m67 = - Cr * Particle.energy ** 3 * self.k1 ** 2 * self.length * (self.closed_orbit[0] ** 2 +
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

        beta0 = Particle.beta

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

        beta0 = Particle.beta

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
        [x0, px0, y0, py0, z0, dp0] = beam.get_particle()
        dp1 = dp0 - (dp0 + 1)**2 * Cr * Particle.energy ** 3 * self.k1 ** 2 * self.length * (x0 ** 2 + y0 ** 2) / 2 / pi
        dp_ave = (dp1 + dp0) / 2
        beam.set_particle([x0, px0, y0, py0, z0, dp_ave])
        beam = self.symplectic_track(beam)
        beam.set_dp(dp1)
        return beam


class SKQuadrupole(Element):
    """skew Quadrupole, b_x = - k1 x, b_y = k1 y"""
    symbol = 350

    def __init__(self, name: str = None, length: float = 0, k1: float = 0, n_slices: int = 1):
        self.name = name
        self.length = length
        self.k1 = k1
        self.n_slices = n_slices
        if k1 > 0:
            self.symbol = 370
        else:
            self.symbol = 360
        raise UnfinishedWork()

    @property
    def matrix(self):
        raise UnfinishedWork()

    @property
    def damping_matrix(self):
        # lambda_q = Cr * Particle.energy ** 3 * self.k1 ** 2 * self.length / pi
        raise UnfinishedWork()

    @property
    def closed_orbit_matrix(self):
        raise UnfinishedWork()

    def symplectic_track(self, beam):
        pass

    def real_track(self, beam: Beam7) -> Beam7:
        pass


class Sextupole(Element):
    """sextupole"""
    symbol = 400

    def __init__(self, name: str = None, length: float = 0, k2: float = 0, n_slices: int = 1):
        self.name = name
        self.length = length
        self.k2 = k2
        self.n_slices = n_slices
        if k2 > 0:
            self.symbol = 420
        else:
            self.symbol = 410

    @property
    def matrix(self):
        k2l = self.k2 * self.length
        x0 = self.closed_orbit[0]
        y0 = self.closed_orbit[2]
        x02_y02_2 = (x0 ** 2 - y0 ** 2) / 2  # (x0 ** 2 - y0 ** 2) / 2
        matrix = np.array([[1, 0, 0, 0, 0, 0],
                           [- k2l * x0, 1, k2l * y0, 0, 0, k2l * x02_y02_2],
                           [0, 0, 1, 0, 0, 0],
                           [k2l * y0, 0, k2l * x0, 1, 0, - k2l * x0 * y0],
                           [- k2l * x02_y02_2, 0, k2l * x0 * y0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        drift = Drift(length=self.length / 2).matrix
        total = drift.dot(matrix).dot(drift)
        return total

    @property
    def damping_matrix(self):
        """I think damping is not in thin len approximation"""
        k2l = self.k2 * self.length
        x0 = self.closed_orbit[0]
        y0 = self.closed_orbit[2]
        lambda_s = Cr * Particle.energy ** 3 * k2l ** 2 * (x0 ** 2 + y0 ** 2) / pi / self.length
        matrix = self.matrix
        matrix[5, 0] = - lambda_s * x0
        matrix[5, 2] = - lambda_s * y0
        matrix[5, 5] = 1 - lambda_s * (x0 ** 2 + y0 ** 2) / 2
        return matrix

    @property
    def closed_orbit_matrix(self):
        """it's different from its transform matrix, x is replaced by closed orbit x0"""
        m67 = - (Cr * Particle.energy ** 3 * self.k2 ** 2 * self.length *
                 (self.closed_orbit[0] ** 2 + self.closed_orbit[2] ** 2) ** 2 / 8 / pi)
        # m67 = m67 * (1 + self.closed_orbit[5]) ** 2
        matrix7 = np.identity(7)
        drift = Drift(length=self.length / 2).matrix
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
        [x0, px0, y0, py0, ct0, dp0] = beam.get_particle()

        beta0 = Particle.beta

        ds = self.length
        k2 = self.k2
        # drift
        d1 = np.sqrt(1 - px0 * px0 - py0 * py0 + 2 * dp0 / beta0 + dp0 * dp0)

        x1 = x0 + ds * px0 / d1 / 2
        y1 = y0 + ds * py0 / d1 / 2
        ct1 = ct0 + ds * (1 - (1 + beta0 * dp0) / d1) / beta0 / 2
        # kick
        px1 = px0 - (x1 * x1 - y1 * y1) * k2 * ds / 2
        py1 = py0 + x1 * y1 * k2 * ds
        # drift
        d1 = np.sqrt(1 - px1 * px1 - py1 * py1 + 2 * dp0 / beta0 + dp0 * dp0)

        x2 = x1 + ds * px1 / d1 / 2
        y2 = y1 + ds * py1 / d1 / 2
        ct2 = ct1 + ds * (1 - (1 + beta0 * dp0) / d1) / beta0 / 2

        beam.set_particle([x2, px1, y2, py1, ct2, dp0])
        return beam

    def real_track(self, beam: Beam7) -> Beam7:
        [x0, px0, y0, py0, ct0, dp0] = beam.get_particle()
        dp1 = dp0 - (dp0 + 1)**2 * (Cr * Particle.energy ** 3 * self.k2 ** 2 * self.length *
                                    (x0 ** 2 + y0 ** 2) ** 2 / 8 / pi)
        dp_ave = (dp1 + dp0) / 2
        beam.set_particle([x0, px0, y0, py0, ct0, dp_ave])
        beam = self.symplectic_track(beam)
        beam.set_dp(dp1)
        return beam


class RFCavity(Element):
    """thin len approximation, don't have length. The unit of voltage should be the same as Particle.energy, MeV"""
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
        temp_val = 2 * pi * self.f_rf / Particle.beta / c  # h / R
        matrix = np.identity(6)
        matrix[5, 4] = (self.voltage * temp_val * np.cos(self.phase) / Particle.energy)
        drift = Drift(length=self.length / 2).matrix
        total = drift.dot(matrix).dot(drift)
        return total

    @property
    def damping_matrix(self):
        temp_val = 2 * pi * self.f_rf / Particle.beta / c  # h / R
        z0 = self.closed_orbit[4]
        matrix = self.matrix
        matrix[1, 1] = 1 - self.voltage * (np.sin(self.phase) + temp_val * z0 * np.cos(self.phase)) / Particle.energy
        matrix[3, 3] = matrix[1, 1]
        return matrix

    @property
    def closed_orbit_matrix(self):
        m67 = self.voltage * np.sin(self.phase) / Particle.energy
        # m67 = self.voltage * np.sin(self.phase + self.omega_rf * self.closed_orbit[4] / c) / Particle.energy
        matrix7 = np.identity(7)
        matrix7[0:6, 0:6] = self.matrix
        matrix7[5, 6] = m67
        return matrix7

    def symplectic_track(self, beam):
        """rf cavity tracking is simplified by thin len approximation"""
        [x0, px0, y0, py0, z0, dp0] = beam.get_particle()
        beta0 = Particle.beta
        ds = self.length
        # First        apply        a        drift        through        ds / 2
        d1 = np.sqrt(1 - px0 * px0 - py0 * py0 + 2 * dp0 / beta0 + dp0 * dp0)
        x1 = x0 + ds * px0 / d1 / 2
        y1 = y0 + ds * py0 / d1 / 2
        z1 = z0 + ds * (1 - (1 + beta0 * dp0) / d1) / beta0 / 2
        # Next, apply        an        rf        'kick'
        vnorm = self.voltage / Particle.energy
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
        beta0 = Particle.beta
        ds = self.length
        # First        apply        a        drift        through        ds / 2
        d1 = np.sqrt(1 - px0 * px0 - py0 * py0 + 2 * delta0 / beta0 + delta0 * delta0)
        x1 = x0 + ds * px0 / d1 / 2
        y1 = y0 + ds * py0 / d1 / 2
        z1 = z0 + ds * (1 - (1 + beta0 * delta0) / d1) / beta0 / 2
        # Next, apply        an        rf        'kick'
        vnorm = self.voltage / Particle.energy
        delta1 = delta0 + vnorm * np.sin(self.phase - self.omega_rf * z1 / c)
        # Finally, apply        a        second        drift        through        ds / 2
        d1 = np.sqrt(1 - px0 * px0 - py0 * py0 + 2 * delta1 / beta0 + delta1 * delta1)
        x2 = x1 + ds * px0 / d1 / 2
        y2 = y1 + ds * py0 / d1 / 2
        z2 = z1 + ds * (1 - (1 + beta0 * delta1) / d1) / beta0 / 2
        # damping
        dp0_div_dp1 = (delta0 * Particle.beta + 1) / (delta1 * Particle.beta + 1)
        px1 = px0 * dp0_div_dp1
        py1 = py0 * dp0_div_dp1
        beam.set_particle([x2, px1, y2, py1, z2, delta1])
        return beam
