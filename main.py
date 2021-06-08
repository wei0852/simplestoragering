# Wei, Bingfeng
# wbf2016@mail.ustc.edu.cn

"""Slim Storage Ring

Energy in MeV !!!!!
"""

from abc import ABCMeta, abstractmethod
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi, physical_constants, c

LENGTH_PRECISION = 10


def calculate_constants():
    """Andrzej Wolski (2014)"""

    h_bar = physical_constants['natural unit of action in eV s'][0]
    re = physical_constants['classical electron radius'][0]
    me = physical_constants['electron mass energy equivalent in MeV'][0]
    # Cr = q ** 2 / (3 * epsilon_0 * (m * c**2) ** 4)   p.221
    cr = 4 * pi * re / (3 * me ** 3)  # this is only for electrons. unit m/MeV**3
    cq = 55 * h_bar * c / (32 * np.sqrt(3) * me * 1e6)  # p.232  unit s
    # this part is to verify cl
    # Pr = (Cr * c / 2 / pi) * (\beta_0^4 * E_0^4 / \rho^2)    p.221
    # <\dot{N} u^2> = 2 Cq \gamma_0^2 E_0 P_r / \rho          p.232
    # ==>  <\dot{N} u^2> = Cq Cr c E0^5 \gamma^2 \beta^4 / \pi \rho^3
    # <\dot{N} u^2 / E_0^2> = 2 Cl \gamma^5 / |\rho|^3  Slim Formalism orbit motion
    # therefore,
    # cl = cq * cr * c * me ** 3 / 2 / pi  # \beta \approx 1
    cl = 55 * re * h_bar * c ** 2 / (48 * np.sqrt(3) * me * 1e6)
    return cr, cq, cl


Cr, Cq, Cl = calculate_constants()


class UnfinishedWork(Exception):
    """user's exception for marking the unfinished work"""

    def __init__(self, *args):
        self.name = args[0]


class Particle(object):
    """set particle's type and energy"""
    energy = None
    gamma = None
    beta = None

    @classmethod
    def set_energy(cls, energy):
        cls.energy = energy
        text = "electron mass energy equivalent in MeV"
        mass = physical_constants[text][0]
        cls.gamma = cls.energy / mass
        cls.beta = np.sqrt(1 - 1 / cls.gamma ** 2)

    @classmethod
    def __str__(cls):
        return "mass: %s MeV, gamma = %s" % (cls.energy / cls.gamma, cls.gamma)


class Beam(object):
    """beam"""

    def __init__(self, particle=None):
        self.matrix = None
        self.precision = 1e-9  # the precision must be small
        if particle is not None:
            self.init_particle(particle)

    def init_particle(self, particle):
        assert len(particle) == 6
        self.matrix = np.eye(6, 7) * self.precision
        for i in range(6):
            self.matrix[i, :] = self.matrix[i, :] + particle[i]

    def set_particle(self, particle):
        particle[4] = -particle[4]
        for i in range(6):
            self.matrix[i, :] = particle[i]

    def get_particle(self):
        x = []
        for i in range(6):
            x.append(self.matrix[i, :])
        x[4] = -x[4]
        return [x[i] for i in range(6)]

    def get_particle_array(self):
        p = np.zeros(6)
        for i in range(6):
            p[i] = self.matrix[i, 6]
        return p

    def set_dp(self, dp):
        self.matrix[5, :] = dp


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


class MatchSymbol(object):
    """match symbol"""

    def __init__(self):
        raise UnfinishedWork()


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
    def symplectic_track(self, beam: Beam) -> Beam:
        """assuming that the energy is constant, the result is symplectic."""
        pass

    @abstractmethod
    def real_track(self, beam: Beam) -> Beam:
        """tracking with energy loss, the result is not symplectic"""
        pass

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

    def symplectic_track(self, beam: Beam) -> Beam:
        assert isinstance(beam, Beam)
        return beam

    def real_track(self, beam: Beam) -> Beam:
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

    def real_track(self, beam: Beam) -> Beam:
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
        assert isinstance(beam, Beam)
        [x0, px0, y0, py0, z0, dp0] = beam.get_particle()
        ds = self.length
        d1 = np.sqrt(1 - px0 ** 2 - py0 ** 2 + 2 * dp0 / Particle.beta + dp0 ** 2)
        x1 = x0 + ds * px0 / d1
        y1 = y0 + ds * py0 / d1
        z1 = z0 + ds * (1 - (1 + Particle.beta * dp0) / d1) / Particle.beta
        beam.set_particle([x1, px0, y1, py0, z1, dp0])
        return beam

    def real_track(self, beam: Beam) -> Beam:
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

    def real_track(self, beam: Beam) -> Beam:
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

    def real_track(self, beam: Beam) -> Beam:
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

    def real_track(self, beam: Beam) -> Beam:
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

    def real_track(self, beam: Beam) -> Beam:
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

    def real_track(self, beam: Beam) -> Beam:
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

    def real_track(self, beam: Beam) -> Beam:
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


class CSLattice(object):
    """lattice object, solve by Courant-Snyder method"""

    def __init__(self, ele_list: list, periods_number: int, coupling=0.00):
        self.length = 0
        self.periods_number = periods_number
        self.coup = coupling
        self.elements = []
        self.rf_cavity = None
        for ele in ele_list:
            # ele.s = self.length
            if isinstance(ele, RFCavity):
                self.rf_cavity = ele
            self.elements.append(ele)
            self.length = round(self.length + ele.length, LENGTH_PRECISION)
        self.ele_slices = None
        self.__slice()
        # initialize twiss
        self.twiss_x0 = None
        self.twiss_y0 = None
        self.eta_x0 = None
        self.eta_y0 = None
        self.__solve_initial_twiss()
        # solve twiss
        self.nux = None
        self.nuy = None
        self.__solve_along()
        # integration
        self.xi_x = None
        self.xi_y = None
        self.I1 = None
        self.I2 = None
        self.I3 = None
        self.I4 = None
        self.I5 = None
        self.radiation_integrals()
        # global parameters
        self.Jx = None
        self.Jy = None
        self.Js = None
        self.sigma_e = None
        self.emittance = None
        self.U0 = None
        self.f_c = None
        self.tau0 = None
        self.tau_s = None
        self.tau_x = None
        self.tau_y = None
        self.alpha = None
        self.emitt_x = None
        self.emitt_y = None
        self.etap = None
        self.sigma_z = None
        self.global_parameters()

    def __slice(self):
        self.ele_slices = []
        current_s = 0
        current_identifier = 0
        for ele in self.elements:
            [new_list, current_s] = ele.slice(current_s, current_identifier)
            self.ele_slices += new_list
            current_identifier += 1
        last_ele = LineEnd(s=self.length, identifier=current_identifier)
        self.ele_slices.append(last_ele)

    def __solve_initial_twiss(self):
        matrix = np.identity(6)
        for ele in self.elements:
            matrix = ele.matrix.dot(matrix)
        # x direction
        cos_mu = (matrix[0, 0] + matrix[1, 1]) / 2
        assert abs(cos_mu) < 1, "can not find period solution, UNSTABLE!!!"
        mu = np.arccos(cos_mu) * np.sign(matrix[0, 1])
        beta = matrix[0, 1] / np.sin(mu)
        alpha = (matrix[0, 0] - matrix[1, 1]) / (2 * np.sin(mu))
        gamma = - matrix[1, 0] / np.sin(mu)
        self.twiss_x0 = np.array([beta, alpha, gamma])
        # y direction
        cos_mu = (matrix[2, 2] + matrix[3, 3]) / 2
        assert abs(cos_mu) < 1, "can not find period solution, UNSTABLE!!!"
        mu = np.arccos(cos_mu) * np.sign(matrix[2, 3])
        beta = matrix[2, 3] / np.sin(mu)
        alpha = (matrix[2, 2] - matrix[3, 3]) / (2 * np.sin(mu))
        gamma = - matrix[3, 2] / np.sin(mu)
        self.twiss_y0 = np.array([beta, alpha, gamma])
        # solve eta
        sub_matrix_x = matrix[0:2, 0:2]
        matrix_etax = np.array([matrix[0, 5], matrix[1, 5]])
        self.eta_x0 = np.linalg.inv(np.identity(2) - sub_matrix_x).dot(matrix_etax)
        sub_matrix_y = matrix[2:4, 2:4]
        matrix_etay = np.array([matrix[2, 5], matrix[3, 5]])
        self.eta_y0 = np.linalg.inv(np.identity(2) - sub_matrix_y).dot(matrix_etay)

    def __solve_along(self):
        [betax, alphax, gammax] = self.twiss_x0
        [betay, alphay, gammay] = self.twiss_y0
        [etax, etaxp] = self.eta_x0
        [etay, etayp] = self.eta_y0
        psix = 0
        psiy = 0
        for ele in self.ele_slices:
            ele.betax = betax
            ele.betay = betay
            ele.alphax = alphax
            ele.alphay = alphay
            ele.gammax = gammax
            ele.gammay = gammay
            ele.etax = etax
            ele.etay = etay
            ele.etaxp = etaxp
            ele.etayp = etayp
            ele.psix = psix
            ele.psiy = psiy
            ele.curl_H = ele.gammax * ele.etax ** 2 + 2 * ele.alphax * ele.etax * ele.etaxp + ele.betax * ele.etaxp ** 2
            [betax, alphax, gammax] = ele.next_twiss('x')
            [betay, alphay, gammay] = ele.next_twiss('y')
            [etax, etaxp] = ele.next_eta_bag('x')
            [etay, etayp] = ele.next_eta_bag('y')
            psix, psiy = ele.next_phase()
        self.nux = psix * self.periods_number / 2 / pi
        self.nuy = psiy * self.periods_number / 2 / pi

    def radiation_integrals(self):
        integral1 = 0
        integral2 = 0
        integral3 = 0
        integral4 = 0
        integral5 = 0
        chromaticity_x = 0
        chromaticity_y = 0
        for ele in self.ele_slices:
            integral1 = integral1 + ele.length * ele.etax * ele.h
            integral2 = integral2 + ele.length * ele.h ** 2
            integral3 = integral3 + ele.length * abs(ele.h) ** 3
            integral4 = integral4 + ele.length * (ele.h ** 2 + 2 * ele.k1) * ele.etax * ele.h
            integral5 = integral5 + ele.length * ele.curl_H * abs(ele.h) ** 3
            chromaticity_x = chromaticity_x - (ele.k1 + ele.h ** 2 - ele.etax * ele.k2) * ele.length * ele.betax
            chromaticity_y = chromaticity_y + (ele.k1 - ele.etax * ele.k2) * ele.length * ele.betay
            if 200 <= ele.symbol < 300:
                integral4 = integral4 + (ele.h ** 2 * ele.etax * np.tan(ele.theta_in)
                                         - ele.h ** 2 * ele.etax * np.tan(ele.theta_out))
                chromaticity_x = chromaticity_x + ele.h * (np.tan(ele.theta_in) + np.tan(ele.theta_out)) * ele.betax
                chromaticity_y = chromaticity_y - ele.h * (np.tan(ele.theta_in) + np.tan(ele.theta_out)) * ele.betay
        self.I1 = integral1 * self.periods_number
        self.I2 = integral2 * self.periods_number
        self.I3 = integral3 * self.periods_number
        self.I4 = integral4 * self.periods_number
        self.I5 = integral5 * self.periods_number
        self.xi_x = chromaticity_x * self.periods_number / (4 * pi)
        self.xi_y = chromaticity_y * self.periods_number / (4 * pi)

    def global_parameters(self):
        self.Jx = 1 - self.I4 / self.I2
        self.Jy = 1
        self.Js = 2 + self.I4 / self.I2
        self.sigma_e = Particle.gamma * np.sqrt(Cq * self.I3 / (self.Js * self.I2))
        self.emittance = Cq * Particle.gamma * Particle.gamma * self.I5 / (self.Jx * self.I2)
        self.U0 = Cr * Particle.energy ** 4 * self.I2 / (2 * pi)
        self.f_c = c * Particle.beta / (self.length * self.periods_number)
        if self.rf_cavity is not None:
            self.rf_cavity.f_c = self.f_c
        self.tau0 = 2 * Particle.energy / self.U0 / self.f_c
        self.tau_s = self.tau0 / self.Js
        self.tau_x = self.tau0 / self.Jx
        self.tau_y = self.tau0 / self.Jy
        self.alpha = self.I1 * self.f_c / c  # momentum compaction factor
        self.emitt_x = self.emittance / (1 + self.coup)
        self.emitt_y = self.emittance * self.coup / (1 + self.coup)
        self.etap = self.alpha - 1 / Particle.gamma ** 2  # phase slip factor

    def matrix_output(self, file_name: str = 'matrix.txt'):
        """output uncoupled matrix for each element and contained matrix"""

        matrix = np.identity(6)
        file = open(file_name, 'w')
        location = 0.0
        for ele in self.elements:
            file.write(f'{match_symbol(ele.symbol)} {ele.name} at s={location},  {ele.magnets_data()}\n')
            location = round(location + ele.length, LENGTH_PRECISION)
            file.write(str(ele.matrix) + '\n')
            file.write('contained matrix:\n')
            matrix = ele.matrix.dot(matrix)
            file.write(str(matrix))
            file.write('\n\n--------------------------\n\n')
        file.close()

    def output_twiss(self, file_name: str = 'twiss_data.txt'):
        """output s, ElementName, betax, alphax, psix, betay, alphay, psiy, etax, etaxp"""

        file1 = open(file_name, 'w')
        file1.write('& s, ElementName, betax, alphax, psix, betay, alphay, psiy, etax, etaxp\n')
        last_identifier = 123465
        for ele in self.ele_slices:
            if ele.identifier != last_identifier:
                file1.write(f'{ele.s:.6e} {ele.name:10} {ele.betax:.6e}  {ele.alphax:.6e}  {ele.psix:.6e}  '
                            f'{ele.betay:.6e}  {ele.alphay:.6e}  {ele.psiy:.6e}  {ele.etax:.6e}  {ele.etaxp:.6e}\n')
                last_identifier = ele.identifier
        file1.close()

    def __str__(self):
        val = ""
        val += ("nux =       " + str(self.nux))
        val += ("\nnuy =       " + str(self.nuy))
        val += ("\ncurl_H =    " + str(self.ele_slices[0].curl_H))
        val += ("\nI1 =        " + str(self.I1))
        val += ("\nI2 =        " + str(self.I2))
        val += ("\nI3 =        " + str(self.I3))
        val += ("\nI4 =        " + str(self.I4))
        val += ("\nI5 =        " + str(self.I5))
        val += ("\nJs =        " + str(self.Js))
        val += ("\nJx =        " + str(self.Jx))
        val += ("\nJy =        " + str(self.Jy))
        val += ("\nenergy =    " + str(Particle.energy) + "MeV")
        val += ("\ngamma =     " + str(Particle.gamma))
        val += ("\nsigma_e =   " + str(self.sigma_e))
        val += ("\nemittance = " + str(self.emittance * 1e9) + " nm*rad")
        val += ("\nLength =    " + str(self.length * self.periods_number) + " m")
        val += ("\nU0 =        " + str(self.U0 * 1000) + "  keV")
        val += ("\nTperiod =   " + str(1 / self.f_c * 1e9) + " nsec")
        val += ("\nalpha =     " + str(self.alpha))
        val += ("\neta_p =     " + str(self.etap))
        val += ("\ntau0 =      " + str(self.tau0 * 1e3) + " msec")
        val += ("\ntau_e =     " + str(self.tau_s * 1e3) + " msec")
        val += ("\ntau_x =     " + str(self.tau_x * 1e3) + " msec")
        val += ("\ntau_y =     " + str(self.tau_y * 1e3) + " msec")
        val += ("\nxi_x =     " + str(self.xi_x))
        val += ("\nxi_y =     " + str(self.xi_y))
        if self.sigma_z is not None:
            val += ("\nsigma_z =   " + str(self.sigma_z))
        return val


class SlimRing(object):
    """lattice object, solve by slim method"""

    def __init__(self, ele_list: list):
        self.length = 0
        self.elements = []
        for ele in ele_list:
            # ele.s = self.length
            self.elements.append(ele)
            self.length = round(self.length + ele.length, LENGTH_PRECISION)
        self.rf_cavity = None
        self.ele_slices = None
        self.damping = None
        self.U0 = 0
        self.f_c = 0
        self.__set_rf()
        self.__slice()
        self.solve_closed_orbit()
        # self.solve_damping()
        # self.along_ring()

    def __set_rf(self):
        """solve U0 and set rf parameters"""
        i2 = 0
        for ele in self.elements:
            i2 = i2 + ele.length * ele.h ** 2
            if isinstance(ele, RFCavity):
                self.rf_cavity = ele
        self.U0 = Cr * Particle.energy ** 4 * i2 / (2 * pi)
        # self.T_period = self.length * self.periods_number / (c * Particle.beta)
        self.f_c = c * Particle.beta / self.length
        if self.rf_cavity is not None:
            self.rf_cavity.f_c = self.f_c

    def __slice(self):
        self.ele_slices = []
        current_s = 0
        current_identifier = 0
        for ele in self.elements:
            [new_list, current_s] = ele.slice(current_s, current_identifier)
            self.ele_slices += new_list
            current_identifier += 1
        last_ele = LineEnd(s=self.length, identifier=current_identifier)
        self.ele_slices.append(last_ele)

    def solve_closed_orbit(self):
        """solve closed orbit, iterate to solve, renew the x0 and matrix"""
        x0 = np.array([0, 0, 0, 0, 0, 0])
        matrix7 = np.identity(7)
        i = 1
        print('\n-------------------\nsearching closed orbit:')
        while i < 300:
            for ele in self.ele_slices:
                ele.closed_orbit = deepcopy(x0)
                x0 = ele.next_closed_orbit
                matrix7 = ele.closed_orbit_matrix.dot(matrix7)
            coefficient_matrix = (matrix7 - np.identity(7))[0: 6, 0: 6]
            result_vec = -matrix7[0: 6, 6].T
            x0 = np.linalg.solve(coefficient_matrix, result_vec)
            print(f'\niterated {i} times, current result is: \n    {x0}')  # TODO: this is not iteration
            i += 1
            if max(abs(x0 - self.ele_slices[0].closed_orbit)) < 1e-8:
                break
        print(f'\nclosed orbit at s=0 is \n    {x0}\n--------------')

    def track_close_orbit(self):
        print('\n-------------------\ntracking closed orbit:')
        xco = np.zeros(6)
        matrix = np.zeros([6, 6])
        resdl = 1
        j = 1
        beam = Beam(xco)
        while j <= 10 and resdl > 1e-16:
            beam = Beam(xco)
            for ele in self.ele_slices:
                ele.closed_orbit = np.array(beam.get_particle_array())
                beam = ele.real_track(beam)
            for i in range(6):
                matrix[:, i] = (beam.matrix[:, i] - beam.matrix[:, 6]) / beam.precision
            d = beam.matrix[:, 6] - xco
            dco = np.linalg.inv(np.identity(6) - matrix).dot(d)
            xco = xco + dco
            # dco = np.linalg.inv(matrix).dot(beam.matrix[:, 6])# TODO: why not Newton method?
            # xco = xco - dco
            resdl = dco.dot(dco.T)
            print(f'iterated {j} times, current result is \n    {beam.matrix[:, 6]}\n')
            j += 1
        print(f'closed orbit at s=0 is \n    {beam.matrix[:, 6]}\n')
        print(f'{matrix}\n')
        eig_val, eig_matrix = np.linalg.eig(matrix)
        for eig in eig_val:
            plt.scatter(np.real(eig), np.imag(eig), s=10, c='r')
        print(f'eig_val = {eig_val}')
        print(f'eig_vector = {eig_matrix[:, 4]}')
        damping = - np.log(np.abs(eig_val))
        print(f'damping  = {damping}')
        print(f'damping time = {1 / self.f_c / damping}')
        print('\ncheck:')
        print(f'sum damping = {damping[0] + damping[2] + damping[4]}, '
              f'2U0/E0 = {2 * self.U0 / Particle.energy}')
        print(f'\nring tune = {np.angle(eig_val) / 2 / pi}')
        print('\n--------------------------------------------\n')

    def solve_damping(self):
        matrix = np.identity(6)
        for ele in self.ele_slices:
            matrix = ele.damping_matrix.dot(matrix)
        eig_val, eig_matrix = np.linalg.eig(matrix)
        for eig in eig_val:
            plt.scatter(np.real(eig), np.imag(eig), s=10, c='b')
        self.damping = - np.log(np.abs(eig_val))
        print(f'{matrix}\n')
        print(f'eig_vals = {eig_val}')
        print(f'eig_vector = {eig_matrix[:, 4]}')
        print(f'damping  = {self.damping}')
        print(f'damping time = {1 / self.f_c / self.damping}')
        print('\ncheck:')
        print(f'sum damping = {self.damping[0] + self.damping[2] + self.damping[4]}, '
              f'2U0/E0 = {2 * self.U0 / Particle.energy}')
        print('\n--------------------------------------------\n')

    def along_ring(self):
        """solve tune along the ring"""
        matrix = np.identity(6)
        for ele in self.ele_slices:
            matrix = ele.matrix.dot(matrix)
        eig_val, ring_eig_matrix = np.linalg.eig(matrix)  # Ei is eig_matrix[:, i]  E_ki is eig_matrix[i, k]
        print(f'ring tune = {np.angle(eig_val) / 2 / pi}\n')
        # solve average decomposition and tune along the lattice
        ave_deco_square = np.zeros(6)
        sideways_photons = np.zeros(6)
        eig_matrix = ring_eig_matrix
        for ele in self.ele_slices:
            eig_matrix = ele.matrix.dot(eig_matrix)
            if ele.h != 0:
                for k in range(6):
                    ave_deco_square[k] += abs(eig_matrix[4, k]) ** 2 * abs(ele.h) ** 3 * ele.length
                    sideways_photons[k] += abs(eig_matrix[2, k]) ** 2 * abs(ele.h) ** 3 * ele.length
        for k in range(6):
            ave_deco_square[k] = ave_deco_square[k] * Cl * Particle.gamma ** 5 / c / self.damping[k]
            sideways_photons[k] = sideways_photons[k] * Cl * Particle.gamma ** 3 / c / self.damping[k] / 2
        # solve equilibrium beam
        eig_matrix = ring_eig_matrix
        for ele in self.ele_slices:
            equilibrium_beam = np.zeros((6, 6))
            for j in range(6):
                for i in range(j + 1):
                    for k in range(6):
                        equilibrium_beam[i, j] += ((ave_deco_square[k] + sideways_photons[k]) *
                                                   np.real(eig_matrix[i, k] * np.conj(eig_matrix[j, k])))
            for i in range(6):
                for j in range(i):
                    equilibrium_beam[i, j] = equilibrium_beam[j, i]
            ele.beam = deepcopy(equilibrium_beam)
            eig_matrix = ele.matrix.dot(eig_matrix)

    def print_tune(self):
        matrix = np.identity(6)
        for ele in self.ele_slices:
            matrix = ele.matrix.dot(matrix)
        eig_val, eig_matrix = np.linalg.eig(matrix)
        tune = np.angle(eig_val) / 2 / pi
        print(f'tune  = {tune}')

    def print_damping(self):
        matrix = np.identity(6)
        for ele in self.ele_slices:
            matrix = ele.damping_matrix.dot(matrix)
        eig_val, eig_matrix = np.linalg.eig(matrix)
        damping = - np.log(np.abs(eig_val))
        tune = np.angle(eig_val) / 2 / pi
        print('\nin damping matrix')
        print(f'tune = {tune}')
        print(f'damping  = {damping}')
        print('\n---------     assert ---------')
        print(f'sum damping = {damping[0] + damping[2] + damping[4]}, 2U0/E0 = {2 * self.U0 / Particle.energy}')

    def matrix_output(self, file_name: str = 'matrix.txt'):
        """output uncoupled matrix for each element and contained matrix"""

        matrix = np.identity(6)
        file = open(file_name, 'w')
        location = 0.0
        for ele in self.elements:
            file.write(f'{match_symbol(ele.symbol)} {ele.name} at s={location},  {ele.magnets_data()}\n')
            location = round(location + ele.length, LENGTH_PRECISION)
            file.write(str(ele.matrix) + '\n')
            file.write('contained matrix:\n')
            matrix = ele.matrix.dot(matrix)
            file.write(str(matrix))
            file.write('\n\n--------------------------\n\n')
        file.close()

    def coupled_matrix_output(self, filename: str = 'matrix.txt'):
        matrix = np.identity(6)
        element_matrix = np.identity(6)
        file = open(filename, 'w')
        location = 0.0
        first_ele = self.ele_slices[0]
        last_identifier = first_ele.identifier
        file.write(f'{match_symbol(first_ele.symbol)} {first_ele.name} at s={location} \n')
        file.write(f'closed orbit: \n    {first_ele.closed_orbit}\n')
        for ele in self.ele_slices:
            if ele.identifier != last_identifier:
                matrix = element_matrix.dot(matrix)
                file.write('element matrix:\n' + str(element_matrix) + '\n')
                file.write('contained matrix:\n')
                file.write(str(matrix))
                element_matrix = np.identity(6)
                file.write('\n\n--------------------------------------------\n\n')
                file.write(f'{match_symbol(ele.symbol)} {ele.name} at s={location} \n')
                file.write(f'closed orbit: {ele.closed_orbit}\n')
            element_matrix = ele.matrix.dot(element_matrix)
            location = round(location + ele.length, LENGTH_PRECISION)
            last_identifier = ele.identifier
        matrix = element_matrix.dot(matrix)
        file.write(str(element_matrix) + '\n')
        file.write('full matrix:\n')
        file.write(str(matrix))
        file.close()

    def output_equilibrium_beam(self, filename: str = 'equilibrium_beam.txt'):
        file = open(filename, 'w')
        location = 0.0
        last_identifier = 123465
        for ele in self.ele_slices:
            if ele.identifier != last_identifier:
                file.write(f'{match_symbol(ele.symbol)} {ele.name} at s={location} \n')
                file.write(f'closed orbit: {ele.closed_orbit}\n')
                file.write('equilibrium beam:\n')
                file.write(str(ele.beam))
                file.write('\n\n--------------------------------------------\n\n')
            location = round(location + ele.length, LENGTH_PRECISION)
            last_identifier = ele.identifier
        file.close()



