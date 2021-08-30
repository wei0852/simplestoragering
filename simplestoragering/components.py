# -*- coding: utf-8 -*-
"""Components:
Drift:
........."""

from .particles import Beam7
import numpy as np
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from .constants import pi, LENGTH_PRECISION
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
    nux = None
    betay = None
    alphay = None
    gammay = None
    nuy = None
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
        """:return nux, nuy"""
        dpsix = np.arctan(self.matrix[0, 1] / (self.matrix[0, 0] * self.betax - self.matrix[0, 1] * self.alphax))
        nux = self.nux + dpsix / 2 / pi
        dpsiy = np.arctan(self.matrix[2, 3] / (self.matrix[2, 2] * self.betay - self.matrix[2, 3] * self.alphay))
        nuy = self.nuy + dpsiy / 2 / pi
        return nux, nuy

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

    def slice(self, initial_s, identifier):
        """slice component to element list, return [ele_list, final_z], the identifier identifies different magnet"""
        ele_list = []
        current_s = initial_s
        self.identifier = identifier
        self.s = current_s
        ele_list.append(self)
        return [ele_list, current_s]

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
        raise UnfinishedWork('VBend')

    def symplectic_track(self, beam):
        raise UnfinishedWork('VBend')

    def real_track(self, beam: Beam7) -> Beam7:
        raise UnfinishedWork('VBend')


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
        # lambda_q = Cr * RefParticle.energy ** 3 * self.k1 ** 2 * self.length / pi
        raise UnfinishedWork()

    @property
    def closed_orbit_matrix(self):
        raise UnfinishedWork()

    def symplectic_track(self, beam):
        raise UnfinishedWork('SKQ')

    def real_track(self, beam: Beam7) -> Beam7:
        raise UnfinishedWork('SKQ')
