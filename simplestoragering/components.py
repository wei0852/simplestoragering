# -*- coding: utf-8 -*-
"""Components:
Drift:
........."""

from .particles import Beam7
import numpy as np
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from .constants import pi, LENGTH_PRECISION
from .particles import RefParticle


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

    # matrix = None

    @property
    @abstractmethod
    def damping_matrix(self):
        """matrix with coupled effects and damping rates to calculate tune and damping rate"""
        pass

    @property
    @abstractmethod
    def matrix(self):
        """matrix with coupled effects to calculate tune and bunch distribution"""
        return np.identity(6)

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

    @abstractmethod
    def copy(self):
        """calculate integral parameters

        return: I1, I2, I3, I4, I5, xi_x, xi_y"""
        pass

    @property
    def type(self):
        """return the type of the component"""

        return self.__class__.__name__

    @property
    def nux(self):
        return self.psix / 2 / pi

    @property
    def nuy(self):
        return self.psiy / 2 / pi

    @abstractmethod
    def slice(self, n_slices: int) -> list:
        """Slice a component into a list of elements. twiss data will be calculated for each element.

        return: ele_list"""
        pass

    def linear_optics(self):
        twiss0 = np.array(
            [self.betax, self.alphax, self.gammax, self.betay, self.alphay, self.gammay, self.etax, self.etaxp,
             self.etay, self.etayp, self.psix, self.psiy])
        return np.zeros(7), twiss0

    def __repr__(self):
        return self.name

    def __str__(self):
        text = str(self.name)
        text += (' ' * max(0, 6 - len(str(self.name))))
        text += (': ' + self.type)
        text += (':   s = ' + str(self.s))
        text += f',   length = {self.length: .6f}'
        if self.h != 0:
            theta = self.length * self.h * 180 / pi
            text += f',   theta = {theta: .6f}'
        if self.k1 != 0:
            text += f',   k1 = {self.k1: .6f}'
        if self.k2 != 0:
            text += f',   k2 = {self.k2: .6f}'
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
    """mark class. If record is True, the coordinates will be recorded every time the particle passes the Mark."""

    def __init__(self, name: str, record=True):
        self.name = name
        self.n_slices = 1
        self.data = None
        self.record = record

    @property
    def matrix(self):
        """matrix with coupled effects to calculate tune and bunch distribution"""
        return np.identity(6)

    def slice(self, n_slices: int) -> list:
        """slice component to element list, return [ele_list, final_z], the identifier identifies different magnet"""
        ele_list = [self]
        return ele_list

    @property
    def damping_matrix(self):
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
        if self.record:
            if self.data is None:
                self.data = beam.get_particle_array()
            else:
                self.data = np.vstack((self.data, beam.get_particle_array()))
        return beam

    def real_track(self, beam: Beam7) -> Beam7:
        if self.record:
            if self.data is None:
                self.data = beam.get_particle_array()
            else:
                self.data = np.vstack((self.data, beam.get_particle_array()))
        return beam

    def clear(self):
        """clear the particle coordinate data."""
        self.data = None

    def copy(self):
        return Mark(self.name, self.record)


class LineEnd(Element):
    """mark the end of a line, store the data at the end."""

    def __init__(self, s, identifier, name='_END_'):
        self.name = name
        self.s = s
        self.identifier = identifier
        self.n_slices = 1

    @property
    def matrix(self):
        """matrix with coupled effects to calculate tune and bunch distribution"""
        return np.identity(6)

    def slice(self, n_slices: int) -> list:
        return [self]

    @property
    def damping_matrix(self):
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

    def copy(self):
        return LineEnd(self.s, self.identifier, self.name)


def assin_twiss(ele: Element, twiss):
    ele.betax = twiss[0]
    ele.alphax = twiss[1]
    ele.gammax = twiss[2]
    ele.betay = twiss[3]
    ele.alphay = twiss[4]
    ele.gammay = twiss[5]
    ele.etax = twiss[6]
    ele.etaxp = twiss[7]
    ele.etay = twiss[8]
    ele.etayp = twiss[9]
    ele.psix = twiss[10]
    ele.psiy = twiss[11]
