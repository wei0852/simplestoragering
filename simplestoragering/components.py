# -*- coding: utf-8 -*-
"""Components:
Drift:
........."""

import numpy as np
from abc import ABCMeta, abstractmethod
from .globalvars import pi


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
    s = 0
    h = 0
    k1 = 0
    k2 = 0
    k3 = 0
    closed_orbit = np.zeros(6)
    beam = None
    betax = 0
    alphax = 0
    gammax = 0
    psix = 0
    betay = 0
    alphay = 0
    gammay = 0
    psiy = 0
    etax = 0
    etaxp = 0
    etay = 0
    etayp = 0

    @property
    @abstractmethod
    def matrix(self):
        """matrix with coupled effects to calculate tune and bunch distribution"""
        return np.identity(6)

    @abstractmethod
    def symplectic_track(self, beam):
        """assuming that the energy is constant, the result is symplectic."""
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

    def __init__(self, name: str):
        self.name = name
        self.data = None
        self.record = False

    @property
    def matrix(self):
        """matrix with coupled effects to calculate tune and bunch distribution"""
        return np.identity(6)

    def slice(self, n_slices: int) -> list:
        """slice component to element list, return ele_list"""
        ele_list = [self]
        return ele_list

    def symplectic_track(self, particle: np.ndarray):
        if self.record:
            if self.data is None:
                if particle.ndim == 2:
                    self.data = particle[:, 6]
                elif particle.ndim == 1:
                    self.data = particle
            else:
                if particle.ndim == 2:
                    self.data = np.vstack((self.data, particle[:, 6]))
                elif particle.ndim == 1:
                    self.data = np.vstack((self.data, particle))
        return particle

    def clear(self):
        """clear the particle coordinate data."""
        self.data = None

    def copy(self):
        return Mark(self.name)

    def __repr__(self):
        return f"Mark('{self.name}')"


class LineEnd(Element):
    """mark the end of a line, store the data at the end."""

    def __init__(self, s, name='_END_'):
        self.name = name
        self.s = s

    @property
    def matrix(self):
        """matrix with coupled effects to calculate tune and bunch distribution"""
        return np.identity(6)

    def slice(self, n_slices: int) -> list:
        return [self]

    def symplectic_track(self, beam):
        return beam

    def copy(self):
        return LineEnd(self.s, self.name)

    def __repr__(self):
        return f'LineEnd()'


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


def next_twiss(matrix, twiss0):
    sub = matrix[:2, :2]
    twiss1 = np.zeros(12)
    matrix_cal = np.array([[sub[0, 0] ** 2, -2 * sub[0, 0] * sub[0, 1], sub[0, 1] ** 2],
                           [-sub[0, 0] * sub[1, 0], 2 * sub[0, 1] * sub[1, 0] + 1, -sub[0, 1] * sub[1, 1]],
                           [sub[1, 0] ** 2, -2 * sub[1, 0] * sub[1, 1], sub[1, 1] ** 2]])
    twiss1[:3] = matrix_cal.dot(twiss0[:3])
    sub = matrix[2:4, 2:4]
    matrix_cal = np.array([[sub[0, 0] ** 2, -2 * sub[0, 0] * sub[0, 1], sub[0, 1] ** 2],
                           [-sub[0, 0] * sub[1, 0], 2 * sub[0, 1] * sub[1, 0] + 1, -sub[0, 1] * sub[1, 1]],
                           [sub[1, 0] ** 2, -2 * sub[1, 0] * sub[1, 1], sub[1, 1] ** 2]])
    twiss1[3:6] = matrix_cal.dot(twiss0[3:6])
    twiss1[6:8] = matrix[:2, :2].dot(twiss0[6:8]) + np.array([matrix[0, 5], matrix[1, 5]])
    twiss1[8:10] = matrix[2:4, 2:4].dot(twiss0[8:10]) + np.array([matrix[2, 5], matrix[3, 5]])
    dpsix = np.arctan(matrix[0, 1] / (matrix[0, 0] * twiss0[0] - matrix[0, 1] * twiss0[1]))
    while dpsix < 0:
        dpsix += pi
    twiss1[10] = twiss0[10] + dpsix
    dpsiy = np.arctan(matrix[2, 3] / (matrix[2, 2] * twiss0[3] - matrix[2, 3] * twiss0[4]))
    while dpsiy < 0:
        dpsiy += pi
    twiss1[11] = twiss0[11] + dpsiy
    return twiss1
