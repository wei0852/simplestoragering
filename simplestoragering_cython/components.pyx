# -*- coding: utf-8 -*-
# cython: language_level=3
"""Components:
Drift:
........."""

cimport numpy as np
import numpy as np
from .globalvars cimport Cr, refgamma, pi


cdef class Element():
    """parent class of magnet components

    Attributes:
        name: str 
        length: float
        type: str, type of component.
        s: float, location in a line or ring.
        h: float, curvature of the reference trajectory.
        theta_in, theta_out: float, edge angle of Bends.
        k1, k2, k3: float, multipole strength. (q/P0) (\partial^n B_y)/(\partial x^n)
        Ax, Ay: float, aperture.
        n_slices: int, slice magnets for RDTs calculation and particle tracking.
        closed_orbit: float[6]
        betax, alphax, gammax, psix, betay, alphay, gammay, psiy, etax, etaxp, etay, etayp: float, twiss parameters and dispersion.
        nux, nuy: float, psix / 2 pi, psiy / 2 pi
        curl_H: float, dispersion H-function (curly-H function).
        matrix: 6x6 np.ndarray, transport matrix.

    Methods:
        copy(), return a same component without data about beam or lattice. 
        slice(n_slices: int) -> list[Elements]
        linear_optics() -> np.array([i1, i2, i3, i4, i5, xix, xiy]), 
                           np.array([betax, alphax, gammax, betay, alphay, gammay, etax, etaxp, etay, etayp, psix, psiy])
    """

    @property
    def matrix(self):
        """matrix with coupled effects to calculate tune and bunch distribution"""
        return np.identity(6)

    cdef int symplectic_track(self, double[6] particle):
        """assuming that the energy is constant, the result is symplectic."""
        pass

    cpdef copy(self):
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

    def slice(self, n_slices: int) -> list:
        """Slice a component into a list of elements. twiss data will be calculated for each element.

        return: ele_list"""
        pass

    cpdef linear_optics(self):
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


cdef class Mark(Element):
    """Mark(name)
    
    Attributes:
        data: nx6 np.ndarray.
    The coordinates will be recorded every time the particle passes the Mark in function symplectic_track()"""

    def __init__(self, name: str):
        self.name = name
        self.n_slices = 1
        self.data = None
        self.record = 0

    @property
    def matrix(self):
        """matrix with coupled effects to calculate tune and bunch distribution"""
        return np.identity(6)

    def slice(self, n_slices: int) -> list:
        """slice component to element list, return ele_list"""
        ele_list = [self]
        return ele_list

    cdef int symplectic_track(self, double[6] particle):
        if self.record:
            p = np.array([particle[0], particle[1], particle[2], particle[3], particle[4], particle[5]])
            if self.data is None:
                self.data = p
            else:
                self.data = np.vstack((self.data, p))
        return 0

    def clear(self):
        """clear the particle coordinate data."""
        self.data = None

    cpdef copy(self):
        return Mark(self.name)


cdef class LineEnd(Element):
    """mark the end of a line, store the data at the end."""

    def __init__(self, s, name='_END_'):
        self.name = name
        self.s = s
        self.n_slices = 1

    @property
    def matrix(self):
        """matrix with coupled effects to calculate tune and bunch distribution"""
        return np.identity(6)

    def slice(self, n_slices: int) -> list:
        return [self]

    cdef int symplectic_track(self, double[6] particle):
        return 0

    cpdef copy(self):
        return LineEnd(self.s, self.name)


cdef assin_twiss(Element ele,double[12] twiss):
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
