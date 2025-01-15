# -*- coding: utf-8 -*-
# cython: language_level=3

"""Components:
Drift:
........."""

cimport numpy as np
import numpy as np
from .globalvars import Cr, refgamma


cdef extern from "<math.h>":
    double atan(double x)

cdef extern from "<math.h>":
    double tan(double x)

cdef extern from "<math.h>":
    double sin(double x)

cdef extern from "<math.h>":
    double cos(double x)

cdef extern from "<math.h>":
    double sinh(double x)

cdef extern from "<math.h>":
    double cosh(double x)

cdef extern from "<math.h>":
    double sqrt(double x)

cdef extern from "<math.h>":
    double pow(double x, double y)


cdef class Element():
    """parent class of magnet components

    Attributes:
        name: str 
        length: double
        type: str, type of component.
        s: double, location in a line or ring.
        h: double, curvature of the reference trajectory.
        theta_in, theta_out: double, edge angle of Bends.
        k1, k2, k3: double, multipole strength. (q/P0) (\partial^n B_y)/(\partial x^n)
        Ax, Ay: double, aperture.
        n_slices: int, slice magnets for RDTs calculation and particle tracking.
        closed_orbit: double[6]
        betax, alphax, gammax, psix, betay, alphay, gammay, psiy, etax, etaxp, etay, etayp: double, twiss parameters and dispersion.
        nux, nuy: double, psix / 2 pi, psiy / 2 pi
        curl_H: double, dispersion H-function (curly-H function).
        matrix: 6x6 np.ndarray, transport matrix.

    Methods:
        copy()
        slice(n_slices: int) -> list[Elements]
        linear_optics() -> np.array([i1, i2, i3, i4, i5, xix, xiy]), 
                           np.array([betax, alphax, gammax, betay, alphay, gammay, etax, etaxp, etay, etayp, psix, psiy])
    """
    cdef public str name
    cdef public double length
    cdef public double s
    cdef public double h
    cdef public double k1
    cdef public double k2
    cdef public double k3
    cdef public double Ax, Ay
    cdef public int n_slices
    cdef public double[6] closed_orbit
    cdef public double betax
    cdef public double alphax
    cdef public double gammax
    cdef public double psix
    cdef public double betay
    cdef public double alphay
    cdef public double gammay
    cdef public double psiy
    cdef public double etax
    cdef public double etaxp
    cdef public double etay
    cdef public double etayp

    cpdef copy(self)

    cdef int symplectic_track(self, double[6] particle)

    cpdef linear_optics(self)

    cpdef off_momentum_optics(self, delta)


cdef class Mark(Element):
    cdef public np.ndarray data
    cdef public bint record
    cpdef copy(self)

    cdef int symplectic_track(self, double[6] particle)


cdef class LineEnd(Element):
    cpdef copy(self)

    cdef int symplectic_track(self, double[6] particle)

cdef assin_twiss(Element ele,double[12] twiss)
