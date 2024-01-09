# -*- coding: utf-8 -*-
# cython: language_level=3
from .exceptions import ParticleLost, Unstable
from .c_functions cimport symplectic_track_ele
from .CSLattice import CSLattice
from .globalvars cimport refgamma, refbeta, refenergy, pi
from .c_functions cimport next_twiss
import numpy as np
cimport numpy as np


cdef extern from "<math.h>":
    double fabs(double x)


cdef extern from "<math.h>":
    double acos(double x)


cpdef symplectic_track(particle, lattice, int n_turns, record = True):
    """symplectic_track(particle, CSLattice lattice, int n_turns).
    particle is a list of float of length 6, or a np.ndarray.
    if record is True, Mark object in lattice will record coordinate data when the particle is passed.
    return ParticleLost exception if particle lost, else return None."""
    cdef double[6] p
    assert len(particle) == 6, 'illegal particle data'
    for i in range(6):
        p[i] = particle[i]
    if record is True:
        for k in lattice.mark:
            for m in lattice.mark[k]:
                m.clear()
                m.record = 1
    else:
        for k in lattice.mark:
            for m in lattice.mark[k]:
                m.clear()
                m.record = 0
    for i in range(n_turns):
        for j in range(lattice.n_periods):
            for ele in lattice.elements:
                flag = symplectic_track_ele(ele, p)
                if flag == -1:
                    raise ParticleLost(ele.s + lattice.length * i + lattice.length * j / lattice.n_periods)
    return np.array(p)

cpdef track_4d_closed_orbit(lattice: CSLattice, delta, matrix_precision=1e-9, resdl_limit=1e-12, verbose=True):
    """4D track to compute closed orbit with energy deviation.
    Args:
        delta: the momentum deviation.
        matrix_precision: the small deviation to calculate transfer matrix by tracking.
        resdl_limit: the limit to judge if the orbit is closed.

    Return:
        {'closed_orbit': (4, ) np.ndarray,
         'nux': float, decimal part
         'nuy': float, decimal part}

    reference: SAMM: Simple Accelerator Modelling in Matlab, A. Wolski, 2013
    """

    cdef double[6] particle0, particle1, particle2, particle3, particle4
    cdef double precision
    cdef np.ndarray[dtype=np.float64_t, ndim=2] matrix = np.zeros([4, 4])
    cdef double[:, :] mv = matrix

    if verbose:
        print('\n-------------------\ntracking 4D closed orbit:\n')
    xco = np.array([0.0, 0.0, 0.0, 0.0])
    resdl = 1
    iter_times = 1
    precision = matrix_precision
    d = np.zeros(4)
    while iter_times <= 10 and resdl > resdl_limit:
        particle0 = [0, 0, 0, 0, 0, 0]
        particle1 = [precision, 0, 0, 0, 0, 0]
        particle2 = [0, precision, 0, 0, 0, 0]
        particle3 = [0, 0, precision, 0, 0, 0]
        particle4 = [0, 0, 0, precision, 0, 0]
        for i in range(4):
            particle0[i] = particle0[i] + xco[i]
            particle1[i] = particle1[i] + xco[i]
            particle2[i] = particle2[i] + xco[i]
            particle3[i] = particle3[i] + xco[i]
            particle4[i] = particle4[i] + xco[i]
        particle0[5] = delta
        particle1[5] = delta
        particle2[5] = delta
        particle3[5] = delta
        particle4[5] = delta
        for nper in range(lattice.n_periods):
            for ele in lattice.elements:
                ele.closed_orbit = particle0
                flag0 = symplectic_track_ele(ele, particle0)
                flag1 = symplectic_track_ele(ele, particle1)
                flag2 = symplectic_track_ele(ele, particle2)
                flag3 = symplectic_track_ele(ele, particle3)
                flag4 = symplectic_track_ele(ele, particle4)
                if (flag0 + flag1 + flag2 + flag3 + flag4) != 0:
                    raise Exception(f'particle lost at {ele.s}')
        for i in range(4):
            mv[i, 0] = (particle1[i] - particle0[i]) / precision
            mv[i, 1] = (particle2[i] - particle0[i]) / precision
            mv[i, 2] = (particle3[i] - particle0[i]) / precision
            mv[i, 3] = (particle4[i] - particle0[i]) / precision
        for i in range(4):
            d[i] = particle0[i] - xco[i]
        dco = np.linalg.inv(np.identity(4) - matrix).dot(d)
        xco = xco + dco
        resdl = sum(dco ** 2) ** 0.5
        if verbose:
            print(f'iterated {iter_times} times, current result is \n    {particle0}\n')
        iter_times += 1
    if verbose:
        print(f'verify:\n    closed orbit at s=0 is \n    {xco}\n')
    cos_mu = (mv[0, 0] + mv[1, 1]) / 2
    if fabs(cos_mu) >= 1:
        raise Unstable('can not find period solution')
    nux = acos(cos_mu) * np.sign(mv[0, 1]) / 2 / pi
    cos_mu = (mv[2, 2] + mv[3, 3]) / 2
    if fabs(cos_mu) >= 1:
        raise Unstable('can not find period solution')
    nuy = acos(cos_mu) * np.sign(mv[2, 3]) / 2 / pi
    if verbose:
        print(f'tune is {nux - np.floor(nux):.6f}, {nuy - np.floor(nuy):.6f}')
    return_data = {'closed_orbit': xco[:4], 'nux': nux - np.floor(nux), 'nuy': nuy - np.floor(nuy)}
    return return_data
