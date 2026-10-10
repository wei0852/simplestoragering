# -*- coding: utf-8 -*-
# cython: language_level=3
from .exceptions import ParticleLost, Unstable
from .c_functions cimport symplectic_track_ele, radiation_track_ele
from .CSLattice import CSLattice
from .components cimport Mark
from .globalvars cimport refgamma, refbeta, refenergy, pi
from .c_functions cimport next_twiss
import numpy as np
cimport numpy as np
from libc.math cimport sin, acos, fabs


cdef list _prepare_mark_recording(lattice, Py_ssize_t n_turns, bint record):
    """Configure every Mark once, outside the element-by-element hot loop."""
    cdef list marks = []
    cdef Mark mark
    cdef object mark_group
    cdef Py_ssize_t n_records = n_turns * lattice.n_periods
    if n_records < 0:
        n_records = 0

    for mark_group in lattice.mark.values():
        for mark in mark_group:
            marks.append(mark)
            if record:
                mark.prepare_recording(n_records)
            else:
                mark.clear()
                mark.record = 0
    return marks


cdef void _finish_mark_recording(list marks, bint record):
    cdef Mark mark
    if record:
        for mark in marks:
            mark.finish_recording()


cpdef symplectic_track(particle, lattice, n_turns: int, record = True):
    """symplectic_track(particle, lattice, n_turns: int, record = True).
    particle is a list of float of length 6, or a np.ndarray.
    if record is True, Mark object in lattice will record coordinate data when the particle is passed.
    return ParticleLost exception if particle lost, else return the final coordinate."""
    cdef double[6] p
    assert len(particle) == 6, 'illegal particle data'
    for i in range(6):
        p[i] = particle[i]
    if isinstance(lattice, CSLattice):
        marks = _prepare_mark_recording(lattice, n_turns, record is True)
        try:
            for i in range(n_turns):
                for j in range(lattice.n_periods):
                    for ele in lattice.elements:
                        flag = symplectic_track_ele(ele, p)
                        if flag == -1:
                            raise ParticleLost(ele.s + lattice.length * i + lattice.length * j / lattice.n_periods)
        finally:
            _finish_mark_recording(marks, record is True)
    else:
        for i in range(n_turns):
            for ele in lattice:
                flag = symplectic_track_ele(ele, p)
                if flag == -1:
                    raise ParticleLost(ele.name)
    return np.array(p)

cpdef radiation_track(particle, lattice, n_turns: int, record = True):
    cdef double[6] p
    assert len(particle) == 6, 'illegal particle data'
    for i in range(6):
        p[i] = particle[i]
    if isinstance(lattice, CSLattice):
        lattice.RFCavity.enable = 1
        marks = _prepare_mark_recording(lattice, n_turns, record is True)
        try:
            for i in range(n_turns):
                for j in range(lattice.n_periods):
                    for ele in lattice.elements:
                        flag = radiation_track_ele(ele, p)
                        if flag == -1:
                            raise ParticleLost(ele.s + lattice.length * i + lattice.length * j / lattice.n_periods)
        finally:
            _finish_mark_recording(marks, record is True)
    else:
        for i in range(n_turns):
            for ele in lattice:
                flag = radiation_track_ele(ele, p)
                if flag == -1:
                    raise ParticleLost(ele.name)
    return np.array(p)    

cpdef track_4d_closed_orbit(lattice: CSLattice, delta, verbose=True, matrix_precision=1e-9, resdl_limit=1e-12):
    """track_4d_closed_orbit(lattice: CSLattice, delta, verbose=True, matrix_precision=1e-9, resdl_limit=1e-12)
    4D track to compute closed orbit with energy deviation.
    
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

    cdef double[6] particle0, particle1, particle2, particle3, particle4, particle5
    cdef double precision
    cdef np.ndarray[dtype=np.float64_t, ndim=2] matrix = np.zeros([5, 5])
    cdef double[:, :] mv = matrix
    
    if verbose:
        print('\n-------------------\ntracking 4D closed orbit:\n')
    xco = np.array([0.0, 0.0, 0.0, 0.0])
    resdl = 1
    iter_times = 1
    precision = matrix_precision
    d = np.zeros(4)
    while iter_times <= 10 and resdl > resdl_limit:
        particle0 = [0, 0, 0, 0, 0, delta]
        particle1 = [precision, 0, 0, 0, 0, delta]
        particle2 = [0, precision, 0, 0, 0, delta]
        particle3 = [0, 0, precision, 0, 0, delta]
        particle4 = [0, 0, 0, precision, 0, delta]
        particle5 = [0, 0, 0, 0, 0 ,precision + delta]
        for i in range(4):
            particle0[i] = particle0[i] + xco[i]
            particle1[i] = particle1[i] + xco[i]
            particle2[i] = particle2[i] + xco[i]
            particle3[i] = particle3[i] + xco[i]
            particle4[i] = particle4[i] + xco[i]
            particle5[i] = particle5[i] + xco[i]
        for nper in range(lattice.n_periods):
            for ele in lattice.elements:
                ele.closed_orbit = particle0
                flag0 = symplectic_track_ele(ele, particle0)
                flag1 = symplectic_track_ele(ele, particle1)
                flag2 = symplectic_track_ele(ele, particle2)
                flag3 = symplectic_track_ele(ele, particle3)
                flag4 = symplectic_track_ele(ele, particle4)
                flag5 = symplectic_track_ele(ele, particle5)
                if (flag0 + flag1 + flag2 + flag3 + flag4 + flag5) != 0:
                    raise Unstable(f'particle lost at {ele.s}')
        for i in range(4):
            mv[i, 0] = (particle1[i] - particle0[i]) / precision
            mv[i, 1] = (particle2[i] - particle0[i]) / precision
            mv[i, 2] = (particle3[i] - particle0[i]) / precision
            mv[i, 3] = (particle4[i] - particle0[i]) / precision
            mv[i, 4] = (particle5[i] - particle0[i]) / precision
        for i in range(4):
            d[i] = particle0[i] - xco[i]
        dco = np.linalg.solve(np.identity(4) - matrix[:4, :4], d)
        xco = xco + dco
        resdl = sum(dco ** 2) ** 0.5
        if verbose:
            print(f'iterated {iter_times} times, current result is \n    {particle0}\n')
        iter_times += 1
    if verbose:
        print(f'verify:\n    closed orbit at s=0 is \n    {xco}\n')
    # x
    cos_mu = (mv[0, 0] + mv[1, 1]) / 2
    if fabs(cos_mu) >= 1:
        raise Unstable('can not find period solution')
    mu = acos(cos_mu) * np.sign(mv[0, 1])
    nux = mu / 2 / pi
    # y
    cos_mu = (mv[2, 2] + mv[3, 3]) / 2
    if fabs(cos_mu) >= 1:
        raise Unstable('can not find period solution')
    mu = acos(cos_mu) * np.sign(mv[2, 3])
    nuy = mu / 2 / pi
    if verbose:
        print(f'tune is {nux - np.floor(nux):.6f}, {nuy - np.floor(nuy):.6f}')
    return_data = {'closed_orbit': xco[:4], 'nux': nux - np.floor(nux), 'nuy': nuy - np.floor(nuy)}
    return return_data


cpdef track_6d_closed_orbit(lattice: CSLattice, verbose=True, matrix_precision=1e-9, resdl_limit=1e-12):
    """track_6d_closed_orbit(lattice: CSLattice, verbose=True, matrix_precision=1e-9, resdl_limit=1e-12)
    6D tracking to compute the closed orbit.
    
    Args:
        matrix_precision: the small deviation to calculate transfer matrix by tracking.
        resdl_limit: the limit to judge if the orbit is closed.
    
    Return:
        {'closed_orbit': (6, ) np.ndarray, 
         'nux': float, decimal part 
         'nuy': float, decimal part}

    reference: SAMM: Simple Accelerator Modelling in Matlab, A. Wolski, 2013
    """

    cdef double[6] particle0, particle1, particle2, particle3, particle4, particle5, particle6
    cdef double precision
    cdef np.ndarray[dtype=np.float64_t, ndim=2] matrix = np.zeros([6, 6])
    cdef double[:, :] mv = matrix
    
    if verbose:
        print('\n-------------------\ntracking 6D closed orbit:\n')
    xco = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    resdl = 1
    iter_times = 1
    precision = matrix_precision
    d = np.zeros(6)
    while iter_times <= 10 and resdl > resdl_limit:
        particle0 = [0, 0, 0, 0, 0, 0]
        particle1 = [precision, 0, 0, 0, 0, 0]
        particle2 = [0, precision, 0, 0, 0, 0]
        particle3 = [0, 0, precision, 0, 0, 0]
        particle4 = [0, 0, 0, precision, 0, 0]
        particle5 = [0, 0, 0, 0 ,precision, 0]
        particle6 = [0, 0, 0, 0, 0 ,precision]
        for i in range(6):
            particle0[i] = particle0[i] + xco[i]
            particle1[i] = particle1[i] + xco[i]
            particle2[i] = particle2[i] + xco[i]
            particle3[i] = particle3[i] + xco[i]
            particle4[i] = particle4[i] + xco[i]
            particle5[i] = particle5[i] + xco[i]
            particle6[i] = particle6[i] + xco[i]
        for nper in range(lattice.n_periods):
            for ele in lattice.elements:
                ele.closed_orbit = particle0
                flag0 = radiation_track_ele(ele, particle0)
                flag1 = radiation_track_ele(ele, particle1)
                flag2 = radiation_track_ele(ele, particle2)
                flag3 = radiation_track_ele(ele, particle3)
                flag4 = radiation_track_ele(ele, particle4)
                flag5 = radiation_track_ele(ele, particle5)
                flag6 = radiation_track_ele(ele, particle6)
                if (flag0 + flag1 + flag2 + flag3 + flag4 + flag5 + flag6) != 0:
                    raise Unstable(f'particle lost at {ele.s}')
        for i in range(6):
            mv[i, 0] = (particle1[i] - particle0[i]) / precision
            mv[i, 1] = (particle2[i] - particle0[i]) / precision
            mv[i, 2] = (particle3[i] - particle0[i]) / precision
            mv[i, 3] = (particle4[i] - particle0[i]) / precision
            mv[i, 4] = (particle5[i] - particle0[i]) / precision
            mv[i, 5] = (particle6[i] - particle0[i]) / precision
        for i in range(6):
            d[i] = particle0[i] - xco[i]
        dco = np.linalg.solve(np.identity(6) - matrix, d)
        xco = xco + dco
        resdl = sum(dco ** 2) ** 0.5
        if verbose:
            print(f'iterated {iter_times} times, current result is \n    {particle0}\n')
        iter_times += 1
    if verbose:
        print(f'verify:\n    closed orbit at s=0 is \n    {xco}\n')
    # x
    cos_mu = (mv[0, 0] + mv[1, 1]) / 2
    if fabs(cos_mu) >= 1:
        raise Unstable('can not find period solution')
    mu = acos(cos_mu) * np.sign(mv[0, 1])
    nux = mu / 2 / pi
    # y
    cos_mu = (mv[2, 2] + mv[3, 3]) / 2
    if fabs(cos_mu) >= 1:
        raise Unstable('can not find period solution')
    mu = acos(cos_mu) * np.sign(mv[2, 3])
    nuy = mu / 2 / pi
    if verbose:
        print(f'tune is {nux - np.floor(nux):.6f}, {nuy - np.floor(nuy):.6f}')
    return_data = {'closed_orbit': xco, 'nux': nux - np.floor(nux), 'nuy': nuy - np.floor(nuy)}
    return return_data
