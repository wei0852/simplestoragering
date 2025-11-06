# -*- coding: utf-8 -*-
# cython: language_level=3

import numpy as np
cimport numpy as np
import time
from .globalvars cimport pi, refenergy
from .components cimport Element

cdef extern from "<math.h>":
    double atan(double x)

cdef symplectic_track_ele(Element ele, double[6] particle):
    flag = ele.symplectic_track(particle)
    if flag == -1:
        return -1
    else:
        return 0

cdef radiation_track_ele(Element ele, double[6] particle):
    flag = ele.radiation_track(particle)
    if flag == -1:
        return -1
    else:
        return 0

cdef track_matrix(Element ele, double[6][6] matrix):
    cdef double precision=3e-8
    cdef double[6] p1 = ele.closed_orbit
    cdef double[6] p2 = ele.closed_orbit
    cdef double[6] p3 = ele.closed_orbit
    cdef double[6] p4 = ele.closed_orbit
    cdef double[6] p5 = ele.closed_orbit
    cdef double[6] p6 = ele.closed_orbit
    cdef double[6] p0 = ele.closed_orbit
    p1[0] += precision
    p2[1] += precision
    p3[2] += precision
    p4[3] += precision
    p5[4] += precision
    p6[5] += precision
    flag0 = ele.symplectic_track(p0)
    flag1 = ele.symplectic_track(p1)
    flag2 = ele.symplectic_track(p2)
    flag3 = ele.symplectic_track(p3)
    flag4 = ele.symplectic_track(p4)
    flag5 = ele.symplectic_track(p5)
    flag6 = ele.symplectic_track(p6)
    if (flag0 + flag1 + flag2 + flag3 + flag4 + flag6) != 0:
        return -1
    for i in range(6):
        matrix[i][0] = (p1[i] - p0[i]) / precision
        matrix[i][1] = (p2[i] - p0[i]) / precision
        matrix[i][2] = (p3[i] - p0[i]) / precision
        matrix[i][3] = (p4[i] - p0[i]) / precision
        matrix[i][4] = (p5[i] - p0[i]) / precision
        matrix[i][5] = (p6[i] - p0[i]) / precision
    
cdef next_twiss(double[6][6] matrix,double[12] data0, double[12] data):
    # x direction
    cdef double dpsix, dpsiy, psix, psiy
    cdef double[3][3] mct

    mct[0][0] = pow(matrix[0][0], 2)
    mct[0][1] = -2 * matrix[0][0] * matrix[0][1]
    mct[0][2] = pow(matrix[0][1], 2)
    mct[1][0] = -matrix[0][0] * matrix[1][0]
    mct[1][1] = 2 * matrix[0][1] * matrix[1][0] + 1
    mct[1][2] = -matrix[0][1] * matrix[1][1]
    mct[2][0] = pow(matrix[1][0], 2)
    mct[2][1] = -2 * matrix[1][0] * matrix[1][1]
    mct[2][2] = pow(matrix[1][1], 2)
    data[0] = mct[0][0] * data0[0] + mct[0][1] * data0[1] + mct[0][2] * data0[2]
    data[1] = mct[1][0] * data0[0] + mct[1][1] * data0[1] + mct[1][2] * data0[2]
    data[2] = mct[2][0] * data0[0] + mct[2][1] * data0[1] + mct[2][2] * data0[2]

    # y direction

    mct[0][0] = matrix[2][2] ** 2
    mct[0][1] = -2 * matrix[2][2] * matrix[2][3]
    mct[0][2] = matrix[2][3] ** 2
    mct[1][0] = -matrix[2][2] * matrix[3][2]
    mct[1][1] = 2 * matrix[2][3] * matrix[3][2] + 1
    mct[1][2] = -matrix[2][3] * matrix[3][3]
    mct[2][0] = matrix[3][2] ** 2
    mct[2][1] = -2 * matrix[3][2] * matrix[3][3]
    mct[2][2] = matrix[3][3] ** 2
    data[3] = mct[0][0] * data0[3] + mct[0][1] * data0[4] + mct[0][2] * data0[5]
    data[4] = mct[1][0] * data0[3] + mct[1][1] * data0[4] + mct[1][2] * data0[5]
    data[5] = mct[2][0] * data0[3] + mct[2][1] * data0[4] + mct[2][2] * data0[5]

    # etax
    data[6] = matrix[0][0] * data0[6] + matrix[0][1] * data0[7] + matrix[0][5]
    data[7] = matrix[1][0] * data0[6] + matrix[1][1] * data0[7] + matrix[1][5]

    # etay
    data[8] = matrix[2][2] * data0[8] + matrix[2][3] * data0[9]  + matrix[2][5]
    data[9] = matrix[3][2] * data0[8] + matrix[3][3] * data0[9]  + matrix[3][5]

    # phase
    dpsix = atan(matrix[0][1] / (matrix[0][0] * data0[0] - matrix[0][1] * data0[1]))
    while dpsix < 0:
        dpsix += pi
    data[10] = data0[10]  + dpsix
    dpsiy = atan(matrix[2][3] / (matrix[2][2] * data0[3] - matrix[2][3] * data0[4]))
    while dpsiy < 0:
        dpsiy += pi
    data[11] = data0[11]  + dpsiy

