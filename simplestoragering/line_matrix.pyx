# -*- coding: utf-8 -*-
# cython: language_level=3
import numpy as np
cimport numpy as np
from .components cimport Element
from .Drift cimport drift_matrix
from .HBend cimport hbend_matrix
from .Quadrupole cimport quad_matrix

cpdef line_matrix(list ele_list):
    """line_matrix(ele_list: list[Element])
    
    return 6X6 transfer matrix"""
    cdef double[6][6] matrix=[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
    for ele in ele_list:
        ele_matrix(ele, matrix)
    return np.array(matrix)

cdef int ele_matrix(Element ele, double[6][6] matrix):
    cdef double[6][6] ele_matrix
    cdef double[6][6] temp_matrix
    if ele.type == 'Drift' or ele.type == 'Sextupole':
        drift_matrix(ele_matrix, ele.length)
    elif ele.type == 'HBend':
        hbend_matrix(ele_matrix, ele.length, ele.h, ele.theta_in, ele.theta_out, ele.k1, ele.gap, ele.fint_in, ele.fint_out)
    elif ele.type == 'Quadrupole':
        quad_matrix(ele_matrix, ele.length, ele.k1)
    elif ele.type == 'Octupole':
        drift_matrix(ele_matrix, ele.length)
    else:
        ele_matrix = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
     
    for i in range(6):
        for j in range(6):
            temp_matrix[i][j] = matrix[i][j]

    for i in range(6):
        for j in range(6):
            matrix[i][j] = 0
            for k in range(6):
                matrix[i][j] += ele_matrix[i][k] * temp_matrix[k][j]
    return 0
