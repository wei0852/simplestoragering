# -*- coding: utf-8 -*-
# cython: language_level=3
import numpy as np
cimport numpy as np
from .components cimport Element
from .Drift cimport drift_matrix
from .HBend cimport hbend_matrix
from .Quadrupole cimport quad_matrix


cpdef line_matrix(list ele_list)

cdef int ele_matrix(Element ele, double[6][6] matrix)
