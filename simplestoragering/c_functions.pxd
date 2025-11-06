# -*- coding: utf-8 -*-
# cython: language_level=3

import numpy as np
cimport numpy as np
import time
from .globalvars cimport pi
from .components cimport Element

cdef symplectic_track_ele(Element ele, double[6] particle)

cdef radiation_track_ele(Element ele, double[6] particle)

cdef track_matrix(Element ele, double[6][6] matrix)

cdef next_twiss(double[6][6] matrix,double[12] data0, double[12] data)
