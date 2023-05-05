# -*- coding: utf-8 -*-
# cython: language_level=3

cdef double Cq, Cr, Cl, c, pi, me_MeV
cdef double refgamma, refbeta, refenergy, rigidity


cpdef cq()

cpdef cr()

cpdef cl()

cpdef refEnergy()

cpdef set_ref_gamma(double gamma)

cpdef set_ref_energy(double energy_MeV)

cpdef double calculate_beta(double delta)

cdef extern from "<math.h>":
    double sqrt(double x)

cdef extern from "<math.h>":
    double pow(double x, double y)
    