# -*- coding: utf-8 -*-
# cython: language_level=3
"""global variables"""


cdef double Cq = 3.831938640893901e-13
cdef double Cr = 8.846273822420376e-14 
cdef double Cl = 2.1581408349289595e-19
cdef double c = 299792458.0
cdef double pi = 3.141592653589793
cdef double me_MeV = 510.99895069e-3


cpdef cq():
    return Cq

cpdef cr():
    return Cr

cpdef cl():
    return Cl

cpdef refEnergy():
    return refenergy

cpdef set_ref_gamma(double gamma):
    """set gamma of reference particle.
    
    set_ref_gamma(gamma: float)"""
    global refbeta, refenergy, refgamma, rigidity
    refgamma = gamma
    refenergy = me_MeV * gamma
    refbeta = sqrt(1 - 1 / pow(gamma, 2))
    rigidity = gamma * me_MeV * refbeta * 1e6 / c

cpdef set_ref_energy(double energy_MeV):
    """set energy of reference particle.
    
    set_ref_energy(energy_MeV: float)"""
    global refbeta, refenergy, refgamma, rigidity
    refenergy = energy_MeV
    refgamma = energy_MeV / me_MeV
    refbeta = sqrt(1 - 1 / pow(refgamma, 2))
    rigidity = refgamma * me_MeV * refbeta * 1e6 / c


cpdef double calculate_beta(double delta):
    """calculate beta of particle"""

    gamma = refgamma * (delta * refbeta + 1)
    beta = (1 - 1 / gamma ** 2) ** 0.5
    return beta

