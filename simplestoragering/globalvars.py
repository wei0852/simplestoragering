# -*- coding: utf-8 -*-
"""global variables"""
from scipy.constants import physical_constants


LENGTH_PRECISION = 10

c = 299792458.0
pi = 3.141592653589793
me = 0.51099895
Cr = 8.846273822420376e-14
Cl = 2.1581408349289595e-19
Cq = 3.831938640893901e-13


def set_ref_energy(energy_MeV):
    """set reference energy in MeV"""
    RefParticle.set_energy(energy_MeV)


class RefParticle(object):
    """set particle's type and energy"""
    energy = None
    gamma = None
    beta = None
    rigidity = None

    @classmethod
    def set_energy(cls, energy):
        """MeV"""
        cls.energy = energy
        cls.gamma = cls.energy / me + 1
        cls.beta = (1 - 1 / cls.gamma ** 2) ** 0.5
        cls.rigidity = cls.gamma * me * cls.beta * 1e6 / c

    @classmethod
    def __str__(cls):
        return f"Electron, gamma = {cls.gamma: .3f}"


def calculate_beta(delta):
    """calculate beta of particle"""

    gamma = RefParticle.gamma * (delta * RefParticle.beta + 1)
    beta = (1 - 1 / gamma ** 2) ** 0.5
    return beta


def calculate_constants():
    """Andrzej Wolski (2014)"""

    h_bar = physical_constants['natural unit of action in eV s'][0]
    re = physical_constants['classical electron radius'][0]
    # Cr = q ** 2 / (3 * epsilon_0 * (m * c**2) ** 4)   p.221
    cr = 4 * pi * re / (3 * me ** 3)  # this is only for electrons. unit m/MeV**3
    cq = 55 * h_bar * c / (32 * 3 ** 0.5 * me * 1e6)  # p.232  unit s
    # this part is to verify cl
    # Pr = (Cr * c / 2 / pi) * (\beta_0^4 * E_0^4 / \rho^2)    p.221
    # <\dot{N} u^2> = 2 Cq \gamma_0^2 E_0 P_r / \rho          p.232
    # ==>  <\dot{N} u^2> = Cq Cr c E0^5 \gamma^2 \beta^4 / \pi \rho^3
    # <\dot{N} u^2 / E_0^2> = 2 Cl \gamma^5 / |\rho|^3  Slim Formalism orbit motion
    # therefore,
    # cl = cq * cr * c * me ** 3 / 2 / pi  # \beta \approx
    cl = 55 * re * h_bar * c ** 2 / (48 * 3 ** 0.5 * me * 1e6)
    return cr, cq, cl
