# -*- coding: utf-8 -*-
"""constants:
pi
c: speed of light
Cr
Cq
Cl:
LENGTH_PRECISION: the length precision of components."""
from scipy.constants import pi, c, physical_constants


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


me = physical_constants['electron mass energy equivalent in MeV'][0]
Cr, Cq, Cl = calculate_constants()
LENGTH_PRECISION = 10


