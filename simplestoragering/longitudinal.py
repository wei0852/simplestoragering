"""Momentum compaction and phase slip calculations."""
import numpy as np
from .track import symplectic_track, track_4d_closed_orbit

def track_alpha_c(ring, max_order=4,h=1e-6) -> list[float]:
    """
    calculate momentum compaction factor based on particle tracking.
    """

    # f(delta) = ΔC(delta)/C0
    def f(delta):
        off_cb = track_4d_closed_orbit(ring,delta,verbose=False)
        rout = symplectic_track(np.concatenate((off_cb['closed_orbit'], (0, delta))), ring, 1)
        return -rout[4] / ring.length

    try:
        cavity_state = ring.RFCavity.enable
        ring.RFCavity.enable = False
        reset_flag = True
    except AttributeError:
        reset_flag = False

    alpha_coeffs = []

    for i in range(max_order):
        order = i + 1  # α_i -> delta^(i+1)

        if order == 1:
            deriv = (f(h) - f(-h)) / (2 * h)
            alpha_coeffs.append(deriv)

        elif order == 2:
            deriv = (f(h) - 2*f(0) + f(-h)) / (h**2)
            alpha_coeffs.append(deriv / 2)

        elif order == 3:
            deriv = (f(2*h) - 2*f(h) + 2*f(-h) - f(-2*h)) / (2 * h**3)
            alpha_coeffs.append(deriv / 6)

        elif order == 4:
            deriv = (f(2*h) - 4*f(h) + 6*f(0) - 4*f(-h) + f(-2*h)) / (h**4)
            alpha_coeffs.append(deriv / 24)

        else:
            print(f"Warnning: max order is 4!!!")
            alpha_coeffs.append(0)
    if reset_flag:
        ring.RFCavity.enable = cavity_state
    return alpha_coeffs


def calculate_eta_from_alpha(alpha_coeffs, gamma_0, max_order=None) -> list[float]:
    """
    calculate phase slip factor according to momentum compaction factor

    Args:
        alpha_coeffs: momentum compaction factors [alpha_0, alpha_1, alpha_2, ...]
        gamma_0:
        max_order:

    Return:
        eta_coeffs:  phase slip factors [η0, η1, η2, ...]
    """

    if max_order is None:
        max_order = len(alpha_coeffs)

    beta_0 = np.sqrt(1 - 1/gamma_0**2)
    beta_0_sq = beta_0**2
    gamma_0_sq = gamma_0**2

    # eta_{-1}=1 and eta_{i<-1}=0
    eta = [0] * (max_order + 2)
    eta[0] = 1  # eta_{-1} = 1

    eta_coeffs = []

    for j in range(max_order):   # j=i-1

        eta_i_minus_1 = eta[j] if j >= 0 else 0
        eta_i_minus_2 = eta[j-1] if j-1 >= 0 else 0
        eta_i_minus_3 = eta[j-2] if j-2 >= 0 else 0
        eta_i_minus_4 = eta[j-3] if j-3 >= 0 else 0

        alpha_i = alpha_coeffs[j] if j < len(alpha_coeffs) else 0

        term1 = alpha_i
        term2 = -eta_i_minus_1 / gamma_0_sq
        term3 = (3 * beta_0_sq * eta_i_minus_2) / (2 * gamma_0_sq)
        term4 = ((1 - 5 * beta_0_sq) * beta_0_sq * eta_i_minus_3) / (2 * gamma_0_sq)
        term5 = (-5 * (3 - 7 * beta_0_sq) * beta_0_sq**2 * eta_i_minus_4) / (8 * gamma_0_sq)

        eta_i = term1 + term2 + term3 + term4 + term5

        eta[j+1] = eta_i
        eta_coeffs.append(eta_i)

    return eta_coeffs
