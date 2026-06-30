__all__ = ['TouscheckLifetime']

import numpy as np
from scipy.constants import physical_constants, speed_of_light
from scipy.integrate import quad
from scipy.special import iv  # Modified Bessel function of the first kind


def TouscheckLifetime(E0, betax, betay, etax, etay, alphax, alphay, etaxp, etayp, dpm, dpp, ele_length, ring_length, Ib, emitx, emity, sigma_e, sigs):

    e0 = physical_constants['elementary charge'][0]  # 1.60217646e-19 Coulomb
    r0 = physical_constants['classical electron radius'][0]  # 2.817940327e-15 m
    spl = speed_of_light  # 299792458 speed of light in vacuum
    assert len(dpm) == len(dpp)
    dppinput = np.vstack((dpm, dpp)).T
    Tlcol = np.zeros(2)
    sigma_e = sigma_e

    Nb = Ib / (spl / ring_length) / e0  # Number of particles per bunch
    mass_el_ev = physical_constants['electron mass energy equivalent in MeV'][0]
    relgamma = E0 / mass_el_ev
    relbeta = np.sqrt(1 - 1. / relgamma**2)
    length = ele_length

    sigxb = np.sqrt(emitx * betax)
    sigyb = np.sqrt(emity * betay)

    sigx = np.sqrt(emitx * betax + sigma_e ** 2 * etax ** 2)
    sigy = np.sqrt(emity * betay + sigma_e ** 2 * etay ** 2)

    Dtx = etax * alphax + etaxp * betax
    Dty = etay * alphay + etayp * betay

    sigp2 = sigma_e ** 2
    Dx2 = etax ** 2
    Dy2 = etay ** 2
    Dtx2 = Dtx**2
    Dty2 = Dty**2
    sigxb2 = sigxb**2
    sigyb2 = sigyb**2

    sighinv2 = 1. / sigp2 + (Dx2 + Dtx2) / sigxb2 + (Dy2 + Dty2) / sigyb2
    sigh = np.sqrt(1. / sighinv2)

    B1 = 1. / (2. * relbeta**2 * relgamma**2) * ((betax ** 2 / sigxb2) * (1 - (sigh ** 2 * Dtx2 / sigxb2)) + (betay ** 2 / sigyb2) * (1 - (sigh ** 2 * Dty2 / sigyb2)))
    B2sq = 1. / (4. * relbeta**4 * relgamma**4) * ((betax ** 2 / sigxb2) * (1 - (sigh ** 2 * Dtx2 / sigxb2)) - (betay ** 2 / sigyb2) * (1 - (sigh ** 2 * Dty2 / sigyb2))) ** 2 + (sigh ** 4 * betax ** 2 * betay ** 2 * Dtx2 * Dty2) / (relbeta ** 4 * relgamma ** 4 * sigxb2 * sigyb2)
    B2 = np.sqrt(B2sq)
    contributionsTL = np.zeros(dppinput.shape)
    assert dppinput.shape[0] == len(betax), f'{len(dppinput)}, {len(betax)}'

    for dppcolnum in range(np.size(dppinput, 1)):
        dpp = dppinput[:, dppcolnum]
        um = relbeta**2 * dpp**2
        val = np.zeros(B1.shape)
        km = np.arctan(np.sqrt(um))
        FpiWfact = np.sqrt(np.pi * (B1**2 - B2**2)) * um

        for ii in range(len(betax)):
            val[ii] = quad(lambda k: TLT_IntPiw_k(k, km[ii], B1[ii], B2[ii]), km[ii], np.pi/2)[0]

        frontfact = (r0**2 * spl * Nb) / (8 * np.pi * relgamma ** 2 * sigs * np.sqrt((sigx**2) * (sigy**2) - sigma_e ** 4 * etax ** 2 * etay ** 2) * um) * 2 * FpiWfact

        contributionsTL[:, dppcolnum] = frontfact * val
        Tlcol[dppcolnum] = 1 / (1 / np.sum(length) * np.sum(contributionsTL[:, dppcolnum] * length))
    Tl = len(Tlcol) / np.sum(1. / Tlcol)
    return Tl, contributionsTL


def TLT_IntPiw_k(k, km, B1, B2):
    """
    Integral in Piwinski Formula for the Lifetime with u = tan^2(k)
    """
    t = np.tan(k) ** 2
    tm = np.tan(km) ** 2

    # In case the Bessel function has too large value (more than 10^251), it
    # is substituted by its exponential approximation: I_0(x) ~ exp(x)/sqrt(2*pi*x)

    if B2 * t < 500:
        I = ((2 * t + 1) ** 2 * ((t / tm) / (1 + t) - 1) / t
             + t
             - np.sqrt(t * tm * (1 + t))
             - (2 + 1 / (2 * t)) * np.log((t / tm) / (1 + t))) * np.exp(-B1 * t) * iv(0, B2 * t) * np.sqrt(1 + t)
    else:
        I = ((2 * t + 1) ** 2 * ((t / tm) / (1 + t) - 1) / t
             + t
             - np.sqrt(t * tm * (1 + t))
             - (2 + 1 / (2 * t)) * np.log((t / tm) / (1 + t))) * np.exp(B2 * t - B1 * t) / np.sqrt(
            2 * np.pi * B2 * t) * np.sqrt(1 + t)

    return I
