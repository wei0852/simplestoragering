"""Horizontal normal-form analysis.

The normal form maps the horizontal coordinates ``(x, px)`` to
action-angle variables with the off-momentum Twiss parameters at the start
of the lattice.  At ``order=2`` the transformation is linear.  Higher
orders add the horizontal resonance driving terms computed by
``CSLattice.driving_terms``.

``compute_normal_form`` calls ``lattice.off_momentum_optics(delta)``,
which updates the Twiss parameters of the lattice and resets some ring
parameters.
"""
from dataclasses import dataclass

import numpy as np


__all__ = [
    'NormalForm',
    'compute_normal_form',
]


@dataclass
class NormalForm:
    """Horizontal normal-form coordinates.

    For the linear transform (``order=2``), ``x`` and ``px`` are the
    Courant-Snyder normalized coordinates.  For higher orders they also
    contain the resonance-driving-term corrections.

    Attributes:
        action: action ``Jx``, including the factor 1/2.
        phase: action-angle phase in radians.
        x: normalized horizontal coordinate.
        px: normalized horizontal angle.
    """
    action: np.ndarray
    phase: np.ndarray
    x: np.ndarray
    px: np.ndarray


def compute_normal_form(x, px, ring, delta=0.0, order=4) -> NormalForm:
    """Apply the horizontal normal form up to the requested order.

    Args:
        x, px: horizontal coordinate and angle.
        ring: CSLattice.  It is left in the off-momentum state.
        delta: momentum deviation.
        order: highest order to include:

            * ``2``: linear (quadratic Hamiltonian) transformation only;
            * ``3``: include third-order resonance driving terms;
            * ``4``: include third- and fourth-order driving terms.

    Returns:
        NormalForm with the action, phase, and normalized coordinates.

    Raises:
        ValueError: if ``order`` is not 2, 3, or 4.
    """
    if order not in (2, 3, 4):
        raise ValueError('order must be 2, 3 or 4')

    ring.off_momentum_optics(delta)
    result = _linear_part(x, px, ring)
    if order == 2:
        return result

    rdt = ring.driving_terms(verbose=False)

    action = result.action
    phase = result.phase
    x_norm = result.x
    px_norm = result.px

    x_mip = x_norm - 1j * px_norm
    x_pip = x_norm + 1j * px_norm
    e_iphi = np.exp(1j * phase)
    sqrt_2J = np.sqrt(2 * action)

    action_nl = action
    phase_nl = phase
    x_nl = x_norm
    px_nl = px_norm
    for f, j, k in _rdt_terms(rdt, order):
        # Jx' = Jx + 1j * f_jk000 * (j-k) * (2Jx)**((j+k)/2) * exp(1j*(j-k)*phi)
        amplitude = sqrt_2J ** (j + k)
        phase_factor = e_iphi ** (j - k)
        action_nl = action_nl + 1j * f * (j - k) * amplitude * phase_factor
        phase_nl = phase_nl - (j + k) * f * sqrt_2J ** (j + k - 2) * phase_factor

        # derivatives of x_mip**j * x_pip**k
        dx_mip = j * x_mip ** (j - 1) * x_pip ** k if j else 0.0
        dx_pip = k * x_mip ** j * x_pip ** (k - 1) if k else 0.0
        x_nl = x_nl + 1j * f * (dx_mip - dx_pip)
        px_nl = px_nl + f * (dx_mip + dx_pip)

    return NormalForm(
        _as_real(action_nl, 'action'),
        _as_real(phase_nl, 'phase'),
        _as_real(x_nl, 'x'),
        _as_real(px_nl, 'px'),
    )


def _linear_part(x, px, ring) -> NormalForm:
    """Linear normal-form transformation using the current lattice optics."""
    x = np.asarray(x, dtype=float) - ring.elements[0].closed_orbit[0]
    px = np.asarray(px, dtype=float) - ring.elements[0].closed_orbit[1]

    betax = ring.elements[0].betax
    alphax = ring.elements[0].alphax

    gammax = (1 + alphax**2) / betax
    action = (gammax * x**2 + 2 * alphax * x * px + betax * px**2) / 2

    sqrt_betax = np.sqrt(betax)
    x_norm = x / sqrt_betax
    px_norm = (alphax * x + betax * px) / sqrt_betax

    phase = np.arctan2(-px_norm, x_norm)
    return NormalForm(action, phase, x_norm, px_norm)


def _rdt_terms(rdt, order):
    """Return ``(f, j, k)`` terms up to the requested order."""
    terms = [
        (rdt.f21000, 2, 1),
        (rdt.f21000.conjugate(), 1, 2),
        (rdt.f30000, 3, 0),
        (rdt.f30000.conjugate(), 0, 3),
    ]
    if order >= 4:
        terms += [
            (rdt.f31000, 3, 1),
            (rdt.f31000.conjugate(), 1, 3),
            (rdt.f40000, 4, 0),
            (rdt.f40000.conjugate(), 0, 4),
        ]
    return terms


def _as_real(value, name):
    """Return the real part, checking that the imaginary part is negligible."""
    imag = float(np.max(np.abs(np.imag(value)), initial=0.0))
    scale = max(1.0, float(np.max(np.abs(value), initial=0.0)))
    if imag > 1e-12 * scale:
        raise ValueError(f'{name} has a non-negligible imaginary part: {imag:.3e}')
    return np.real(value)
