"""Tune analysis with tracking and NAFF.

The particle is tracked once per turn over the whole ring, so the frequency
returned by :func:`tune_analysis` is the fractional tune of the full ring.
Because the tune is a global property of the ring, a dedicated Mark is added
automatically when no Mark name is given, and its position does not matter.

For symplectic tracking the RF cavity is switched off, so that the energy
stays constant and the measurement is a pure transverse tune. With
radiation=True the cavity is left on (radiation damping), and a warning is
issued because the changing energy shifts the tune.
"""
import warnings
from dataclasses import dataclass

import numpy as np

from .CSLattice import CSLattice
from .components import Mark
from .track import symplectic_track, radiation_track
from .exceptions import ParticleLost


__all__ = [
    'TuneScan',
    'tune_analysis',
    'tune_shift',
    'tune_shift_with_x',
    'tune_shift_with_y',
    'tune_shift_with_delta',
    'tune_shift_with_amplitude',
]


_DEFAULT_PARTICLE = [1e-6, 0.0, 1e-6, 0.0, 0.0, 0.0]
_TUNE_MARK_NAME = '_tune_'


def _naff_tune(trajectory, iq, ip):
    """Fractional tune from one plane of a recorded trajectory."""
    import nafflib
    q = trajectory[:, iq] - np.mean(trajectory[:, iq])
    p = trajectory[:, ip] - np.mean(trajectory[:, ip])
    if np.ptp(q) == 0 and np.ptp(p) == 0:
        # no oscillation in this plane (e.g. zero initial amplitude)
        return np.nan
    try:
        return float(nafflib.tune(q, px=p) % 1)
    except Exception:
        return np.nan


def tune_analysis(trajectory):
    """Fractional tunes (nux, nuy) from a recorded trajectory.

    Args:
        trajectory: np.ndarray of shape (n, 6), the coordinate data recorded at
            a Mark, one row per turn.

    Returns:
        (nux, nuy), the fractional tunes. A plane is np.nan if the frequency
        analysis fails for that plane.

    Raises:
        ImportError: if nafflib is not installed.
        ValueError: if trajectory is not a (n, 6) array.
    """
    try:
        import nafflib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'nafflib is required for tune analysis. '
            "Install it with `pip install nafflib` (python >= 3.10)."
        ) from exc
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 6:
        raise ValueError('trajectory must have shape (n, 6).')
    nux = _naff_tune(trajectory, 0, 1)
    nuy = _naff_tune(trajectory, 2, 3)
    return nux, nuy


@dataclass
class TuneScan:
    """Result of a tune scan.

    Attributes:
        param: np.ndarray, the scanned quantity, in the unit given by xlabel.
        nux, nuy: np.ndarray, fractional tunes. np.nan where the particle is
            lost or the frequency analysis fails.
        xlabel: str, label (with unit) of param.
        n_turns: int, number of turns used for the tracking.
        mark: str, name of the Mark used to record the trajectory.
    """
    param: np.ndarray
    nux: np.ndarray
    nuy: np.ndarray
    xlabel: str
    n_turns: int
    mark: str

    def save(self, filename=None, header=None):
        """Save param, nux, nuy to a csv file."""
        filename = 'TuneScan.csv' if filename is None else filename
        lines = [] if header is None else [header]
        lines.append(f'n_turns: {self.n_turns}, mark: {self.mark}')
        lines.append(f'{self.xlabel}, nux, nuy')
        data = np.column_stack((self.param, self.nux, self.nuy))
        np.savetxt(filename, data, delimiter=',', fmt='%14.8f',
                   comments='#', header='\n'.join(lines))

    def show(self, ax=None, plane='both'):
        """Scatter the tunes against the scanned quantity.

        plane: 'both', 'x' or 'y'.
        """
        import matplotlib.pyplot as plt
        created = ax is None
        if created:
            fig, ax = plt.subplots(1, 1)
        if plane in ('both', 'x'):
            ax.scatter(self.param, self.nux, label='$\\nu_x$')
        if plane in ('both', 'y'):
            ax.scatter(self.param, self.nuy, label='$\\nu_y$')
        ax.set_xlabel(self.xlabel, fontsize=15)
        ax.set_ylabel('tunes', fontsize=15)
        ax.grid('on')
        ax.legend(fontsize=12)
        if created:
            plt.tight_layout()
            plt.show()


def _expand_ring(ring, mark):
    """Expand a ring to a single period and make sure a Mark is available.

    Returns (full_ring, mark_name). The whole ring is expanded with
    `ring * ring.n_periods`, so the Mark is passed once per turn and the
    recorded frequency is directly the fractional tune of the full ring.
    When mark is None a new Mark is added at the beginning of the ring.
    """
    if not isinstance(ring, CSLattice):
        raise TypeError('ring must be a CSLattice.')
    full_ring = ring * ring.n_periods
    if mark is None:
        mark = _TUNE_MARK_NAME
        full_ring = CSLattice([Mark(mark)] + full_ring.elements)
    if mark not in full_ring.mark:
        raise ValueError(f'Mark {mark!r} not found in the ring.')
    return full_ring, mark


def _tune_scan(ring, param_list, initial_fn, *, xlabel='', scale=1.0,
               n_turns=517, mark=None, radiation=False, verbose=False):
    """Track initial_fn(value) for every value and return a TuneScan.

    scale converts the scanned value to the display unit of xlabel, and is
    applied to both param and the verbose output.
    """
    if radiation and ring.RFCavity is None:
        raise ValueError('radiation tracking requires a 6D ring with an RFCavity.')
    if radiation:
        warnings.warn(
            'radiation tracking changes the energy, so the tune may shift '
            'compared with symplectic tracking.'
        )
    full_ring, mark_name = _expand_ring(ring, mark)
    if not radiation and full_ring.RFCavity is not None:
        # The tune is a transverse property: turn off the RF cavity so that
        # symplectic tracking keeps the energy constant (4D-like motion).
        full_ring.RFCavity.enable = 0
    param_list = np.asarray(param_list, dtype=float)
    shown = param_list * scale
    nux_list = np.full(param_list.shape, np.nan)
    nuy_list = np.full(param_list.shape, np.nan)
    for i, value in enumerate(param_list):
        particle = np.asarray(initial_fn(value), dtype=float)
        try:
            if radiation:
                radiation_track(particle, full_ring, n_turns, record=True)
            else:
                symplectic_track(particle, full_ring, n_turns, record=True)
            nux, nuy = tune_analysis(full_ring.mark[mark_name][0].data)
        except ParticleLost:
            nux, nuy = np.nan, np.nan
        nux_list[i] = nux
        nuy_list[i] = nuy
        if verbose:
            print(f'{xlabel} = {shown[i]}, nux = {nux}, nuy = {nuy}')
    return TuneScan(shown, nux_list, nuy_list, xlabel, n_turns, mark_name)


def _avoid_zero(values, small):
    """Replace exact zeros so the betatron oscillation is not degenerate."""
    values = np.asarray(values, dtype=float)
    return np.where(values == 0, small, values)


def tune_shift(ring, param_list, param_idx, *, xlabel='', initial=None, **kwargs):
    """Tune shift along a parameter list.

    Args:
        ring: CSLattice.
        param_list: values of the scanned quantity.
        param_idx: index in the 6-dim particle, 0..5 as
            [x, px, y, py, z, delta].
        xlabel: label (with unit) of the scanned quantity.
        initial: base particle of length 6, default [1e-6, 0, 1e-6, 0, 0, 0].
        kwargs: passed to _tune_scan (n_turns, mark, radiation, verbose).

    Returns:
        TuneScan.
    """
    if not 0 <= param_idx <= 5:
        raise ValueError('param_idx must be in 0..5.')
    base = np.array(_DEFAULT_PARTICLE if initial is None else initial, dtype=float)
    if base.shape != (6,):
        raise ValueError('initial must have 6 elements.')

    def initial_fn(value):
        particle = base.copy()
        particle[param_idx] = value
        return particle

    return _tune_scan(ring, param_list, initial_fn, xlabel=xlabel, **kwargs)


def tune_shift_with_x(ring, x_list, **kwargs):
    """Tune shift with the horizontal initial position. Param in mm.

    Zeros in x_list are replaced by the default small amplitude.
    """
    x_list = _avoid_zero(x_list, _DEFAULT_PARTICLE[0])
    return tune_shift(ring, x_list, 0, xlabel='x [mm]', scale=1000, **kwargs)


def tune_shift_with_y(ring, y_list, **kwargs):
    """Tune shift with the vertical initial position. Param in mm.

    Zeros in y_list are replaced by the default small amplitude.
    """
    y_list = _avoid_zero(y_list, _DEFAULT_PARTICLE[2])
    return tune_shift(ring, y_list, 2, xlabel='y [mm]', scale=1000, **kwargs)


def tune_shift_with_delta(ring, delta_list, **kwargs):
    """Tune shift with the momentum deviation (initial energy). Param in %."""
    return tune_shift(ring, delta_list, 5, xlabel='$\\delta$ [%]', scale=100, **kwargs)


def tune_shift_with_amplitude(ring, J_list, plane='x', **kwargs):
    """Tune shift with the betatron action J. Param in m.

    The initial coordinate is built with the Twiss parameters at s=0:
    q = sqrt(2 J beta), q' = -alpha sqrt(2 J / beta), on-momentum.

    Args:
        J_list: betatron action in m. Zeros are replaced by the action of the
            default small amplitude.
        plane: 'x' or 'y'.
    """
    if plane not in ('x', 'y'):
        raise ValueError("plane must be 'x' or 'y'.")
    ring.linear_optics()
    ele0 = ring.elements[0]
    if plane == 'x':
        beta, alpha = ele0.betax, ele0.alphax
    else:
        beta, alpha = ele0.betay, ele0.alphay
    J_list = _avoid_zero(J_list, _DEFAULT_PARTICLE[0] ** 2 / (2 * beta))

    def initial_fn(J):
        particle = np.array(_DEFAULT_PARTICLE)
        amplitude = np.sqrt(2 * J * beta)
        angle = -alpha * np.sqrt(2 * J / beta)
        if plane == 'x':
            particle[0] = amplitude
            particle[1] = angle
        else:
            particle[2] = amplitude
            particle[3] = angle
        return particle

    return _tune_scan(ring, J_list, initial_fn, xlabel=f'$J_{plane}$ [m]', **kwargs)
