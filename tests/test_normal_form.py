"""Tests for the horizontal normal-form analysis."""
import numpy as np
import pytest

import simplestoragering as ssr
from simplestoragering.normal_form import compute_normal_form


def _make_ring(with_sextupole=True):
    ssr.set_ref_energy(2200)
    elements = [
        ssr.Drift('D1', 0.5),
        ssr.Quadrupole('Q1', 0.2, k1=1.5),
        ssr.Drift('D2', 0.5),
        ssr.Quadrupole('Q2', 0.2, k1=-1.5),
    ]
    if with_sextupole:
        elements.append(ssr.Sextupole('S1', 0.1, k2=20.0))
    return ssr.CSLattice(elements)


def test_order_two_matches_twiss():
    ring = _make_ring()
    x = np.array([1e-4])
    px = np.array([2e-5])
    result = compute_normal_form(x, px, ring, delta=0.0, order=2)

    betax = ring.elements[0].betax
    alphax = ring.elements[0].alphax
    x0, px0 = ring.elements[0].closed_orbit[:2]
    gammax = (1 + alphax**2) / betax
    dx = x - x0
    dpx = px - px0

    expected_action = (gammax * dx**2 + 2 * alphax * dx * dpx + betax * dpx**2) / 2
    np.testing.assert_allclose(result.action, expected_action)
    np.testing.assert_allclose(result.x, dx / np.sqrt(betax))
    np.testing.assert_allclose(result.px, (alphax * dx + betax * dpx) / np.sqrt(betax))


def test_normal_form_accepts_arrays_and_validates_order():
    ring = _make_ring()
    x = np.array([1e-5, 2e-5, 3e-5])
    px = np.zeros_like(x)
    result = compute_normal_form(x, px, ring, order=4)

    assert result.action.shape == x.shape
    for value in (result.action, result.phase, result.x, result.px):
        assert np.isrealobj(value)

    for order in (1, 5):
        with pytest.raises(ValueError, match='order'):
            compute_normal_form(x, px, ring, order=order)


def test_normal_form_at_off_momentum_closed_orbit():
    ring = _make_ring()
    delta = 1e-3
    ring.off_momentum_optics(delta)
    x0, px0 = ring.elements[0].closed_orbit[:2]

    result = compute_normal_form(x0, px0, ring, delta=delta, order=4)
    np.testing.assert_allclose(result.action, 0.0, atol=1e-20)
    np.testing.assert_allclose(result.phase, 0.0, atol=1e-20)


def test_action_fluctuation_decreases_with_order():
    ssr.set_ref_energy(2200)
    ring = ssr.CSLattice([
        ssr.Mark('OBS'),
        ssr.Drift('D1', 0.5),
        ssr.Quadrupole('Q1', 0.2, k1=5.0),
        ssr.Drift('D2', 0.5),
        ssr.Quadrupole('Q2', 0.2, k1=-5.0),
        ssr.Sextupole('S1', 0.1, k2=10.0),
        ssr.Octupole('O1', 0.1, k3=100.0),
    ])
    ssr.symplectic_track([1e-3, 0, 0, 0, 0, 0], ring, n_turns=32, record=True)
    data = ring.mark['OBS'][0].data

    fluctuation = {}
    for order in (2, 3, 4):
        # A higher-order normal form should make the action more invariant.
        action = compute_normal_form(data[:, 0], data[:, 1], ring, order=order).action
        fluctuation[order] = np.std(action) / np.mean(action)

    assert fluctuation[2] > fluctuation[3] > fluctuation[4]


def test_higher_orders_without_sextupoles_equal_linear():
    ring = _make_ring(with_sextupole=False)
    x = np.array([1e-5, 2e-5])
    px = np.zeros_like(x)

    linear = compute_normal_form(x, px, ring, delta=0.0, order=2)
    nonlinear = compute_normal_form(x, px, ring, delta=0.0, order=4)

    np.testing.assert_allclose(nonlinear.action, linear.action)
    np.testing.assert_allclose(nonlinear.phase, linear.phase)
    np.testing.assert_allclose(nonlinear.x, linear.x)
    np.testing.assert_allclose(nonlinear.px, linear.px)
