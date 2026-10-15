import numpy as np
import pytest

import simplestoragering as ssr
from HOA_7BA_lattice import generate_ring

pytest.importorskip('nafflib')


def test_tune_analysis_recovers_fractional_tune():
    n = np.arange(2048)
    mu_x = 2 * np.pi * 0.23
    mu_y = 2 * np.pi * 0.31
    z = np.zeros_like(n, dtype=float)
    trajectory = np.column_stack([
        np.cos(mu_x * n), -np.sin(mu_x * n),
        np.cos(mu_y * n), -np.sin(mu_y * n),
        z, z,
    ])
    nux, nuy = ssr.tune_analysis(trajectory)
    assert nux == pytest.approx(0.23, abs=1e-3)
    assert nuy == pytest.approx(0.31, abs=1e-3)


def test_tune_analysis_zero_signal_is_nan():
    trajectory = np.zeros((100, 6))
    nux, nuy = ssr.tune_analysis(trajectory)
    assert np.isnan(nux)
    assert np.isnan(nuy)


def test_tune_analysis_rejects_bad_shape():
    with pytest.raises(ValueError):
        ssr.tune_analysis(np.zeros((10, 4)))


def test_tune_shift_matches_linear_tune():
    ring = generate_ring()
    ring.linear_optics()
    scan = ssr.tune_shift_with_x(ring, x_list=[1e-4], n_turns=512)
    assert scan.nux[0] == pytest.approx(ring.nux % 1, abs=1e-3)
    assert scan.nuy[0] == pytest.approx(ring.nuy % 1, abs=1e-3)


def test_tune_shift_does_not_touch_input_ring_cavity():
    ring = generate_ring()
    ring.linear_optics()
    assert ring.RFCavity.enable == 1
    ssr.tune_shift_with_delta(ring, delta_list=[0.0], n_turns=256)
    assert ring.RFCavity.enable == 1


def test_tune_shift_particle_lost_is_nan():
    ring = generate_ring()
    ring.linear_optics()
    scan = ssr.tune_shift_with_x(ring, x_list=[0.03], n_turns=512)
    assert np.isnan(scan.nux[0])
    assert np.isnan(scan.nuy[0])


def test_tune_shift_with_delta_shifts_tune():
    ring = generate_ring()
    ring.linear_optics()
    on = ssr.tune_shift_with_delta(ring, delta_list=[0.0], n_turns=512)
    off = ssr.tune_shift_with_delta(ring, delta_list=[0.005], n_turns=512)
    assert on.nux[0] == pytest.approx(ring.nux % 1, abs=1e-3)
    assert np.isfinite(off.nux[0])
    assert off.nux[0] != pytest.approx(on.nux[0], abs=1e-5)
    np.testing.assert_array_equal(off.param, [0.5])


def test_tune_shift_with_amplitude_uses_action():
    ring = generate_ring()
    ring.linear_optics()
    scan = ssr.tune_shift_with_amplitude(ring, J_list=[1e-10], plane='x', n_turns=512)
    assert scan.nux[0] == pytest.approx(ring.nux % 1, abs=1e-3)
    assert np.isfinite(scan.nuy[0])
    assert '$J_x$' in scan.xlabel


def test_tune_shift_replaces_zero_x():
    ring = generate_ring()
    ring.linear_optics()
    scan = ssr.tune_shift_with_x(ring, x_list=[0.0], n_turns=512)
    assert np.isfinite(scan.nux[0])
    assert scan.nux[0] == pytest.approx(ring.nux % 1, abs=1e-3)
    # 1e-6 m default amplitude is reported as 1e-3 mm
    assert scan.param[0] == pytest.approx(1e-3)


def test_tune_shift_with_amplitude_replaces_zero():
    ring = generate_ring()
    ring.linear_optics()
    scan = ssr.tune_shift_with_amplitude(ring, J_list=[0.0], plane='x', n_turns=512)
    assert np.isfinite(scan.nux[0])
    assert scan.nux[0] == pytest.approx(ring.nux % 1, abs=1e-3)


def test_tune_scan_save(tmp_path):
    ring = generate_ring()
    ring.linear_optics()
    scan = ssr.tune_shift_with_x(ring, x_list=[1e-4], n_turns=256)
    filename = tmp_path / 'tune.csv'
    scan.save(filename)
    data = np.loadtxt(filename, delimiter=',')
    assert data.shape == (3,)
    assert data[1] == pytest.approx(scan.nux[0])
    assert data[2] == pytest.approx(scan.nuy[0])
