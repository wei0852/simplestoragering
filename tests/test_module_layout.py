"""Regression coverage for public imports after the module split."""
import importlib

import numpy as np
import pytest
import simplestoragering as ssr


@pytest.mark.parametrize("module,names", [
    ("lattice_utils", ["element_index"]),
    ("lattice_io.opa", ["output_opa_file"]),
    ("lattice_io.elegant", ["output_elegant_file"]),
    ("correction", ["adjust_tunes", "chromaticity_correction"]),
    ("longitudinal", ["track_alpha_c", "calculate_eta_from_alpha"]),
    ("find_aperture.dynamic_aperture", ["XDeltaLines", "XDeltaGrid", "XYGrid", "NLine"]),
    ("find_aperture.momentum_aperture", ["LocalMomentumAperture", "DynamicAperturePolyhedron"]),
    ("tune", ["tune_analysis", "tune_shift", "TuneScan"]),
    ("normal_form", ["NormalForm", "compute_normal_form"]),
])
def test_public_imports_resolve_to_implementation(module, names):
    implementation = importlib.import_module("simplestoragering." + module)
    for name in names:
        assert getattr(ssr, name) is getattr(implementation, name)


@pytest.mark.parametrize("grid", [
    lambda: ssr.XYGrid(xlist=[0, 0.002], ylist=[1e-6]),
    lambda: ssr.XDeltaGrid(xlist=[0, 0.002], delta_list=[0]),
])
def test_moved_grids_track_survival_and_loss(grid):
    ssr.set_ref_energy(2200)
    ring = ssr.CSLattice([ssr.Drift("D", length=1, Ax=0.001, Ay=0.001)])
    scan = grid()
    scan.search(ring, n_turns=2)
    assert scan.data.shape == (2, 5)
    np.testing.assert_array_equal(scan.data[:, 2], [0, 1])


def test_element_index_groups_repeated_names():
    ssr.set_ref_energy(2200)
    ring = ssr.CSLattice([ssr.Mark("A"), ssr.Mark("B"), ssr.Mark("A")])
    indices = ssr.element_index(ring)
    assert indices["A"] == [0, 2]
    assert indices["B"] == [1]
