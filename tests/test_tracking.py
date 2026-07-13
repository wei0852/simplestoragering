import numpy as np

import simplestoragering as ssr


def test_mark_recording_has_one_row_per_pass():
    ring = ssr.CSLattice([ssr.Mark("sample")], n_periods=3)

    ssr.symplectic_track([1, 2, 3, 4, 5, 6], ring, n_turns=4, record=True)

    data = ring.mark["sample"][0].data
    assert data.shape == (12, 6)
    np.testing.assert_array_equal(data, np.tile([1, 2, 3, 4, 5, 6], (12, 1)))


def test_mark_recording_is_reset_between_tracking_calls():
    ring = ssr.CSLattice([ssr.Mark("sample")])

    ssr.symplectic_track([1, 0, 0, 0, 0, 0], ring, n_turns=5, record=True)
    ssr.symplectic_track([2, 0, 0, 0, 0, 0], ring, n_turns=2, record=True)

    data = ring.mark["sample"][0].data
    assert data.shape == (2, 6)
    np.testing.assert_array_equal(data[:, 0], [2, 2])


def test_disabling_recording_clears_previous_mark_data():
    ring = ssr.CSLattice([ssr.Mark("sample")])

    ssr.symplectic_track([0, 0, 0, 0, 0, 0], ring, n_turns=2, record=True)
    ssr.symplectic_track([0, 0, 0, 0, 0, 0], ring, n_turns=2, record=False)

    assert ring.mark["sample"][0].data is None
