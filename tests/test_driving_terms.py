"""Tests for the driving-term calculation and the ADTS derived from it."""
import copy

import numpy as np

import simplestoragering as ssr


def _cyclic_rotation(ring, shift):
    """Return a new ring whose element list starts ``shift`` elements later."""
    elements = [ele for ele in ring.elements
                if type(ele).__name__ != 'LineEnd']
    elements = elements[shift:] + elements[:shift]
    return ssr.CSLattice(elements, n_periods=ring.n_periods)


def test_adts_does_not_depend_on_lattice_start(HOA7BA):
    """ADTS are physical observables and cannot depend on where the ring
    element list starts.

    The second-order driving terms receive a periodic-closure contribution
    from the first-order terms.  Adding it twice made the result depend on the
    arbitrary start of the lattice (and changed dQxy by ~20% for lattices with
    n_periods=1 such as SOLEIL).
    """
    ring = copy.deepcopy(HOA7BA)
    ring.linear_optics()
    reference = ring.driving_terms(verbose=False).adts()

    n_elements = len(ring.elements)
    for shift in (n_elements // 3, 2 * n_elements // 3):
        rotated = _cyclic_rotation(ring, shift)
        rotated.linear_optics()
        result = rotated.driving_terms(verbose=False).adts()
        np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-6)
