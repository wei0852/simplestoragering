import pytest
import simplestoragering as ssr
from example_7BA_lattice import generate_ring as generate_7BA

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


@pytest.fixture(scope='session')
def HOA7BA():
    """HOA 7BA lattice."""
    ring = generate_7BA() * 14
    for ele in ring.elements:
        if isinstance(ele, ssr.HBend) or isinstance(ele, ssr.Quadrupole) or isinstance(ele, ssr.Sextupole) or isinstance(ele, ssr.Octupole):
            ele.n_slices = int(ele.length / 0.001)
            # ele.n_slices = 10
    ring.linear_optics()
    return ring

