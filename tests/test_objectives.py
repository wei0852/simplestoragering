"""Check length weighting and stable/legacy RDT result contracts."""
from types import SimpleNamespace

import numpy as np
import pytest
import simplestoragering as ssr
from simplestoragering.objectives import quantify_rdt_fluctuation, ring_averaged_rdt


@pytest.mark.parametrize("weight", [0, 0.01, 0.5])
def test_rdt_metrics_length_weighting_and_legacy_behavior(weight):
    third_keys = ['f2100', 'f3000', 'f1011', 'f1002', 'f1020']
    fourth_keys = ['f3100', 'f4000', 'f2011', 'f1120', 'f2002', 'f2020', 'f0031', 'f0040']
    natural = {key: np.zeros(2) for key in third_keys + fourth_keys}
    natural['f2100'] = np.array([2., 4.])
    natural['f3000'] = np.array([1., 1.])
    natural['f4000'] = np.array([3., 5.])
    rdt = SimpleNamespace(s=np.array([1., 4.]), natural_fluctuation=lambda: natural)
    third = np.sqrt((4 + 3 * 16) / 4 + 1)
    fourth = np.sqrt((9 + 3 * 25) / 4)
    result = ssr.ring_averaged_rdt_metrics(rdt, w=weight)
    assert isinstance(result, ssr.RDTMetrics)
    assert result.third_order == pytest.approx(third)
    assert result.fourth_order == pytest.approx(fourth)
    assert result.weighted == pytest.approx(third + weight * fourth)
    expected = (third, fourth) if weight == 0 else third + weight * fourth
    assert ring_averaged_rdt(rdt, weight) == pytest.approx(expected)
    assert quantify_rdt_fluctuation(rdt, weight) == pytest.approx(expected)


def test_declared_public_exports_are_available():
    assert len(ssr.__all__) == len(set(ssr.__all__))
    namespace = {}
    exec("from simplestoragering import *", namespace)
    for name in ssr.__all__:
        assert namespace[name] is getattr(ssr, name)
