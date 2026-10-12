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


@pytest.fixture
def quadratic_objective_ring(monkeypatch):
    import simplestoragering.correction as correction

    elements = [SimpleNamespace(name=name, type='Sextupole', k2=k2)
                for name, k2 in [('SDa', -10.), ('SDa', -20.), ('SDb', -30.),
                                 ('SF', 40.), ('C1', -50.), ('C2', 60.),
                                 ('X1', 0.), ('X2', 100.)]]
    ring = SimpleNamespace(elements=elements, linear_optics=lambda: None, calls=[],
                           xi_x=5., xi_y=3.)

    def correct(lattice, names, target, use_tracked_ksi):
        assert names == ['C1', 'C2']
        np.testing.assert_array_equal(target, [5, 3])
        assert not use_tracked_ksi
        assert [elements[4].k2, elements[5].k2] == [-50., 60.]
        elements[4].k2 = -100.
        elements[5].k2 = 100.
        lattice.xi_x, lattice.xi_y = 123., 456.

    monkeypatch.setattr(correction, 'chromaticity_correction', correct)
    return ring


@pytest.mark.parametrize('update_k2', [False, True])
@pytest.mark.parametrize('corrected_chromaticity', [None, [5, 3]])
@pytest.mark.parametrize('n,n_samples', [(2, None), (4, None), (4, 40)])
def test_theoretical_minimum_f3rms_absolute_strengths_and_restore(
        quadratic_objective_ring, monkeypatch, n, n_samples,
        corrected_chromaticity, update_k2):
    from simplestoragering.QuadraticFitter import QuadraticFunctionFitter

    ring = quadratic_objective_ring
    original = [ele.k2 for ele in ring.elements]
    methods = []
    sample_points = QuadraticFunctionFitter.sample_points

    def sample(self, n_samples, method):
        methods.append(method)
        return sample_points(self, n_samples, method)

    monkeypatch.setattr(QuadraticFunctionFitter, 'sample_points', sample)

    def driving_terms(verbose):
        assert verbose is False
        k2 = [ele.k2 for ele in ring.elements]
        ring.calls.append(k2)
        np.testing.assert_allclose(k2[:3], k2[0])
        assert -30. <= k2[0] <= -10.
        assert 20. <= k2[3] <= 60.
        q = 2. + (k2[0] + 22.)**2 + 3. * (k2[3] - 32.)**2
        if n == 4:
            assert -1. <= k2[6] <= 1.
            assert 50. <= k2[7] <= 150.
            q += 2. * (k2[6] - 0.25)**2 + (k2[7] - 90.)**2
        keys = ['f2100', 'f3000', 'f1011', 'f1002', 'f1020',
                'f3100', 'f4000', 'f2011', 'f1120', 'f2002', 'f2020', 'f0031', 'f0040']
        natural = {key: np.zeros(1) for key in keys}
        natural['f2100'] = np.array([np.sqrt(q)])
        return SimpleNamespace(s=np.array([1.]), natural_fluctuation=lambda: natural)

    ring.driving_terms = driving_terms
    names = [['SDa', 'SDb'], 'SF'] + (['X1', 'X2'] if n == 4 else [])
    value, strengths = ssr.objectives.theoretical_minimum_f3rms(
        ring, variable_sextupole_name_list=names,
        chromaticity_sextupole_name_list=['C1', 'C2'],
        corrected_chromaticity=corrected_chromaticity, n_samples=n_samples,
        use_tracked_ksi=False, update_k2=update_k2)
    assert len(ring.calls) == (n_samples or (n + 1) * (n + 2) // 2 + 10)
    assert methods == ['lhs' if n == 4 else 'uniform']
    assert value == pytest.approx(2., abs=1e-7)
    expected = [-22., 32.] + ([0.25, 90.] if n == 4 else [])
    np.testing.assert_allclose(strengths, expected, atol=1e-7)
    if update_k2:
        applied = [expected[0]] * 3 + [expected[1], -100., 100.]
        applied += expected[2:] if n == 4 else original[6:]
        np.testing.assert_allclose([ele.k2 for ele in ring.elements], applied, atol=1e-7)
    else:
        assert [ele.k2 for ele in ring.elements] == original
    assert ssr.theoretical_minimum_f3rms is ssr.objectives.theoretical_minimum_f3rms


def test_theoretical_minimum_restores_strengths_on_sampling_failure(quadratic_objective_ring):
    ring = quadratic_objective_ring
    original = [ele.k2 for ele in ring.elements]

    def fail(verbose):
        raise RuntimeError('RDT calculation failed')

    ring.driving_terms = fail
    with pytest.raises(RuntimeError, match='RDT calculation failed'):
        ssr.theoretical_minimum_f3rms(ring, ['SDa'], ['C1', 'C2'], use_tracked_ksi=False)
    assert [ele.k2 for ele in ring.elements] == original


@pytest.mark.parametrize('names,correction_names,kwargs,message', [
    ([], ['C1', 'C2'], {}, 'nonempty'),
    (['SDa', 'SDa'], ['C1', 'C2'], {}, 'distinct'),
    (['SDa'], ['SDa', 'C2'], {}, 'overlap'),
    (['missing'], ['C1', 'C2'], {}, 'not found'),
    (['SDa'], ['C1', 'C2'], {'n_samples': 2}, 'n_samples'),
    (['SDa'], ['C1', 'C2'], {'corrected_chromaticity': [5]}, 'corrected_chromaticity'),
])
def test_theoretical_minimum_validates_inputs(quadratic_objective_ring, names,
                                      correction_names, kwargs, message):
    ring = quadratic_objective_ring
    original = [ele.k2 for ele in ring.elements]
    with pytest.raises(ValueError, match=message):
        ssr.theoretical_minimum_f3rms(ring, names, correction_names, **kwargs)
    assert [ele.k2 for ele in ring.elements] == original


def test_theoretical_minimum_rejects_quadratic_without_minimum(quadratic_objective_ring,
                                                      monkeypatch):
    ring = quadratic_objective_ring
    original = [ele.k2 for ele in ring.elements]
    ring.driving_terms = lambda verbose: None
    monkeypatch.setattr(ssr.objectives, 'ring_averaged_rdt_metrics',
                        lambda rdt, w: SimpleNamespace(
                            third_order=np.sqrt(10 - (ring.elements[0].k2 / -10.)**2)))
    with pytest.raises(ValueError, match='no unique minimum'):
        ssr.theoretical_minimum_f3rms(ring, ['SDa'], ['C1', 'C2'],
                             n_samples=8, use_tracked_ksi=False)
    assert [ele.k2 for ele in ring.elements] == original


@pytest.mark.parametrize('update_k2', [False, True])
def test_theoretical_minimum_restores_on_failure_to_apply(quadratic_objective_ring,
                                                         monkeypatch, update_k2):
    import simplestoragering.correction as correction

    ring = quadratic_objective_ring
    original = [ele.k2 for ele in ring.elements]
    calls = []

    def correct(lattice, names, target, use_tracked_ksi):
        calls.append(1)
        lattice.elements[4].k2 = -100.
        if len(calls) == 9:
            raise RuntimeError('Final correction failed')

    ring.driving_terms = lambda verbose: None
    monkeypatch.setattr(correction, 'chromaticity_correction', correct)
    monkeypatch.setattr(ssr.objectives, 'ring_averaged_rdt_metrics',
                        lambda rdt, w: SimpleNamespace(
                            third_order=np.sqrt(2. + (ring.elements[0].k2 + 14.)**2)))
    if update_k2:
        with pytest.raises(RuntimeError, match='Final correction failed'):
            ssr.theoretical_minimum_f3rms(ring, ['SDa'], ['C1', 'C2'],
                                         n_samples=8, update_k2=True)
        assert len(calls) == 9
    else:
        value, strengths = ssr.theoretical_minimum_f3rms(
            ring, ['SDa'], ['C1', 'C2'], n_samples=8)
        assert value == pytest.approx(2.)
        np.testing.assert_allclose(strengths, [-14.])
        assert len(calls) == 8
    assert [ele.k2 for ele in ring.elements] == original
