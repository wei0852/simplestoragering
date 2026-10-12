"""Regression tests for quadratic fitting without scikit-learn."""

import numpy as np
import pytest

import simplestoragering as ssr
from simplestoragering.QuadraticFitter import QuadraticForm, QuadraticFunctionFitter


def test_public_exports():
    assert ssr.QuadraticForm is QuadraticForm
    assert ssr.QuadraticFunctionFitter is QuadraticFunctionFitter
    assert 'QuadraticForm' in ssr.__all__
    assert 'QuadraticFunctionFitter' in ssr.__all__


@pytest.mark.parametrize('method', ['uniform', 'lhs'])
@pytest.mark.parametrize('n', [1, 3])
def test_recover_quadratic_coefficients_and_predictions(method, n):
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(n, n))
    A = matrix.T @ matrix + np.eye(n)
    b = rng.normal(size=n)
    c = 2.5
    expected = QuadraticForm(A, b, c)
    fitter = QuadraticFunctionFitter(
        lambda x: expected(x)[0], (np.full(n, -2.), np.full(n, 3.)))
    fitted = fitter.fit(n_samples=100, sample_method=method)
    np.testing.assert_allclose(fitted.A, A, atol=1e-12)
    np.testing.assert_allclose(fitted.b, b, atol=1e-12)
    np.testing.assert_allclose(fitted.c, c, atol=1e-12)
    points = rng.uniform(-2, 3, size=(20, n))
    np.testing.assert_allclose(fitted(points), expected(points), atol=1e-12)
    assert fitter.evaluate_fit(points, expected(points))['max_error'] < 1e-10
    assert fitter.evaluate_fit()['R2'] == pytest.approx(1.)
    value, point = fitted.minimum()
    np.testing.assert_allclose(point, -0.5 * np.linalg.solve(A, b), atol=1e-12)
    np.testing.assert_allclose(value, expected(point)[0], atol=1e-12)


def test_underdetermined_fit_reproduces_sample_values():
    fitter = QuadraticFunctionFitter(
        lambda x: 3. + x[0] - 2. * x[1] + x[0] * x[1],
        (np.full(2, -1.), np.full(2, 1.)))
    fitted = fitter.fit(n_samples=2)
    np.testing.assert_allclose(fitted(fitter.samples_X), fitter.samples_y, atol=1e-12)


def test_missing_bounds_and_unsupported_sampling_method():
    fitter = QuadraticFunctionFitter(lambda x: x[0])
    with pytest.raises(ValueError, match='bounds'):
        fitter.fit()
    fitter.bounds = (np.array([-1.]), np.array([1.]))
    with pytest.raises(ValueError, match='Unknown sampling method'):
        fitter.sample_points(method='grid')
