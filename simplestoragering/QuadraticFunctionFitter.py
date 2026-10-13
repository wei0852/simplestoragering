"""Fit scalar functions with quadratic forms using NumPy least squares."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class QuadraticForm:
    """Represent a quadratic function as x.T @ A @ x + b @ x + c."""

    A: np.ndarray  # Symmetric quadratic coefficient matrix, shape (n, n).
    b: np.ndarray  # Linear coefficients, shape (n,).
    c: float       # Constant term.

    def __post_init__(self):
        # Ensure that the quadratic coefficient matrix is symmetric.
        self.A = (self.A + self.A.T) / 2

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate one point or an array of points, returning a 1D array."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.sum(X @ self.A * X, axis=1) + X @ self.b + self.c

    def minimum(self):
        """Return the minimum value and point for a positive definite A."""
        x_min = -0.5 * np.linalg.solve(self.A, self.b)
        f_min = self.evaluate(x_min)
        return f_min[0], x_min

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.evaluate(X)

    def __str__(self):
        text = "f(X) = X^T A X + b X + c\n"
        text += "A = \n"
        text += f"{self.A}\n"
        text += "b = \n"
        text += f"{self.b}\n"
        text += "c = \n"
        text += f"{self.c}\n"
        return text


class QuadraticFunctionFitter:
    """Fit a scalar black-box function with a quadratic polynomial."""

    def __init__(self, f: Callable[[np.ndarray], float],
                 bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Args:
            f: Function accepting an n-dimensional vector and returning a scalar.
            bounds: Optional sampling bounds (lb, ub), each an n-dimensional vector.
        """
        self.f = f
        self.bounds = bounds
        self.quadratic_form = None
        self.n = None  # Number of dimensions.
        self.samples_X = None
        self.samples_y = None

    def sample_points(self, n_samples: int = 100,
                      method: str = 'uniform') -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample the function within the specified bounds.

        Args:
            n_samples: Number of sample points.
            method: Sampling method: 'uniform' or 'lhs' (Latin hypercube).

        Returns:
            X: Sample points with shape (n_samples, n).
            y: Function values with shape (n_samples,).
        """
        if self.bounds is None:
            raise ValueError("Sampling bounds must be specified")

        lb, ub = self.bounds
        self.n = len(lb)

        if method == 'uniform':
            X = np.random.uniform(lb, ub, size=(n_samples, self.n))
        elif method == 'lhs':
            # Latin hypercube sampling provides more even coverage.
            from scipy.stats import qmc

            sampler = qmc.LatinHypercube(d=self.n)
            sample = sampler.random(n_samples)
            X = qmc.scale(sample, lb, ub)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        y = np.array([self.f(x) for x in X])
        self.samples_X = X
        self.samples_y = y
        return X, y

    def fit(self, n_samples: int = 100,
            sample_method: str = 'uniform') -> QuadraticForm:
        """
        Sample the function and fit a quadratic polynomial by least squares.

        Args:
            n_samples: Number of sample points. At least (n + 1) * (n + 2) / 2
                independent samples are needed to determine all coefficients.
            sample_method: Sampling method passed to sample_points.

        Returns:
            The fitted quadratic form. Underdetermined or rank-deficient systems
            yield the minimum-norm least-squares solution.
        """
        X, y = self.sample_points(n_samples, sample_method)
        n = X.shape[1]

        # Order features as: 1, x_0, ..., x_(n-1), x_0**2, x_0*x_1, ... .
        quadratic_terms = [X[:, i] * X[:, j]
                           for i in range(n) for j in range(i, n)]
        X_poly = np.column_stack([np.ones(X.shape[0]), X, *quadratic_terms])
        coeff, _, _, _ = np.linalg.lstsq(X_poly, y, rcond=None)

        c = coeff[0]
        b = coeff[1:1 + n]
        A = np.zeros((n, n))
        idx = 1 + n  # Skip the constant and linear coefficients.
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    A[i, i] = coeff[idx]
                else:
                    # Each off-diagonal coefficient contributes twice to x.T @ A @ x.
                    A[i, j] = coeff[idx] / 2
                    A[j, i] = coeff[idx] / 2
                idx += 1

        self.quadratic_form = QuadraticForm(A, b, c)
        return self.quadratic_form

    def evaluate_fit(self, X_test: np.ndarray = None,
                     y_test: np.ndarray = None) -> dict:
        """
        Evaluate the fitted model using test data or the stored samples.

        Returns:
            A dictionary containing MSE, RMSE, MAE, R2, and max_error.
        """
        if self.quadratic_form is None:
            raise ValueError("Fit the model before evaluating it")

        if X_test is None or y_test is None:
            if self.samples_X is None:
                raise ValueError("Test data are required")
            X_test, y_test = self.samples_X, self.samples_y

        y_pred = self.quadratic_form(X_test)
        mse = np.mean((y_pred - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y_test))
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'max_error': np.max(np.abs(y_pred - y_test))
        }
