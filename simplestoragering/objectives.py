"""This file provides several functions for conveniently calculating objectives to optimize the lattice.
HOWEVER, it's important to note that some of these functions are based on experiential knowledge rather than strict validation.
Users should exercise caution when utilizing these objectives.
"""
from dataclasses import dataclass
from typing import Union
import numpy as np
from .DrivingTerms import DrivingTerms


def quantify_rdt_fluctuation(rdt: DrivingTerms, w: float = 0.01) -> Union[float, tuple[float, float]]:
    """Use the rms value along the ring to quantify natural RDT fluctuation.

    w is the weight for the 4th-order RDTs, if w == 0, return a tuple (f3rms, f4rms),
    else return f3rms + f4rms * w.
    The coefficient w can be estimated by action of particle at the anticipated DA,
    w ~ (2J_x)^0.5 = x / betax^0.5,  usually 0.01 is OK."""
    natural = rdt.natural_fluctuation()
    l_diff = np.diff(rdt.s, prepend=0)
    f3rms = 0
    f4rms = 0
    for k in ['f2100', 'f3000', 'f1011', 'f1002', 'f1020']:
        f3rms += np.average(natural[k] ** 2, weights=l_diff)  # ** 0.5 ** 2
    f3rms = f3rms ** 0.5
    for k in ['f3100', 'f4000', 'f2011', 'f1120', 'f2002', 'f2020', 'f0031', 'f0040']:
        f4rms += np.average(natural[k] ** 2, weights=l_diff)  # ** 0.5 ** 2
    f4rms = f4rms ** 0.5
    if w == 0:
        return f3rms, f4rms
    else:
        return f3rms + f4rms * w


def ring_averaged_rdt(rdt: DrivingTerms, w: float = 0.01) -> Union[float, tuple[float, float]]:
    """Return (third, fourth) at w=0, otherwise third + w * fourth.

    For a fixed result structure, use ring_averaged_rdt_metrics().
    """
    return quantify_rdt_fluctuation(rdt, w)


def compute_h3ring(rdt: DrivingTerms) -> float:
    h3ring = 0
    for k in ['h2100', 'h3000', 'h1011', 'h1002', 'h1020']:
        h3ring += abs(rdt[k]) ** 2
    h3ring = h3ring ** 0.5
    return h3ring


def compute_h4ring(rdt: DrivingTerms) -> float:
    h4ring = 0
    for k in ['h3100', 'h4000', 'h2011', 'h1120', 'h2002', 'h2020', 'h0031', 'h0040']:
        h4ring += abs(rdt[k]) ** 2
    h4ring = h4ring ** 0.5
    return h4ring


@dataclass(frozen=True)
class RDTMetrics:
    """Length-weighted natural RDT fluctuation metrics, grouped by order.

    weighted = third_order + w * fourth_order for the requested weight w.
    These are RMS aggregates, not complex averages of the RDTs.
    """
    third_order: float
    fourth_order: float
    weighted: float


def ring_averaged_rdt_metrics(rdt: DrivingTerms, w: float = 0.01) -> RDTMetrics:
    """Return all three metrics for every weight, including zero.

    Uses the same longitudinal weights and RDT groups as
    quantify_rdt_fluctuation(). At w=0, weighted equals third_order;
    fourth_order is still calculated and included.
    """
    third, fourth = quantify_rdt_fluctuation(rdt, w=0)
    return RDTMetrics(float(third), float(fourth), float(third + w * fourth))


def theoretical_minimum_f3rms(ring, variable_sextupole_name_list,
                       chromaticity_sextupole_name_list,
                       corrected_chromaticity=None, n_samples=None,
                       use_tracked_ksi=True, update_k2=False):
    """Estimate the theoretical minimum of f3rms squared by quadratic fitting.

    Args:
        ring: Ring whose on-momentum optics are used for the RDT calculation.
        variable_sextupole_name_list: Variable sextupole names. Each entry is either a name or a
            sequence of names sharing one variable. All elements with a given
            name are updated, including repeated occurrences.
        chromaticity_sextupole_name_list: Names of sextupole families used for chromaticity
            correction. These must be distinct from the variable sextupoles.
        corrected_chromaticity: Horizontal and vertical chromaticities after
            correction, for the full ring. If None, use the input ring's
            current linear chromaticities, refreshed before sampling.
        n_samples: Sample count. Defaults to (n + 1) * (n + 2) // 2 + 10.
        use_tracked_ksi: Use tracking in chromaticity correction.
        update_k2: If True, apply the minimizing variable strengths and correct
            chromaticity to the same target. Defaults to False.

    Returns:
        (minimum_f3rms_squared, k2_values) from the fitted quadratic form, with
        one absolute sextupole strength per variable group in input order.
        f3rms uses the length weighting in ring_averaged_rdt_metrics(). The
        returned value is a model prediction, not a new evaluation of the ring.

    Raises:
        ValueError: Invalid inputs or a fitted matrix without a unique minimum.

    Sampling uses absolute k2 values centered on each group's mean initial k2.
    The half-width is 50% of the mean absolute initial strength, or 1 m^-3,
    whichever is larger. Every element in a group receives the same sampled k2.
    Sampling is uniform for fewer than four variables and Latin hypercube for
    four or more. The returned analytic minimum is not constrained by the
    sampling range. Both fitting and minimization use absolute k2 coordinates.

    Unless update_k2 is True and the operation succeeds, the input ring's
    sextupole strengths are restored and linear optics are refreshed on exit.
    Failures during sampling or application restore the original strengths.
    """
    from .QuadraticFitter import QuadraticFunctionFitter
    from .correction import chromaticity_correction
    from .lattice_utils import element_index

    groups = [[entry] if isinstance(entry, str) else list(entry)
              for entry in variable_sextupole_name_list]
    n = len(groups)
    if not n or any(not group for group in groups):
        raise ValueError('variable_sextupole_name_list must contain nonempty variable groups')
    names = [name for group in groups for name in group]
    if any(not isinstance(name, str) for name in names):
        raise ValueError('Sextupole names must be strings')
    if len(set(names)) != len(names):
        raise ValueError('Variable sextupole names must be distinct')
    correction_names = list(chromaticity_sextupole_name_list)
    if (not correction_names or
            any(not isinstance(name, str) for name in correction_names) or
            len(set(correction_names)) != len(correction_names)):
        raise ValueError('chromaticity_sextupole_name_list must contain distinct names')
    if set(names).intersection(correction_names):
        raise ValueError('Variable and correction sextupole names must not overlap')

    if corrected_chromaticity is not None:
        corrected_chromaticity = np.asarray(corrected_chromaticity, dtype=float).copy()
        if (corrected_chromaticity.shape != (2,) or
                not np.all(np.isfinite(corrected_chromaticity))):
            raise ValueError('corrected_chromaticity must contain two finite chromaticities')
    n_coefficients = (n + 1) * (n + 2) // 2
    if n_samples is None:
        n_samples = n_coefficients + 10
    if (not isinstance(n_samples, (int, np.integer)) or n_samples < n_coefficients):
        raise ValueError(f'n_samples must be an integer >= {n_coefficients}')

    indices = element_index(ring)
    for name in names + correction_names:
        if name not in indices:
            raise ValueError(f'Sextupole name not found: {name}')
        if any(ring.elements[i].type != 'Sextupole' for i in indices[name]):
            raise ValueError(f'Element is not a sextupole: {name}')
    variable_indices = [[i for name in group for i in indices[name]]
                        for group in groups]
    initial_k2 = {i: ele.k2 for i, ele in enumerate(ring.elements)
                  if ele.type == 'Sextupole'}
    group_strengths = [np.array([initial_k2[i] for i in group])
                       for group in variable_indices]
    if any(not np.all(np.isfinite(strengths)) for strengths in group_strengths):
        raise ValueError('Initial variable sextupole strengths must be finite')
    centers = np.array([np.mean(strengths) for strengths in group_strengths])
    half_widths = np.array([max(0.5 * np.mean(np.abs(strengths)), 1.)
                            for strengths in group_strengths])
    lb, ub = centers - half_widths, centers + half_widths
    sample_method = 'lhs' if n >= 4 else 'uniform'

    def apply_strengths(values):
        # Start every sample from the same strengths, including correction families.
        for i, k2 in initial_k2.items():
            ring.elements[i].k2 = k2
        for value, group_indices in zip(values, variable_indices):
            for i in group_indices:
                ring.elements[i].k2 = value
        ring.linear_optics()
        chromaticity_correction(ring, correction_names, target=corrected_chromaticity,
                                use_tracked_ksi=use_tracked_ksi)
        ring.linear_optics()
    def target_function(values):
        apply_strengths(values)
        rdts = ring.driving_terms(verbose=False)
        return ring_averaged_rdt_metrics(rdts, w=0).third_order ** 2

    fitter = QuadraticFunctionFitter(target_function, bounds=(lb, ub))
    updated = False
    try:
        if corrected_chromaticity is None:
            ring.linear_optics()
            corrected_chromaticity = np.array([ring.xi_x, ring.xi_y], dtype=float)
            if not np.all(np.isfinite(corrected_chromaticity)):
                raise ValueError('Current ring chromaticities must be finite')
        form = fitter.fit(n_samples=n_samples, sample_method=sample_method)
        try:
            np.linalg.cholesky(form.A)
        except np.linalg.LinAlgError as exc:
            raise ValueError('The fitted quadratic form has no unique minimum') from exc
        minimum, k2_values = form.minimum()
        if update_k2:
            apply_strengths(k2_values)
            updated = True
        return minimum**0.5, k2_values
    finally:
        if not updated:
            for i, k2 in initial_k2.items():
                ring.elements[i].k2 = k2
            ring.linear_optics()
