"""Tune and chromaticity correction."""
import numpy as np
from .CSLattice import CSLattice
from .exceptions import Unstable

def chromaticity_correction(lattice: CSLattice, sextupole_name_list: list, target: list=None, initial_k2=None, update_sext=True, use_tracked_ksi=True) -> np.ndarray:
    """correct chromaticity. target = [xi_x, xi_y], initial_k2 should have the same length as sextupole_name_list.

    update_sext (bool, optional): Whether to update the sextupole strengths in the lattice. Defaults to True.
    use_tracked_ksi (bool, optional): Whether to calculate chromaticities by tracking.
        If True, chromaticities will be calculated using the CSLattice.track_chromaticity() method.
        If False, the chromaticities are CSLattice.xi_x and CSLattice.xi_y, and this method is faster. Defaults to True.
        """

    target = [1 / lattice.n_periods, 1 / lattice.n_periods] if target is None else [i  / lattice.n_periods for i in target]
    num_sext = len(sextupole_name_list)
    current_xi_x = lattice.xi_x / lattice.n_periods
    current_xi_y = lattice.xi_y / lattice.n_periods
    if use_tracked_ksi:
        try:
            track_xi = lattice.track_chromaticity(delta=1e-5, order=2, verbose=False)
            xi_corr = [current_xi_x - track_xi['xi1x'] / lattice.n_periods, current_xi_y - track_xi['xi1y'] / lattice.n_periods]
        except Unstable:
            xi_corr = [0, 0]
        target[0] += xi_corr[0]
        target[1] += xi_corr[1]
    # initialize the weight
    weight_x = {n: 0.0 for n in sextupole_name_list}
    weight_y = {n: 0.0 for n in sextupole_name_list}
    for ele in lattice.elements:
        if ele.type == 'Sextupole':
            if ele.name in sextupole_name_list:
                if ele.k2 == 0:
                    ele.k2 = 1
                    optical_data, drop = ele.linear_optics()
                    weight_x[ele.name] += optical_data[5]
                    weight_y[ele.name] += optical_data[6]
                else:
                    optical_data, drop = ele.linear_optics()
                    weight_x[ele.name] += optical_data[5] / ele.k2
                    weight_y[ele.name] += optical_data[6] / ele.k2
                    current_xi_x -= optical_data[5]
                    current_xi_y -= optical_data[6]
    weight_matrix = np.zeros([2, len(sextupole_name_list)])
    for i in range(len(sextupole_name_list)):
        weight_matrix[0, i] = weight_x[sextupole_name_list[i]]
        weight_matrix[1, i] = weight_y[sextupole_name_list[i]]
    initial_k2 = [0 for _ in range(num_sext)] if initial_k2 is None else initial_k2
    for i in range(num_sext):
        current_xi_x += weight_x[sextupole_name_list[i]] * initial_k2[i]
        current_xi_y += weight_y[sextupole_name_list[i]] * initial_k2[i]
    delta_target = np.array([-current_xi_x + target[0], -current_xi_y + target[1]])
    solution = np.linalg.pinv(weight_matrix).dot(delta_target)
    initial_k2 = [initial_k2[i] + solution[i] for i in range(num_sext)]
    if update_sext:
        for ele in lattice.elements:
            if ele.name in sextupole_name_list:
                ele.k2 = initial_k2[sextupole_name_list.index(ele.name)]
    # TODO: Control the upper limit of Sextupoles
    return np.array(initial_k2)

def adjust_tunes(lattice: CSLattice, quadrupole_name_list: list, target: list, iterations=5, initialize=True) -> np.ndarray:
    """adjust tunes of lattice.

    initialize: bool=True. If True, the on-momentum linear optics will be calculated before adjustment are made.
                Setting it to False can save time.
    """
    test_k = 1e-6  # TODO: add limit
    if initialize:
        lattice.linear_optics()
    nux0 = lattice.nux / lattice.n_periods
    nuy0 = lattice.nuy / lattice.n_periods
    weight_matrix = np.zeros([2, len(quadrupole_name_list)])
    for i, k in enumerate(quadrupole_name_list):
        test_lattice = lattice * 1  # one period
        found_ele = False
        for ele in test_lattice.elements:
            if ele.name == k:
                ele.k1 = ele.k1 * (1 + test_k)
                found_ele = True
        assert found_ele, f'quadrupole {k} not found.'
        test_lattice.linear_optics()
        weight_matrix[0, i] = test_lattice.nux - nux0
        weight_matrix[1, i] = test_lattice.nuy - nuy0
    sensitive_matrix = np.linalg.pinv(weight_matrix) * test_k
    delta_nux = (target[0] - lattice.nux) / lattice.n_periods
    delta_nuy = (target[1] - lattice.nuy) / lattice.n_periods
    solution = sensitive_matrix.dot(np.array([delta_nux, delta_nuy]))
    k1s = np.zeros(len(quadrupole_name_list))
    for ele in lattice.elements:
        if ele.name in quadrupole_name_list:
            quad_idx = quadrupole_name_list.index(ele.name)
            ele.k1 = ele.k1 * (1 + solution[quad_idx])
            k1s[quad_idx] = ele.k1
    lattice.linear_optics()
    delta_nux = target[0] - lattice.nux
    delta_nuy = target[1] - lattice.nuy
    if (abs(delta_nux) > 1e-3 or abs(delta_nuy) > 1e-3) and iterations > 1:
        return adjust_tunes(lattice, quadrupole_name_list, target, iterations=iterations-1, initialize=False)
    return k1s
