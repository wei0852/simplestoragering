# -*- coding: utf-8 -*-
# cython: language_level=3

import numpy as np
import time
from .globalvars import refEnergy
from .Octupole import Octupole
from .CSLattice import CSLattice
from .exceptions import Unstable


def output_opa_file(lattice: CSLattice, file_name=None):
    """output .opa file for OPA (https://ados.web.psi.ch/opa/), the suffix '.opa' will be added to the end of the file name."""
    
    file_name = 'output_opa.opa' if file_name is None else file_name + '.opa'
    with open(file_name, 'w') as file:
        file.write(f'energy = {refEnergy() / 1000: 6f};\r\n')
        file.write('\r\n\r\n{------ table of elements -----------------------------}\r\n\r\n')
        ele_list = []
        drift_list = []
        quad_list = []
        bend_list = []
        sext_list = []
        oct_list = []
        for ele in lattice.elements:
            if ele.name not in ele_list:
                if ele.type == 'Drift':
                    drift_list.append(f'{ele.name:6}: drift, l = {ele.length:.6f};\r\n')
                elif ele.type == 'Quadrupole':
                    quad_list.append(f'{ele.name:6}: quadrupole, l = {ele.length:.6f}, k = {ele.k1:.6f};\r\n')
                elif ele.type == 'HBend':
                    bend_list.append(f'{ele.name:6}: bending, l = {ele.length:.6f}, t = {ele.theta * 180 / np.pi:.6f}, k '
                                     f'= {ele.k1:.6f}, t1 = {ele.theta_in * 180 / np.pi:.6f}, t2 = '
                                     f'{ele.theta_out * 180 / np.pi:.6f}, gap = {ele.gap};\r\n')
                elif ele.type == 'Sextupole':
                    sext_list.append(f'{ele.name:6}: sextupole, l = {ele.length:.6f}, k = {ele.k2 / 2:.6f}, n = 4;\r\n')
                elif isinstance(ele, Octupole):
                    oct_list.append(f'{ele.name + "_d":6}: drift, l = {ele.length / ele.n_slices / 2:.6f};\r\n')
                    oct_list.append(f'{ele.name + "_k":6}: multipole, n = 4, k = {ele.k3 * ele.length / ele.n_slices / 6:.6f};\r\n')
                    oct_list.append(f'{ele.name}: {(ele.name + "_d, " + ele.name + "_k, " + ele.name + "_d, ") * (ele.n_slices - 1)}{ele.name}_d, {ele.name}_k, {ele.name}_d;\r\n')
            if ele.type == 'Drift' or ele.type == 'Quadrupole' or ele.type == 'HBend' or ele.type == 'Sextupole' or isinstance(ele, Octupole):
                ele_list.append(ele.name)
        for ele in drift_list:
            file.write(ele)
        file.write('\r\n')
        for ele in quad_list:
            file.write(ele)
        file.write('\r\n')
        for ele in bend_list:
            file.write(ele)
        file.write('\r\n')
        for ele in sext_list:
            file.write(ele)
        for ele in oct_list:
            file.write(ele)
        file.write('\r\n\r\n{------ table of segments --------------------------}\r\n\r\n')
        if lattice.n_periods == 1:
            file.write(f'ring : {ele_list[0]}')
            for i in range(len(ele_list)-1):
                file.write(f', {ele_list[i+1]}')
        else:
            file.write(f'cell : {ele_list[0]}')
            for i in range(len(ele_list)-1):
                file.write(f', {ele_list[i+1]}')
            file.write(f';\r\nring : {lattice.n_periods}*cell')
        file.write(';\r\n')


def output_elegant_file(lattice: CSLattice, filename=None, new_version=True):
    """output .lte file for ELEGANT (https://ops.aps.anl.gov/manuals/elegant_latest/elegant.html)
    the suffix '.lte' will be added to the end of the file name.
    If new_version (after ELEGANT 2021.1), use N_SLICES parameter, else use N_KICKS.
    """

    filename = 'output_lte.lte' if filename is None else filename + '.lte'
    with open(filename, 'w') as file:
        file.write(f'! output from python code at {time.strftime("%m.%d,%H:%M")}\n')
        file.write(f'\n!------------- table of elements ----------------\n\n')
        ele_list = []
        drift_list = []
        quad_list = []
        bend_list = []
        sext_list = []
        oct_list = []
        for ele in lattice.elements:
            if ele.name not in ele_list:
                if ele.type == 'Drift':
                    drift_list.append(f'{ele.name:6}: EDRIFT, l = {ele.length:.6f}\n')
                elif ele.type == 'Quadrupole':
                    if new_version:
                        quad_list.append(f'{ele.name:6}: KQUAD, l = {ele.length:.6f}, k1 = {ele.k1:.6f}, N_SLICES = {ele.n_slices}\n')
                    else:
                        quad_list.append(f'{ele.name:6}: KQUAD, l = {ele.length:.6f}, k1 = {ele.k1:.6f}, N_KICKS = {ele.n_slices * 4}\n')
                elif ele.type == 'HBend':
                    if new_version:
                        bend_list.append(f'{ele.name:6}: csbend, l = {ele.length:.6f}, angle = {ele.theta:.6f}, k1 '
                                     f'= {ele.k1:.6f}, e1 = {ele.theta_in:.6f}, e2 = '
                                     f'{ele.theta_out:.6f}, HGAP={ele.gap / 2}, FINT1={ele.fint_in}, FINT2={ele.fint_out}, N_SLICES = {ele.n_slices}\n')
                    else:
                        bend_list.append(f'{ele.name:6}: csbend, l = {ele.length:.6f}, angle = {ele.theta:.6f}, k1 '
                                     f'= {ele.k1:.6f}, e1 = {ele.theta_in:.6f}, e2 = '
                                     f'{ele.theta_out:.6f}, HGAP={ele.gap / 2}, FINT1={ele.fint_in}, FINT2={ele.fint_out}, N_KICKS = {ele.n_slices}\n')
                elif ele.type == 'Sextupole':
                    if new_version:
                        sext_list.append(f'{ele.name:6}: KSEXT, l = {ele.length:.6f}, k2 = {ele.k2:.6f}, N_SLICES = {ele.n_slices}\n')
                    else:
                        sext_list.append(f'{ele.name:6}: KSEXT, l = {ele.length:.6f}, k2 = {ele.k2:.6f}, N_KICKS = {ele.n_slices * 4}\n')
                elif ele.type == 'Octupole':
                    if new_version:
                        oct_list.append(f'{ele.name:6}: KOCT, l = {ele.length:.6f}, k3 = {ele.k3:.6f}, N_SLICES = {ele.n_slices}\n')
                    else:
                        oct_list.append(f'{ele.name:6}: KOCT, l = {ele.length:.6f}, k3 = {ele.k3:.6f}, N_KICKS = {ele.n_slices * 4}\n')
            if ele.type == 'Drift' or ele.type == 'Quadrupole' or ele.type == 'HBend' or ele.type == 'Sextupole' or ele.type == 'Octupole':
                ele_list.append(ele.name)
        for ele in drift_list:
            file.write(ele)
        file.write('\n')
        for ele in quad_list:
            file.write(ele)
        file.write('\n')
        for ele in bend_list:
            file.write(ele)
        file.write('\n')
        for ele in sext_list:
            file.write(ele)
        for ele in oct_list:
            file.write(ele)
        file.write('\n\n{------ table of segments --------------------------}\n\n')
        if lattice.n_periods == 1:
            file.write(f'ring0 : line=({ele_list[0]}')
            for i in range(len(ele_list) - 1):
                file.write(f', {ele_list[i + 1]}')
        else:
            file.write(f'cell : line=({ele_list[0]}')
            for i in range(len(ele_list) - 1):
                file.write(f', {ele_list[i + 1]}')
            file.write(f')\nring0 : line=({lattice.n_periods}*cell')
        file.write(')\n')
        file.write('MAL: MALIGN,ON_PASS=1,FORCE_MODIFY_MATRIX=0\n')
        file.write('ring: line=(MAL, ring0)')


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


def element_index(ring: CSLattice) -> dict:
    ele_idx = {}
    for i, ele in enumerate(ring.elements):
        if ele.name not in ele_idx:
            ele_idx[ele.name] = [i]
        else:
            ele_idx[ele.name].append(i)
    return ele_idx
