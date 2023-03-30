# -*- coding: utf-8 -*-

import numpy as np
import time
import simplestoragering as ssr
from .globalvars import pi


def compute_transfer_matrix_by_tracking(element_list: list, particle, with_e_loss=False, precision=1e-9):
    """calculate the transfer matrix by tracking, the tracking data is copied from SAMM (Andrzej Wolski).

    element_list: list of elements.
    particle: list with a length of 6"""

    assert len(particle) == 6
    if isinstance(particle, list):
        particle = np.array(particle)
    else:
        assert isinstance(particle, np.ndarray)
    beam = np.eye(6, 7) * precision
    for i in range(6):
        beam[:, i] = beam[:, i] + particle
    for ele in element_list:
        if with_e_loss:
            beam = ele.real_track(beam)
        else:
            beam = ele.symplectic_track(beam)
    matrix = np.zeros([6, 6])
    for i in range(6):
        matrix[:, i] = (beam[:, i] - beam[:, 6]) / precision
    return matrix


def track_4d_closed_orbit(element_list, delta):
    """4D track to compute closed orbit with energy deviation.

    delta: momentum deviation."""

    print('\n-------------------\ntracking 4D closed orbit:\n')
    xco = np.array([0, 0, 0, 0])
    matrix = np.zeros([4, 4])
    resdl = 1
    j = 1
    precision = 1e-9
    while j <= 10 and resdl > 1e-16:
        beam = np.eye(6, 7) * precision
        for i in range(7):
            beam[:4, i] = beam[:4, i] + xco
            beam[5, i] = beam[5, i] + delta
        for ele in element_list:
            ele.closed_orbit = beam[:, 6]
            beam = ele.symplectic_track(beam)
        for i in range(4):
            matrix[:, i] = (beam[:4, i] - beam[:4, 6]) / precision
        d = beam[:4, 6] - xco
        dco = np.linalg.inv(np.identity(4) - matrix).dot(d)
        xco = xco + dco
        resdl = dco.dot(dco.T)
        print(f'iterated {j} times, current result is \n    {beam[:4, 6]}\n')
        j += 1
    print(f'closed orbit at s=0 is \n    {xco}\n')
    # verify closed orbit.
    beam = np.eye(6, 7) * precision
    for i in range(7):
        beam[:4, i] = beam[:4, i] + xco
        beam[5, i] = beam[5, i] + delta
    for el in element_list:
        el.closed_orbit = beam[:, 6]
        beam = el.symplectic_track(beam)
    print(f'\nclosed orbit at end is \n    {beam[:4, 6]}\n')
    for i in range(4):
        matrix[:, i] = (beam[:4, i] - beam[:4, 6]) / precision
    cos_mu = (matrix[0, 0] + matrix[1, 1]) / 2
    assert abs(cos_mu) < 1, "can not find period solution, UNSTABLE!!!"
    nux = np.arccos(cos_mu) * np.sign(matrix[0, 1]) / 2 / pi
    nuy = np.arccos((matrix[2, 2] + matrix[3, 3]) / 2) * np.sign(matrix[2, 3]) / 2 / pi
    print(f'tune is {nux - np.floor(nux):.6f}, {nuy - np.floor(nuy):.6f}')
    return_data = {'closed_orbit': xco[:4], 'nux': nux - np.floor(nux), 'nuy': nuy - np.floor(nuy)}
    return return_data


def output_opa_file(lattice, file_name=None):
    file_name = 'output_opa.opa' if file_name is None else file_name + '.opa'
    with open(file_name, 'w') as file:
        file.write(f'energy = {ssr.RefParticle.energy / 1000: 6f};\r\n')
        file.write('\r\n\r\n{------ table of elements -----------------------------}\r\n\r\n')
        ele_list = []
        drift_list = []
        quad_list = []
        bend_list = []
        sext_list = []
        for ele in lattice.elements:
            if ele.name not in ele_list:
                if ele.type == 'Drift':
                    drift_list.append(f'{ele.name:6}: drift, l = {ele.length:.6f};\r\n')
                elif ele.type == 'Quadrupole':
                    quad_list.append(f'{ele.name:6}: quadrupole, l = {ele.length:.6f}, k = {ele.k1:.6f};\r\n')
                elif ele.type == 'HBend':
                    bend_list.append(f'{ele.name:6}: bending, l = {ele.length:.6f}, t = {ele.theta * 180 / pi:.6f}, k '
                                     f'= {ele.k1:.6f}, t1 = {ele.theta_in * 180 / pi:.6f}, t2 = '
                                     f'{ele.theta_out * 180 / pi:.6f};\r\n')
                elif ele.type == 'Sextupole':
                    sext_list.append(f'{ele.name:6}: sextupole, l = {ele.length:.6f}, k = {ele.k2 / 2:.6f}, n = {ele.n_slices};\r\n')
            if ele.type == 'Drift' or ele.type == 'Quadrupole' or ele.type == 'HBend' or ele.type == 'Sextupole':
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


def output_elegant_file(lattice, filename=None):
    filename = 'output_lte.lte' if filename is None else filename + '.lte'
    with open(filename, 'w') as file:
        file.write(f'! output from python code at {time.strftime("%m.%d,%H:%M")}\n')
        file.write(f'\n!------------- table of elements ----------------\n\n')
        ele_list = []
        drift_list = []
        quad_list = []
        bend_list = []
        sext_list = []
        for ele in lattice.elements:
            if ele.name not in ele_list:
                if ele.type == 'Drift':
                    drift_list.append(f'{ele.name:6}: EDRIFT, l = {ele.length:.6f}\n')
                elif ele.type == 'Quadrupole':
                    quad_list.append(f'{ele.name:6}: KQUAD, l = {ele.length:.6f}, k1 = {ele.k1:.6f}, N_SLICES = 4\n')
                elif ele.type == 'HBend':
                    bend_list.append(f'{ele.name:6}: csbend, l = {ele.length:.6f}, angle = {ele.theta:.6f}, k1 '
                                     f'= {ele.k1:.6f}, e1 = {ele.theta_in:.6f}, e2 = '
                                     f'{ele.theta_out:.6f}\n')
                elif ele.type == 'Sextupole':
                    sext_list.append(f'{ele.name:6}: KSEXT, l = {ele.length:.6f}, k2 = {ele.k2:.6f}, N_SLICES = {ele.n_slices}\n')
            if ele.type == 'Drift' or ele.type == 'Quadrupole' or ele.type == 'HBend' or ele.type == 'Sextupole':
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
        file.write('\n\n{------ table of segments --------------------------}\n\n')
        if lattice.n_periods == 1:
            file.write(f'ring : line=({ele_list[0]}')
            for i in range(len(ele_list) - 1):
                file.write(f', {ele_list[i + 1]}')
        else:
            file.write(f'cell : line=({ele_list[0]}')
            for i in range(len(ele_list) - 1):
                file.write(f', {ele_list[i + 1]}')
            file.write(f')\nring : line=({lattice.n_periods}*cell')
        file.write(')\n')


def chromaticity_correction(lattice, sextupole_name_list: list, target=None, initial_k2=None, update_data=True, printout=True):
    """correct chromaticity. target = [xi_x, xi_y], initial_k2 should have the same length as sextupole_name_list."""

    target = [1 / lattice.n_periods, 1 / lattice.n_periods] if target is None else [i / lattice.n_periods for i in target]
    num_sext = len(sextupole_name_list)
    remaining_xi_x = lattice.natural_xi_x / lattice.n_periods
    remaining_xi_y = lattice.natural_xi_y / lattice.n_periods
    # initialize the weight
    weight_x = {n: 0 for n in sextupole_name_list}
    weight_y = {n: 0 for n in sextupole_name_list}
    for ele in lattice.elements:
        if ele.type == 'Sextupole':
            if ele.name not in sextupole_name_list:
                optical_data, drop = ele.linear_optics()
                remaining_xi_x += optical_data[5]
                remaining_xi_y += optical_data[6]
            else:
                if ele.k2 == 0:
                    ele.k2 = 1
                optical_data, drop = ele.linear_optics()
                weight_x[ele.name] += optical_data[5] / ele.k2
                weight_y[ele.name] += optical_data[6] / ele.k2
    remaining_xi_x = remaining_xi_x
    remaining_xi_y = remaining_xi_y
    weight_matrix = np.zeros([2, len(sextupole_name_list)])
    for i in range(len(sextupole_name_list)):
        weight_matrix[0, i] = weight_x[sextupole_name_list[i]]
        weight_matrix[1, i] = weight_y[sextupole_name_list[i]]
    initial_k2 = [0 for _ in range(num_sext)] if initial_k2 is None else initial_k2
    xi_x = remaining_xi_x
    xi_y = remaining_xi_y
    for i in range(num_sext):
        xi_x += weight_x[sextupole_name_list[i]] * initial_k2[i]
        xi_y += weight_y[sextupole_name_list[i]] * initial_k2[i]
    delta_target = np.array([-xi_x + target[0], -xi_y + target[1]])
    solution = np.linalg.pinv(weight_matrix).dot(delta_target)
    initial_k2 = [initial_k2[i] + solution[i] for i in range(num_sext)]
    if printout:
        print(f'result is\n{initial_k2}\n')
    if update_data:
        for ele in lattice.elements:
            if ele.name in sextupole_name_list:
                ele.k2 = initial_k2[sextupole_name_list.index(ele.name)]
    # TODO: limit the strength of sextupole
