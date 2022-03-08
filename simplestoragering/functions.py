# -*- coding: utf-8 -*-
import copy

from .particles import Beam7
import numpy as np
import time
import simplestoragering as ssr
from .constants import pi


def next_twiss(matrix, twiss0):
    sub = matrix[:2, :2]
    twiss1 = np.zeros(12)
    matrix_cal = np.array([[sub[0, 0] ** 2, -2 * sub[0, 0] * sub[0, 1], sub[0, 1] ** 2],
                           [-sub[0, 0] * sub[1, 0], 2 * sub[0, 1] * sub[1, 0] + 1, -sub[0, 1] * sub[1, 1]],
                           [sub[1, 0] ** 2, -2 * sub[1, 0] * sub[1, 1], sub[1, 1] ** 2]])
    twiss1[:3] = matrix_cal.dot(twiss0[:3])
    sub = matrix[2:4, 2:4]
    matrix_cal = np.array([[sub[0, 0] ** 2, -2 * sub[0, 0] * sub[0, 1], sub[0, 1] ** 2],
                           [-sub[0, 0] * sub[1, 0], 2 * sub[0, 1] * sub[1, 0] + 1, -sub[0, 1] * sub[1, 1]],
                           [sub[1, 0] ** 2, -2 * sub[1, 0] * sub[1, 1], sub[1, 1] ** 2]])
    twiss1[3:6] = matrix_cal.dot(twiss0[3:6])
    twiss1[6:8] = matrix[:2, :2].dot(twiss0[6:8]) + np.array([matrix[0, 5], matrix[1, 5]])
    twiss1[8:10] = matrix[2:4, 2:4].dot(twiss0[8:10]) + np.array([matrix[2, 5], matrix[3, 5]])
    dpsix = np.arctan(matrix[0, 1] / (matrix[0, 0] * twiss0[0] - matrix[0, 1] * twiss0[1]))
    while dpsix < 0:
        dpsix += pi
    twiss1[10] = twiss0[10] + dpsix
    dpsiy = np.arctan(matrix[2, 3] / (matrix[2, 2] * twiss0[3] - matrix[2, 3] * twiss0[4]))
    while dpsiy < 0:
        dpsiy += pi
    twiss1[11] = twiss0[11] + dpsiy
    return twiss1


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

    # print('\n-------------------\ntracking 4D closed orbit:\n')
    # xco = np.array([0, 0, 0, 0, delta])
    # matrix = np.zeros([5, 5])
    # resdl = 1
    # j = 1
    # precision = 1e-9
    # d = np.zeros(5)
    # while j <= 10 and resdl > 1e-16:
    #     beam = np.eye(6, 7) * precision
    #     for i in range(6):
    #         beam[:4, i] = beam[:4, i] + xco[:4]
    #         beam[5, i] = beam[5, i] + delta
    #     beam[5, 6] = beam[5, 6] + delta
    #     for ele in element_list:
    #         ele.closed_orbit = beam[:, 6]
    #         beam = ele.symplectic_track(beam)
    #     for i in range(4):
    #         matrix[:4, i] = (beam[:4, i] - beam[:4, 6]) / precision
    #     matrix[:4, 4] = (beam[:4, 5] - beam[:4, 6]) / precision
    #     matrix[4, 4] = 1
    #     d[:4] = beam[:4, 6] - xco[:4]
    #     dco = np.linalg.inv(np.identity(5) - matrix).dot(d)
    #     xco = xco + dco
    #     resdl = dco.dot(dco.T)
    #     print(f'iterated {j} times, current result is \n    {beam[:4, 6]}\n')
    #     j += 1
    # print(f'closed orbit at s=0 is \n    {xco}\n')
    # beam = np.eye(6, 7) * precision
    # for i in range(4):
    #     beam[i, :] = beam[i, :] + xco[i]
    # for el in element_list:
    #     el.closed_orbit = beam[:, 6]
    #     beam = el.symplectic_track(beam)
    # print(f'\nclosed orbit at end is \n    {beam[:4, 6]}\n')

    print('\n-------------------\ntracking 4D closed orbit:\n')
    xco = np.array([0, 0, 0, 0, 0, delta])
    matrix = np.zeros([6, 6])
    resdl = 1
    j = 1
    precision = 1e-9
    while j <= 10 and resdl > 1e-16:
        beam = np.eye(6, 7) * precision
        for i in range(7):
            beam[:, i] = beam[:, i] + xco
        for ele in element_list:
            # ele.closed_orbit = beam[:, 6]
            beam = ele.symplectic_track(beam)
        for i in range(7):
            beam[4, i] = 0
            beam[5, i] = delta
        for i in range(6):
            matrix[:, i] = (beam[:, i] - beam[:, 6]) / precision
        d = beam[:, 6] - xco
        dco = np.linalg.inv(np.identity(6) - matrix).dot(d)
        xco = xco + dco
        resdl = dco.dot(dco.T)
        print(f'iterated {j} times, current result is \n    {beam[:, 6]}\n')
        j += 1
    print(f'closed orbit at s=0 is \n    {xco}\n')
    beam = np.eye(6, 7) * precision
    for i in range(7):
        beam[:, i] = beam[:, i] + xco
    for el in element_list:
        el.closed_orbit = beam[:, 6]
        beam = el.symplectic_track(beam)
    print(f'\nclosed orbit at end is \n    {beam[:, 6]}\n')


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
                    sext_list.append(f'{ele.name:6}: sextupole, l = {ele.length:.6f}, k = {ele.k2 / 2:.6f}, n = 4;\r\n')
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
        if lattice.periods_number == 1:
            file.write(f'ring : {ele_list[0]}')
            for i in range(len(ele_list)-1):
                file.write(f', {ele_list[i+1]}')
        else:
            file.write(f'cell : {ele_list[0]}')
            for i in range(len(ele_list)-1):
                file.write(f', {ele_list[i+1]}')
            file.write(f';\r\nring : {lattice.periods_number}*cell')
        file.write(';\r\n')


# def read_opa_file(filename):
#     """read opa file and return cs_lattice, Only some components can be recognized
#
#     Only the file output by opa can be recognized. If the opening fails,
#     you can try to open and save the file with opa first."""
#
#     with open(filename, 'r') as file:
#         for line in file.readlines():
#             item


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
                    quad_list.append(f'{ele.name:6}: KQUAD, l = {ele.length:.6f}, k1 = {ele.k1:.6f}, N_SLICES = {ele.n_slices}\n')
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
        if lattice.periods_number == 1:
            file.write(f'ring : line=({ele_list[0]}')
            for i in range(len(ele_list) - 1):
                file.write(f', {ele_list[i + 1]}')
        else:
            file.write(f'cell : line=({ele_list[0]}')
            for i in range(len(ele_list) - 1):
                file.write(f', {ele_list[i + 1]}')
            file.write(f')\nring : line=({lattice.periods_number}*cell')
        file.write(')\n')


def chromaticity_correction(lattice, sextupole_name_list: list, target=None, initial_k2=None, update_data=True):
    """correct chromaticity. target = [xi_x, xi_y], initial_k2 should have the same length as sextupole_name_list."""

    target = [1, 1] if target is None else target
    num_sext = len(sextupole_name_list)
    remaining_xi_x = lattice.natural_xi_x * 4 * pi
    remaining_xi_y = lattice.natural_xi_y * 4 * pi
    # initialize the weight
    weight_x = {n: 0 for n in sextupole_name_list}
    weight_y = {n: 0 for n in sextupole_name_list}
    for ele in lattice.ele_slices:
        if ele.type == 'Sextupole':
            if ele.name not in sextupole_name_list:
                remaining_xi_x += ele.betax * ele.etax * ele.length * ele.k2
                remaining_xi_y += - ele.betay * ele.etax * ele.length * ele.k2
            else:
                weight_x[ele.name] += ele.betax * ele.etax * ele.length
                weight_y[ele.name] += - ele.betay * ele.etax * ele.length
    remaining_xi_x = remaining_xi_x / 4 / pi
    remaining_xi_y = remaining_xi_y / 4 / pi
    for ele_name in weight_x:
        weight_x[ele_name] = weight_x[ele_name] / 4 / pi
        weight_y[ele_name] = weight_y[ele_name] / 4 / pi
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
    print(f'result is\n{initial_k2}\n')
    if update_data:
        for ele in lattice.elements:
            if ele.name in sextupole_name_list:
                ele.k2 = initial_k2[sextupole_name_list.index(ele.name)]
        for ele in lattice.ele_slices:
            if ele.name in sextupole_name_list:
                ele.k2 = initial_k2[sextupole_name_list.index(ele.name)]
    # TODO: 如何控制六极磁铁强度上限
