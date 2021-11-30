# -*- coding: utf-8 -*-
from .particles import Beam7
from numpy import zeros
import simplestoragering as ssr
from .constants import pi


def compute_transfer_matrix_by_tracking(comp_list: list, beam: list, with_e_loss=False):
    """return transfer matrix from com_i to com_f"""

    assert len(beam) == 6
    beam = Beam7(beam)
    ele_slices = []
    current_s = 0
    current_identifier = 0
    for ele in comp_list:
        [new_list, current_s] = ele.slice(current_s, current_identifier)
        ele_slices += new_list
        current_identifier += 1
    for ele in ele_slices:
        if with_e_loss:
            beam = ele.real_track(beam)
        else:
            beam = ele.symplectic_track(beam)
    matrix = zeros([6, 6])
    for i in range(6):
        matrix[:, i] = (beam.matrix[:, i] - beam.matrix[:, 6]) / beam.precision
    return matrix


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
