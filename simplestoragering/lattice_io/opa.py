"""OPA lattice export."""
import numpy as np
from ..globalvars import refEnergy
from ..Octupole import Octupole
from ..CSLattice import CSLattice

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
