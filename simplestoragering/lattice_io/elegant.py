"""Elegant lattice export."""
import time
from ..CSLattice import CSLattice

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
                        bend_list.append(f'{ele.name:6}: csbend, l = {ele.length:.6f}, angle = {ele.theta:.10f}, k1 '
                                     f'= {ele.k1:.6f}, e1 = {ele.theta_in:.10f}, e2 = '
                                     f'{ele.theta_out:.10f}, HGAP={ele.gap / 2}, FINT1={ele.fint_in}, FINT2={ele.fint_out}, N_SLICES = {ele.n_slices}\n')
                    else:
                        bend_list.append(f'{ele.name:6}: csbend, l = {ele.length:.6f}, angle = {ele.theta:.10f}, k1 '
                                     f'= {ele.k1:.6f}, e1 = {ele.theta_in:.10f}, e2 = '
                                     f'{ele.theta_out:.10f}, HGAP={ele.gap / 2}, FINT1={ele.fint_in}, FINT2={ele.fint_out}, N_KICKS = {ele.n_slices}\n')
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
