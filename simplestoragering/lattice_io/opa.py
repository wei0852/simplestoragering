"""OPA lattice export."""
import math

from ..CSLattice import CSLattice
from ..Drift import Drift
from ..HBend import HBend
from ..Octupole import Octupole
from ..Quadrupole import Quadrupole
from ..Sextupole import Sextupole
from ..globalvars import refEnergy


def _format_drift(ele):
    return [f'{ele.name:6}: drift, l = {ele.length:.6f};\r\n']


def _format_quad(ele):
    return [f'{ele.name:6}: quadrupole, l = {ele.length:.6f}, k = {ele.k1:.6f};\r\n']


def _format_bend(ele):
    return [f'{ele.name:6}: bending, l = {ele.length:.6f}, t = {ele.theta * 180 / math.pi:.6f}, k '
            f'= {ele.k1:.6f}, t1 = {ele.theta_in * 180 / math.pi:.6f}, t2 = '
            f'{ele.theta_out * 180 / math.pi:.6f}, gap = {ele.gap};\r\n']


def _format_sext(ele):
    return [f'{ele.name:6}: sextupole, l = {ele.length:.6f}, k = {ele.k2 / 2:.6f}, n = 4;\r\n']


def _format_oct(ele):
    """Write an octupole as a drift - multipole - drift sequence (OPA has no octupole)."""
    return [
        f'{ele.name + "_d":6}: drift, l = {ele.length / ele.n_slices / 2:.6f};\r\n',
        f'{ele.name + "_k":6}: multipole, n = 4, k = {ele.k3 * ele.length / ele.n_slices / 6:.6f};\r\n',
        f'{ele.name}_slice: {ele.name}_d, {ele.name}_k, {ele.name}_d;\r\n',
        f'{ele.name}: {ele.n_slices}*{ele.name}_slice;\r\n',
    ]


# Order of this tuple also defines the order of the tables in the .opa file.
_FORMATTERS = (
    (Drift, _format_drift),
    (Quadrupole, _format_quad),
    (HBend, _format_bend),
    (Sextupole, _format_sext),
    (Octupole, _format_oct),
)


def output_opa_file(lattice: CSLattice, file_name=None):
    """output .opa file for OPA (https://ados.web.psi.ch/opa/), the suffix '.opa' will be added to the end of the file name.
    If file_name is None, the lattice name will be used as the file name."""
    file_name = f'{lattice.name}.opa' if file_name is None else file_name + '.opa'

    sequence = []                       # every element, duplicates included
    defined = set()                     # names already written to the table
    tables = {cls: [] for cls, _ in _FORMATTERS}

    for ele in lattice.elements:
        for cls, formatter in _FORMATTERS:
            if isinstance(ele, cls):
                sequence.append(ele.name)
                if ele.name not in defined:
                    defined.add(ele.name)
                    tables[cls].extend(formatter(ele))
                break

    blocks = [''.join(tables[cls]) for cls, _ in _FORMATTERS if tables[cls]]
    lines = [f'energy = {refEnergy() / 1000: 6f};\r\n',
             '\r\n\r\n{------ table of elements -----------------------------}\r\n\r\n',
             '\r\n'.join(blocks),
             '\r\n\r\n{------ table of segments --------------------------}\r\n\r\n']
    if lattice.n_periods == 1:
        lines.append(f'ring : {", ".join(sequence)};\r\n')
    else:
        lines.append(f'cell : {", ".join(sequence)};\r\n')
        lines.append(f'ring : {lattice.n_periods}*cell;\r\n')

    with open(file_name, 'w', encoding='utf-8') as file:
        file.writelines(lines)
