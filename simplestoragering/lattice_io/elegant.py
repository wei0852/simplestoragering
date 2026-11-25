"""Elegant lattice export."""
import math
import time

from ..CSLattice import CSLattice
from ..Drift import Drift
from ..HBend import HBend
from ..Octupole import Octupole
from ..Quadrupole import Quadrupole
from ..RFCavity import RFCavity
from ..Sextupole import Sextupole


def _slices(ele, new_version):
    """N_SLICES for ELEGANT >= 2021.1, N_KICKS for older versions."""
    return (f'N_SLICES = {ele.n_slices}' if new_version
            else f'N_KICKS = {ele.n_slices * 4}')


def _format_drift(ele, new_version, rad_suffix):
    return f'{ele.name:6}: EDRIFT, l = {ele.length:.6f}\n'


def _format_quad(ele, new_version, rad_suffix):
    return (f'{ele.name:6}: KQUAD, l = {ele.length:.6f}, k1 = {ele.k1:.6f}, '
            f'{_slices(ele, new_version)}{rad_suffix}\n')


def _format_bend(ele, new_version, rad_suffix):
    # NOTE: the old code used n_slices (not n_slices * 4) for N_KICKS; kept as is.
    if new_version:
        slices = f'N_SLICES = {ele.n_slices}'
    else:
        slices = f'N_KICKS = {ele.n_slices}'
    return (f'{ele.name:6}: csbend, l = {ele.length:.6f}, angle = {ele.theta:.10f}, k1 '
            f'= {ele.k1:.6f}, e1 = {ele.theta_in:.10f}, e2 = '
            f'{ele.theta_out:.10f}, HGAP={ele.gap / 2}, '
            f'FINT1={ele.fint_in}, FINT2={ele.fint_out}, {slices}{rad_suffix}\n')


def _format_sext(ele, new_version, rad_suffix):
    return (f'{ele.name:6}: KSEXT, l = {ele.length:.6f}, k2 = {ele.k2:.6f}, '
            f'{_slices(ele, new_version)}{rad_suffix}\n')


def _format_oct(ele, new_version, rad_suffix):
    return (f'{ele.name:6}: KOCT, l = {ele.length:.6f}, k3 = {ele.k3:.6f}, '
            f'{_slices(ele, new_version)}{rad_suffix}\n')


def _format_rf(ele, new_version, rad_suffix):
    return (f'{ele.name:6}: RFCA, freq={ele.f_rf}, volt={ele.voltage * 1e6}, '
            f'phase={180 - ele.phase * 180 / math.pi}\n')


# Order of this tuple also defines the order of the tables in the .lte file.
_FORMATTERS = (
    (Drift, _format_drift),
    (Quadrupole, _format_quad),
    (HBend, _format_bend),
    (Sextupole, _format_sext),
    (Octupole, _format_oct),
    (RFCavity, _format_rf),
)


def output_elegant_file(lattice: CSLattice, filename=None, new_version=True):
    """output .lte file for ELEGANT (https://ops.aps.anl.gov/manuals/elegant_latest/elegant.html)
    the suffix '.lte' will be added to the end of the file name.
    If filename is None, the lattice name will be used as the file name.
    If new_version (after ELEGANT 2021.1), use N_SLICES parameter, else use N_KICKS.
    """
    filename = f'{lattice.name}.lte' if filename is None else filename + '.lte'
    rad_suffix = ', SYNCH_RAD=1' if lattice.RFCavity is not None else ''

    sequence = []                       # every element, duplicates included
    defined = set()                     # names already written to the table
    tables = {cls: [] for cls, _ in _FORMATTERS}

    for ele in lattice.elements:
        for cls, formatter in _FORMATTERS:
            if isinstance(ele, cls):
                sequence.append(ele.name)
                if ele.name not in defined:
                    defined.add(ele.name)
                    tables[cls].append(formatter(ele, new_version, rad_suffix))
                break

    blocks = [''.join(tables[cls]) for cls, _ in _FORMATTERS if tables[cls]]
    lines = [f'! output from python code at {time.strftime("%m.%d,%H:%M")}\n',
             '\n!------------- table of elements ----------------\n\n',
             '\n'.join(blocks),
             '\n{------ table of segments --------------------------}\n\n']
    if lattice.n_periods == 1:
        lines.append(f'ring0 : line=({", ".join(sequence)})\n')
    else:
        lines.append(f'cell : line=({", ".join(sequence)})\n')
        lines.append(f'ring0 : line=({lattice.n_periods}*cell)\n')
    lines.append('MAL: MALIGN,ON_PASS=1,FORCE_MODIFY_MATRIX=0\n')
    lines.append('ring: line=(MAL, ring0)')

    with open(filename, 'w', encoding='utf-8') as file:
        file.writelines(lines)
