"""Lattice element lookup helpers."""
from .CSLattice import CSLattice


def element_index(ring: CSLattice) -> dict:
    ele_idx = {}
    for i, ele in enumerate(ring.elements):
        if ele.name not in ele_idx:
            ele_idx[ele.name] = [i]
        else:
            ele_idx[ele.name].append(i)
    return ele_idx
