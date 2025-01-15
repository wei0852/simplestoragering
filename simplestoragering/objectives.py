"""This file provides several functions for conveniently calculating objectives to optimize the lattice.
HOWEVER, it's important to note that some of these functions are based on experiential knowledge rather than strict validation.
Users should exercise caution when utilizing these objectives.
"""
import numpy as np
from .DrivingTerms import DrivingTerms


def quantify_rdt_fluctuation(rdt: DrivingTerms, w=0.01):
    """Use the rms value along the ring to quantify natural RDT fluctuation.

    w is the weight for the 4th-order RDTs, if w == 0, return two values: [h3rms, h4rms],
    else return h3rms + h4rms * w.
    The coefficient w can be estimated by action of particle at the anticipated DA,
    w ~ (2J_x)^0.5 = x / betax^0.5,  usually 0.01 is OK."""
    natural = rdt.natural_fluctuation()
    f3rms = 0
    f4rms = 0
    for k in ['f2100', 'f3000', 'f1011', 'f1002', 'f1020']:
        f3rms += np.average(natural[k] ** 2)  # ** 0.5 ** 2
    f3rms = f3rms ** 0.5
    for k in ['f3100', 'f4000', 'f2011', 'f1120', 'f2002', 'f2020', 'f0031', 'f0040']:
        f4rms += np.average(natural[k] ** 2)  # ** 0.5 ** 2
    f4rms = f4rms ** 0.5
    if w == 0:
        return f3rms, f4rms
    else:
        return f3rms + f4rms * w


def compute_h3ring(rdt: DrivingTerms):
    h3ring = 0
    for k in ['h2100', 'h3000', 'h1011', 'h1002', 'h1020']:
        h3ring += abs(rdt[k]) ** 2
    h3ring = h3ring ** 0.5
    return h3ring


def compute_h4ring(rdt: DrivingTerms):
    h4ring = 0
    for k in ['h3100', 'h4000', 'h2011', 'h1120', 'h2002', 'h2020', 'h0031', 'h0040']:
        h4ring += abs(rdt[k]) ** 2
    h4ring = h4ring ** 0.5
    return h4ring
