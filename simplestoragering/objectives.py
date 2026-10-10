"""This file provides several functions for conveniently calculating objectives to optimize the lattice.
HOWEVER, it's important to note that some of these functions are based on experiential knowledge rather than strict validation.
Users should exercise caution when utilizing these objectives.
"""
from dataclasses import dataclass
from typing import Union
import numpy as np
from .DrivingTerms import DrivingTerms


def quantify_rdt_fluctuation(rdt: DrivingTerms, w: float = 0.01) -> Union[float, tuple[float, float]]:
    """Use the rms value along the ring to quantify natural RDT fluctuation.

    w is the weight for the 4th-order RDTs, if w == 0, return a tuple (h3rms, h4rms),
    else return h3rms + h4rms * w.
    The coefficient w can be estimated by action of particle at the anticipated DA,
    w ~ (2J_x)^0.5 = x / betax^0.5,  usually 0.01 is OK."""
    natural = rdt.natural_fluctuation()
    l_diff = np.diff(rdt.s, prepend=0)
    f3rms = 0
    f4rms = 0
    for k in ['f2100', 'f3000', 'f1011', 'f1002', 'f1020']:
        f3rms += np.average(natural[k] ** 2, weights=l_diff)  # ** 0.5 ** 2
    f3rms = f3rms ** 0.5
    for k in ['f3100', 'f4000', 'f2011', 'f1120', 'f2002', 'f2020', 'f0031', 'f0040']:
        f4rms += np.average(natural[k] ** 2, weights=l_diff)  # ** 0.5 ** 2
    f4rms = f4rms ** 0.5
    if w == 0:
        return f3rms, f4rms
    else:
        return f3rms + f4rms * w


def ring_averaged_rdt(rdt: DrivingTerms, w: float = 0.01) -> Union[float, tuple[float, float]]:
    """Return (third, fourth) at w=0, otherwise third + w * fourth.

    For a fixed result structure, use ring_averaged_rdt_metrics().
    """
    return quantify_rdt_fluctuation(rdt, w)


def compute_h3ring(rdt: DrivingTerms) -> float:
    h3ring = 0
    for k in ['h2100', 'h3000', 'h1011', 'h1002', 'h1020']:
        h3ring += abs(rdt[k]) ** 2
    h3ring = h3ring ** 0.5
    return h3ring


def compute_h4ring(rdt: DrivingTerms) -> float:
    h4ring = 0
    for k in ['h3100', 'h4000', 'h2011', 'h1120', 'h2002', 'h2020', 'h0031', 'h0040']:
        h4ring += abs(rdt[k]) ** 2
    h4ring = h4ring ** 0.5
    return h4ring


@dataclass(frozen=True)
class RDTMetrics:
    """Length-weighted natural RDT fluctuation metrics, grouped by order.

    weighted = third_order + w * fourth_order for the requested weight w.
    These are RMS aggregates, not complex averages of the RDTs.
    """
    third_order: float
    fourth_order: float
    weighted: float


def ring_averaged_rdt_metrics(rdt: DrivingTerms, w: float = 0.01) -> RDTMetrics:
    """Return all three metrics for every weight, including zero.

    Uses the same longitudinal weights and RDT groups as
    quantify_rdt_fluctuation(). At w=0, weighted equals third_order;
    fourth_order is still calculated and included.
    """
    third, fourth = quantify_rdt_fluctuation(rdt, w=0)
    return RDTMetrics(float(third), float(fourth), float(third + w * fourth))
