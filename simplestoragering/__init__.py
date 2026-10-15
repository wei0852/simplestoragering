# -*- coding: utf-8 -*-
"""
simple storage ring for single particle dynamics simulation
"""


from .globalvars import set_ref_energy
from .components import Mark
from .Drift import Drift
from .HBend import HBend
from .Quadrupole import Quadrupole
from .Sextupole import Sextupole
from .Octupole import Octupole
from .RFCavity import RFCavity
from .CSLattice import CSLattice
from .DrivingTerms import DrivingTerms, compute_driving_terms
from .QuadraticFunctionFitter import QuadraticFunctionFitter
from .plotlib import plot_lattice, plot_layout_in_ax, plot_with_background, get_col, plot_resonance_line_in_ax, plot_RDTs_along_ring, plot_rf_bucket
from .lattice_io import output_opa_file, output_elegant_file
from .correction import chromaticity_correction, adjust_tunes
from .longitudinal import calculate_eta_from_alpha, track_alpha_c
from .lattice_utils import element_index
from .line_matrix import line_matrix
from .track import symplectic_track, track_4d_closed_orbit, track_6d_closed_orbit, radiation_track
from .exceptions import ParticleLost, Unstable
from .find_aperture import XYGrid, NLine, XDeltaGrid, LocalMomentumAperture, DynamicAperturePolyhedron, XDeltaLines
from .tune import TuneScan, tune_analysis, tune_shift, tune_shift_with_x, tune_shift_with_y, tune_shift_with_delta, tune_shift_with_amplitude
from .Touschek_Piwinski import TouschekLifetime
from . import objectives

from .objectives import RDTMetrics, ring_averaged_rdt_metrics, theoretical_minimum_f3rms

__all__ = [
    'set_ref_energy',
    'Mark',
    'Drift',
    'HBend',
    'Quadrupole',
    'Sextupole',
    'Octupole',
    'RFCavity',
    'CSLattice',
    'DrivingTerms',
    'compute_driving_terms',
    'QuadraticFunctionFitter',
    'plot_lattice',
    'plot_layout_in_ax',
    'plot_with_background',
    'get_col',
    'plot_resonance_line_in_ax',
    'plot_RDTs_along_ring',
    'plot_rf_bucket',
    'output_opa_file',
    'output_elegant_file',
    'chromaticity_correction',
    'adjust_tunes',
    'calculate_eta_from_alpha',
    'track_alpha_c',
    'element_index',
    'line_matrix',
    'symplectic_track',
    'track_4d_closed_orbit',
    'track_6d_closed_orbit',
    'radiation_track',
    'ParticleLost',
    'Unstable',
    'XYGrid',
    'NLine',
    'XDeltaGrid',
    'LocalMomentumAperture',
    'DynamicAperturePolyhedron',
    'XDeltaLines',
    'TuneScan',
    'tune_analysis',
    'tune_shift',
    'tune_shift_with_x',
    'tune_shift_with_y',
    'tune_shift_with_delta',
    'tune_shift_with_amplitude',
    'TouschekLifetime',
    'objectives',
    'RDTMetrics',
    'ring_averaged_rdt_metrics',
    'theoretical_minimum_f3rms',
]
