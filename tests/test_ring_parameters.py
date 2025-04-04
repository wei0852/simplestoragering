import numpy as np
import simplestoragering as ssr
import PyNAFF as pnf
import random


def test_tunes(HOA7BA):
    OPA_data = {'nux': 43.299440, 'nuy': 16.298506}
    ELEGANT_data = {'nux': 4.329944e+01, 'nuy': 1.629850e+01}
    # AT_data = 
    ssr_data = {'nux': HOA7BA.nux, 'nuy': HOA7BA.nuy}
    for k in OPA_data:
        diff = min(abs(ssr_data[k] - OPA_data[k]), abs(ssr_data[k] - ELEGANT_data[k]))
        tolerance = abs(OPA_data[k] - ELEGANT_data[k])
        # assert diff < tolerance, f'wrong {k}'
        assert diff < 1e-5, f'wrong {k}, {ssr_data[k]}'


def test_geometry(HOA7BA):
    OPA_data = {'C0': 336.000000, 'I2': 6.3096411E-01, 'I3': 5.5501154E-02, 'U0': 0.208096}
    ELEGANT_data = {'C0': 336.000000 + 1e-9, 'I2': 6.309726e-01, 'I3': 5.550196e-02, 'U0': 2.081046e-01}
    ssr_data = {'C0': HOA7BA.length, 'I2': HOA7BA.I2, 'I3': HOA7BA.I3, 'U0': HOA7BA.U0}
    for k in OPA_data:
        tolerance = abs(OPA_data[k] - ELEGANT_data[k])
        diff = min(abs(ssr_data[k] - OPA_data[k]), abs(ssr_data[k] - ELEGANT_data[k]))
        assert diff < tolerance, f'wrong {k}, {ssr_data[k]}'


def test_twiss(HOA7BA):
    OPA_data = {'I1': 3.0255300E-02, 'I5': 1.1208215E-05}
    ELEGANT_data = {'I1': 3.024838e-02, 'I5': 1.120753e-05}
    AT_Matlab_data = {'I1': 3.02462e-02, 'I5': 1.12079e-05}
    ssr_data = {'I1': HOA7BA.I1, 'I5': HOA7BA.I5}
    for k in OPA_data:
        tolerance = abs(OPA_data[k] - ELEGANT_data[k])
        diff = min(abs(ssr_data[k] - OPA_data[k]), abs(ssr_data[k] - ELEGANT_data[k]))
        assert diff < tolerance, f'wrong {k}, {ssr_data[k]}'


def test_emittance(HOA7BA):
    OPA_data = {'emittance': 0.068362*1000}
    ELEGANT_data = {'emittance': 68.35675}
    AT_Matlab_data = {'emittance': 68.3609}
    ssr_data = {'emittance': HOA7BA.emittance*1e12}
    for k in OPA_data:
        tolerance = abs(OPA_data[k] - ELEGANT_data[k])
        diff = min(abs(ssr_data[k] - OPA_data[k]), abs(ssr_data[k] - ELEGANT_data[k]))
        assert diff < tolerance, f'wrong {k}, {ssr_data[k]}'


def test_edge_angle(HOA7BA):
    OPA_data = {'I4': -5.3351779E-01}
    ELEGANT_data = {'I4': -5.335613e-01}
    ssr_data = {'I4': HOA7BA.I4}
    for k in OPA_data:
        tolerance = abs(OPA_data[k] - ELEGANT_data[k])
        diff = min(abs(ssr_data[k] - OPA_data[k]), abs(ssr_data[k] - ELEGANT_data[k]))
        assert diff < tolerance, f'wrong {k}, {ssr_data[k]}'


def test_chromaticity(HOA7BA):
    ELEGANT_data = {'xi1x': 1.999349e+00, 'xi1y': 2.000172e+00}
    # OPA_data = {'xi1x': 3.249700, 'xi1y': 2.387378}
    AT_Matlab_data = {'xi1x': 2.149416648074739, 'xi1y': 1.933703919321239}
    ssr_data = {'xi1x': HOA7BA.xi_x, 'xi1y': HOA7BA.xi_y}
    # ssr_data = HOA7BA.track_chromaticity()
    for k in AT_Matlab_data:
        tolerance = abs(AT_Matlab_data[k] - ELEGANT_data[k])
        diff = min(abs(ssr_data[k] - AT_Matlab_data[k]), abs(ssr_data[k] - ELEGANT_data[k]))
        assert diff < tolerance, f'wrong {k}, {ssr_data[k]}'
        # tolerance = abs(OPA_data[k] - ELEGANT_data[k])
        # diff = min(abs(ssr_data[k] - OPA_data[k]), abs(ssr_data[k] - ELEGANT_data[k]))
        # assert abs(ssr_data[k] - ELEGANT_data[k]) < 1e-2, f'wrong {k}, {ssr_data}'



