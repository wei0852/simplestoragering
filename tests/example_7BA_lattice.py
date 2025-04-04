import simplestoragering as ssr
import numpy as np


def generate_ring() -> ssr.CSLattice:
    d2r = np.pi / 180
    ssr.set_ref_energy(2200)
    D1 = ssr.Drift('D1', length=2.650000)
    D1A = ssr.Drift('D1A', length=2.475000)
    D1B = ssr.Drift('D1B', length=0.075000)
    D2 = ssr.Drift('D2', length=0.250000)
    D2A = ssr.Drift('D2A', length=0.075000)
    D2B = ssr.Drift('D2B', length=0.075000)
    D3 = ssr.Drift('D3', length=0.185000)
    D4 = ssr.Drift('D4', length=0.200000)
    D5 = ssr.Drift('D5', length=0.325000)
    D6 = ssr.Drift('D6', length=0.150000)
    D7 = ssr.Drift('D7', length=0.150000)
    D8 = ssr.Drift('D8', length=0.207000)
    D9 = ssr.Drift('D9', length=0.150000)

    Q1 = ssr.Quadrupole('Q1', length=0.220000, k1=5.818824, n_slices=22)
    Q2 = ssr.Quadrupole('Q2', length=0.220000, k1=-6.088408, n_slices=22)
    Q3 = ssr.Quadrupole('Q3', length=0.140000, k1=6.507626, n_slices=14)

    B1 = ssr.HBend('B1', length=0.750000, theta=2.294735 * d2r, k1=0.000000, theta_in=1.147368 * d2r,
                   theta_out=1.147368 * d2r, n_slices=75, edge_method=1)
    B2 = ssr.HBend('B2', length=0.890000, theta=4.790311 * d2r, k1=-1.482985, theta_in=2.395155 * d2r,
                   theta_out=2.395155 * d2r, n_slices=89, edge_method=1)
    RB = ssr.HBend('RB', length=0.160000, theta=-0.282674 * d2r, k1=6.229233, theta_in=-0.141337 * d2r,
                   theta_out=-0.141337 * d2r, n_slices=16, edge_method=1)

    SF1 = ssr.Sextupole('SF1', length=0.100000, k2=2 * 98.385000, n_slices=10)
    SD1 = ssr.Sextupole('SD1', length=0.100000, k2=- 2 * 105.838000, n_slices=10)
    SD2 = ssr.Sextupole('SD2', length=0.150000, k2=-2 * 209.734000, n_slices=15)
    SF2 = ssr.Sextupole('SF2', length=0.150000, k2=2 * 328.795000, n_slices=15)
    SD3 = ssr.Sextupole('SD3', length=0.150000, k2=-2 * 261.435000, n_slices=15)
    SF3 = ssr.Sextupole('SF3', length=0.150000, k2=2 * 304.099000, n_slices=15)

    ss = ssr.Mark('ss')  # straight section

    CELLH = [ss, D1A, SF1, D1B, Q1, D2A, SD1, D2B, Q2, D3, ss, B1, D4, SD2, D5, Q3,
             D6, SF2, D7, RB, D8, SD3, D9, B2, D9, SD3, D8, RB, D6, SF3, D7, RB, D8,
             SD3, D9, B2, D9, SD3, D8, RB, D6, SF3, D7, RB, D8, SD3, D9]
    RC = [D9, SD3, D8, RB, D7, SF3, D6, RB, D8, SD3, D9, B2, D9, SD3, D8, RB, D7, SF3, D6, RB, D8, SD3, D9, B2, D9, SD3,
          D8,
          RB, D7, SF2, D6, Q3, D5, SD2, D4, B1, D3, Q2, D2B, SD1, D2A, Q1, D1B, SF1, D1A]
    return ssr.CSLattice(CELLH + [B2] + RC, n_periods=14)
