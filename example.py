import simplestoragering as ssr
import numpy as np
import matplotlib.pyplot as plt
import time


def generate_ring():
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

    Q1 = ssr.Quadrupole('Q1', length=0.220000, k1=5.818824)
    Q2 = ssr.Quadrupole('Q2', length=0.220000, k1=-6.088408)
    Q3 = ssr.Quadrupole('Q3', length=0.140000, k1=6.507626)

    B1 = ssr.HBend('B1', length=0.750000, theta=2.294735 * d2r, k1=0.000000, theta_in=1.147368 * d2r,
                   theta_out=1.147368 * d2r)
    B2 = ssr.HBend('B2', length=0.890000, theta=4.790311 * d2r, k1=-1.482985, theta_in=2.395155 * d2r,
                   theta_out=2.395155 * d2r)
    RB = ssr.HBend('RB', length=0.160000, theta=-0.282674 * d2r, k1=6.229233, theta_in=-0.141337 * d2r,
                   theta_out=-0.141337 * d2r)

    sext_slices = 1  # the number of slices affects the results of driving terms and higher-order chromaticities
    SF1 = ssr.Sextupole('SF1', length=0.100000, k2=2 * 98.385000, n_slices=sext_slices)
    SD1 = ssr.Sextupole('SD1', length=0.100000, k2=- 2 * 105.838000, n_slices=sext_slices)
    SD2 = ssr.Sextupole('SD2', length=0.150000, k2=-2 * 209.734000, n_slices=sext_slices)
    SF2 = ssr.Sextupole('SF2', length=0.150000, k2=2 * 328.795000, n_slices=sext_slices)
    SD3 = ssr.Sextupole('SD3', length=0.150000, k2=-2 * 261.435000, n_slices=sext_slices)
    SF3 = ssr.Sextupole('SF3', length=0.150000, k2=2 * 304.099000, n_slices=sext_slices)

    CELLH = [D1A, SF1, D1B, Q1, D2A, SD1, D2B, Q2, D3, B1, D4, SD2, D5, Q3,
             D6, SF2, D7, RB, D8, SD3, D9, B2, D9, SD3, D8, RB, D6, SF3, D7, RB, D8,
             SD3, D9, B2, D9, SD3, D8, RB, D6, SF3, D7, RB, D8, SD3, D9]
    RC = [D9, SD3, D8, RB, D7, SF3, D6, RB, D8, SD3, D9, B2, D9, SD3, D8, RB, D7, SF3, D6, RB, D8, SD3, D9, B2, D9, SD3,
          D8,
          RB, D7, SF2, D6, Q3, D5, SD2, D4, B1, D3, Q2, D2B, SD1, D2A, Q1, D1B, SF1, D1A]
    return ssr.CSLattice(CELLH + [B2] + RC, n_periods=14)


if __name__ == '__main__':
    cell = generate_ring()
    cell.linear_optics()  # the number of periods of cells is 14. calculate the ring data using one cell.
    ssr.plot_lattice(cell, ['betax', 'betay', '100etax'])
    ring = cell * 14
    ring.linear_optics()  # calculate the ring data using all elements of the ring.
    print(ring)

    ring.linear_optics()
    t1 = time.time()
    ring.driving_terms(printout=False)
    t2 = time.time()
    print(f'time = {t2 - t1:.3f} seconds.    Calculate using RDT fluctuations data.')
    # This method using the ELEGANT formula.
    t1 = time.time()
    rdts_another_method = ring.another_method_driving_terms(printout=False)
    t2 = time.time()
    print(f'time = {t2 - t1:.3f} seconds.    Another method calculates one-turn RDTs.')

    # SimpleStorageRing can calculate can calculate multi-period RDTs
    # further reducing computation time.
    # The formula for calculating multi-period RDTs is referenced from [inside_OPA.pdf](https://ados.web.psi.ch/opa).
    t1 = time.time()
    rdts = cell.driving_terms(printout=False)
    rdts.set_periods(n_periods=14)
    t2 = time.time()
    print(f'time = {t2 - t1:.3f} seconds.    Calculate using RDT fluctuations data with one cell.')

    # The ADTSs are driven by h22000 h11110 and h00220
    print(f"\n{'Calculate ADTSs with RDTs':>50}:\n{'dQxx':>20}: {-4 * rdts['h22000'] / np.pi:.0f}, dQxy: {-2 * rdts['h11110'] / np.pi:.0f}, dQyy: {-4 * rdts['h00220'] / np.pi:.0f}")

    # There is another method to calculate ADTS terms
    # which uses the formula in [CERN8805] and [SLS09/97].
    # But this is slower especially when the number of sextupoles or slices of sextupoles is large.
    t1 = time.time()
    adts = ring.adts(n_periods=1, printout=False)
    t2 = time.time()
    print(f'time = {t2 - t1:.3f} seconds.    Calculate ADTSs with [CERN8805] and [SLS09/97] formula.')
    print('                ', end='')
    for k, v in adts.items():
        print(f'{k}: {v:.0f}', end=' ')
    indent = ' ' * 8
    print('.\n', end=indent)
    print('\nThese two methods produce different ADTS terms with some deviation.\n', end=indent)
    print('The second method is more commonly used in other programs,\n', end=indent)
    print('but the former requires much less computation time.\n', end=indent)
    print('Different methods can be selected for different stages of nonlinear optimization.')

    for k in rdts.terms:
        print(f'{k}: {abs(rdts[k]):.2f}, {rdts_another_method[k]:.2f}')

    rdts_fluct = rdts.fluctuation(n_periods=14)  # The fluctuation of RDTs in the complex plane.
    fig = plt.figure(figsize=(10.5, 10))
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95, wspace=0.3)
    for i, k in enumerate(['h21000', 'h30000', 'h10110', 'h10020', 'h10200', 'h20001', 'h00201', 'h10002',
                           'h31000', 'h40000', 'h20110', 'h11200', 'h20020', 'h20200', 'h00310', 'h00400']):
        plt.subplot(4, 4, i + 1)
        plt.scatter(np.real(rdts_fluct[k]), np.imag(rdts_fluct[k]), s=5)
        plt.text(0.99, 0.01, k, transform=plt.gca().transAxes, size=15, horizontalalignment="right")
    plt.suptitle('RDT fluctuations')
    plt.show()

    N = int(len(rdts_fluct['h21000']) / 14)
    N_cell = 100
    multi_cell_fluct = rdts.fluctuation(n_periods=N_cell)  # Here we calculate more cells to show the regularity.
    fig = plt.figure(figsize=(10.5, 5))
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95, wspace=0.3)

    for i in [0, 1, 2]:
        plt.subplot(2, 4, i + 1)
        for k in range(N_cell):
            plt.scatter(np.real(multi_cell_fluct['h21000'][int(k*N+i)]), np.imag(multi_cell_fluct['h21000'][int(k*N+i)]), c='C0', s=5)
            plt.text(0.99, 0.01, f'h21000\n$k$ N + {i+1}', transform=plt.gca().transAxes, size=15, horizontalalignment="right")
        plt.subplot(2, 4, i + 5)
        for k in range(N_cell):
            plt.scatter(np.real(multi_cell_fluct['h31000'][int(k*N+i)]), np.imag(multi_cell_fluct['h31000'][int(k*N+i)]), c='C0', s=5)
            plt.text(0.99, 0.01, f'h31000\n$k$ N + {i+1}', transform=plt.gca().transAxes, size=15, horizontalalignment="right")
    plt.subplot(2, 4, 4)
    for k in range(N_cell):
        plt.scatter(np.real(multi_cell_fluct['h21000'][int(k*N - 1)]), np.imag(multi_cell_fluct['h21000'][int(k*N-1)]), c='C0', s=5)
        plt.text(0.99, 0.01, f'h21000\n$k$ N + N', transform=plt.gca().transAxes, size=15, horizontalalignment="right")
    plt.subplot(2, 4, 8)
    for k in range(N_cell):
        plt.scatter(np.real(multi_cell_fluct['h31000'][int(k*N - 1)]), np.imag(multi_cell_fluct['h31000'][int(k*N-1)]), c='C0', s=5)
        plt.text(0.99, 0.01, f'h31000\n$k$ N + N', transform=plt.gca().transAxes, size=15, horizontalalignment="right")
    plt.suptitle('RDT fluctuations')
    plt.show()

    rdts_plot = ring.driving_terms_plot_data()  # This method calculates the fluctuation of RDTs along the ring.
    fig = plt.figure(figsize=(15, 10))
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95, wspace=0.3)
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    ax11 = ax1.twinx()
    ax22 = ax2.twinx()
    ax33 = ax3.twinx()
    ssr.plot_layout_in_ax(ring.elements, ax11)
    ssr.plot_layout_in_ax(ring.elements, ax22)
    ssr.plot_layout_in_ax(ring.elements, ax33)
    for k in ['h21000', 'h30000', 'h10110', 'h10020', 'h10200']:
        ax1.plot(rdts_plot['s'], np.abs(rdts_plot[k]), label=k)
    ax1.legend()
    for k in ['h20001', 'h00201', 'h10002']:
        ax2.plot(rdts_plot['s'], np.abs(rdts_plot[k]), label=k)
    ax2.legend()
    for k in ['h31000', 'h40000', 'h20110', 'h11200', 'h20020', 'h20200', 'h00310', 'h00400']:
        ax3.plot(rdts_plot['s'], np.abs(rdts_plot[k]), label=k)
    ax3.legend()
    plt.show()

    ring.higher_order_chromaticity()

    fig, ax = plt.subplots(1)
    ax.scatter(ring.nux, ring.nuy)
    ssr.plot_resonance_line_in_ax(ax, order=4, refnux=ring.nux, refnuy=ring.nuy)
    plt.show()
    ssr.plot_lattice(cell.elements, 'etax')
    ele_slices = cell.slice_elements(0.1, 0.1, 0.1, 0.1)  # slice elements to obtain smooth curves.
    ssr.plot_lattice(ele_slices, ['betax', 'betay'])

    # can not set the limit of sextupole strengths.
    sext_k2 = ssr.chromaticity_correction(ring, sextupole_name_list=['SD1', 'SF1', 'SD2', 'SF2'], target=[3, 3])
    print(sext_k2)
    ring.linear_optics()
    print(ring.xi_x, ring.xi_y)
