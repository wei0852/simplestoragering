import matplotlib.pyplot as plt
from simplestoragering.objectives import quantify_rdt_fluctuation
import simplestoragering as ssr
import numpy as np
from example_lattice import generate_ring


if __name__ == '__main__':
    ring = generate_ring()
    ring.linear_optics()  # the number of periods of cells is 14. calculate the ring data using one cell.
    ssr.plot_lattice(ring, ['betax', 'betay'])
    
    # slice elements to draw detailed curves.
    ele_slices = ring.slice_elements(drift_length=0.1, bend_length=0.01, quad_length=0.01, sext_length=0.01, oct_length=0.01)
    [left, bottom, width, height] = [0.15, 0.15, 0.8, 0.8]
    label_size = 18
    tick_size = 15
    fig = plt.figure(figsize=(6, 4.2))
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.plot(ssr.get_col(ele_slices, 's'), ssr.get_col(ele_slices, '100etax'))
    ax1.set_xlabel('s [m]', fontsize=label_size)
    ax1.set_ylabel('100 * $\\eta_x$', fontsize=label_size)
    ax1.xaxis.set_tick_params(labelsize=tick_size)
    ax1.yaxis.set_tick_params(labelsize=tick_size)
    ssr.plot_layout_in_ax(ring.elements, ax1.twinx())
    plt.show()

    print(ring, '\n')

    print('---------   ring.xi_x and ring.xi_y are calculated using linear pass method. --------------')
    print('--------   use ring.track_chromaticity() to get more accurate chromaticities  -------------')
    
    # calculate chromaticity using tracking, may be different from ring.xi_x and ring.xi_y
    ksi = ring.track_chromaticity(order=4, verbose=True, delta=1e-3)  

    print('\n--------------------    Nonlinear driving terms   ---------------------')
    rdts = ring.driving_terms(verbose=True)
    f_rms = quantify_rdt_fluctuation(rdts, w=1 / 200)
    f3rms, f4rms = quantify_rdt_fluctuation(rdts, w=0)
    ssr.plot_RDTs_along_ring(ring, RDT_type='f')
    ssr.plot_RDTs_along_ring(ring, RDT_type='h')

    print(f'f_rms: {f_rms:.2f}, f3rms: {f3rms:.2f}, f4rms: {f4rms:.2f}')

    print('\n--------------------    ADTS   ---------------------')
    print('use driving terms, ')
    ADTS = rdts.adts()  # calculate ADTS terms using driving terms.
    print(ADTS)

    print('\n------------------  off-momentum ---------------')
    try:
        ring.off_momentum_optics(delta=0.03)
        betax = ring.elements[0].betax
        rdts = ring.driving_terms(verbose=False)
        f_rms = quantify_rdt_fluctuation(rdts, w=1 / 200)
        f3rms, f4rms = quantify_rdt_fluctuation(rdts, w=0)
        print(f'delta=3%: nux={ring.nux:.2f}, nuy={ring.nuy:.2f}\n    betax: {betax}, f_rms: {f_rms:.2f}, f3rms: {f3rms:.2f}, f4rms: {f4rms:.2f}')
        # plt.plot(ssr.get_col(ring.elements, 's'), ssr.get_col(ring.elements, 'closed_orbit_x'))
        # plt.xlabel('s [m]', fontsize=label_size)
        # plt.ylabel('closed-orbit x [m]', fontsize=label_size)
        # ssr.plot_layout_in_ax(ring.elements, plt.gca().twinx())
        # plt.tight_layout()
        # plt.show()
    except ssr.Unstable:
        print('unstable')
    
    # !!! twiss parameters in the elements were changed by the method off_momentum_optics(),
    # so it's importance to re-initialize the data.
    ring.linear_optics()

    print('\n------------------  adjust chromaticity & tunes ---------------')
    k2_list = ssr.chromaticity_correction(ring, sextupole_name_list=['SF3', 'SD3'], target=[3, 3])
    ring.linear_optics()
    print(f'new sf3: {k2_list[0]:.6f}, new sd3: {k2_list[1]:.6f}, ring.xi_x={ring.xi_x:.2f}, ring.xi_y={ring.xi_y:.2f}')
    ring.track_chromaticity()

    k1_list = ssr.adjust_tunes(ring, quadrupole_name_list=['Q1', 'Q2'], target=[43.4, 16.4], iterations=5)
    print(f'new Q1: {k1_list[0]:3f}, new Q2: {k1_list[1]:.3f}', f'nux: {ring.nux:.4f}', f"nuy: {ring.nuy:.4f}")

    print('\n----------------       (4D)  tracking    -----------------')
    # 4D tracking because there is no rf-cavity class in the package.
    ring = generate_ring()
    ring.linear_optics()

    DA1 = ssr.NLine(n_lines=5, xmax=0.02, ymax=0.01, n_points=10)
    DA1.search(ring, n_turns=100)
    plt.plot(DA1.aperture[:, 0] * 1000, DA1.aperture[:, 1] * 1000)
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.show()

    DA2 = ssr.XYGrid(xmax=0.02, nx=20, ymax=0.01, ny=10, delta=0)
    DA2.search(ring, n_turns=100)
    plt.scatter(DA2.data[:, 0] * 1000, DA2.data[:, 1] * 1000, c=DA2.data[:, 2], cmap='binary', marker='s')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.show()

    DA3 = ssr.XDeltaGrid(xmax=0.02, nx=20, delta_max=0.05, ndelta=5)
    DA3.search(ring, n_turns=100)
    plt.scatter(DA3.data[:, 1] * 100, DA3.data[:, 0] * 1000, c=DA3.data[:, 2], cmap='binary', marker='s')
    plt.xlabel('delta [%]')
    plt.ylabel('x [mm]')
    plt.show()

    ssr.symplectic_track(particle=[1e-2, 0, 0, 0, 0, 0], lattice=ring, n_turns=100, record=True)
    #record = true, the coordinates of particle as it passes through the Mark will be recorded.
    #record = false, faster
    plt.subplot(1, 2, 1)
    data = ring.mark['ss'][0].data
    plt.scatter(data[:, 0], data[:, 1])
    plt.title(f's={ring.mark["ss"][0].s}')
    plt.xlabel('x [m]')
    plt.ylabel('px')
    plt.subplot(1, 2, 2)
    data = ring.mark['ss'][1].data
    plt.title(f's={ring.mark["ss"][1].s:.3f}')
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlabel('x [m]')
    plt.ylabel('px')
    plt.show()

    DA3d = ssr.DynamicAperturePolyhedron(0.02, 10, 20, 1, 0.1,
                                         delta_list_m=np.linspace(-0.04, -0.1, 7), delta_list_p=np.linspace(0.02, 0.07, 6))
    DA3d.search(ring)
    fast_ma = DA3d.fast_Touschek_tracking(ring)
    plt.plot(fast_ma[:, 0], fast_ma[:, 1] * 100, color='C0')
    plt.plot(fast_ma[:, 0], fast_ma[:, 2] * 100, color='C0')
    plt.xlabel('s [m]')
    plt.ylabel('$\\delta$ [%]')
    plt.title('fast Touschek tracking')
    ssr.plot_layout_in_ax(ring.elements, plt.gca().twinx())
    plt.show()

    LMA = ssr.LocalMomentumAperture(ring, ds=10.)
    LMA.search(n_turns=100)
    LMA.save(filename='example_4D_LMA_data.csv', header='LMA data tracked by SimpleStorageRing')
    plt.plot(LMA.s, LMA.max_delta * 100, color='C0')
    plt.plot(LMA.s, LMA.min_delta * 100, color='C0')
    plt.xlabel('s [m]')
    plt.ylabel('$\\delta$ [%]')
    ssr.plot_layout_in_ax(ring.elements, plt.gca().twinx())
    plt.show()

