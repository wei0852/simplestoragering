import matplotlib.pyplot as plt
from simplestoragering.objectives import quantify_rdt_fluctuation
import simplestoragering as ssr
import numpy as np
from example_lattice import generate_ring
import PyNAFF as pnf


def track_tunes(ring, xin):
    ring2 = ring * ring.n_periods
    ssr.symplectic_track(xin, ring2, 1027)
    trajectory = ring2.mark['ss'][0].data

    pnf_in = trajectory[:, 0] - np.mean(trajectory[:, 0])
    pnf_out = pnf.naff(pnf_in, turns=1026, )
    try:
        nux = pnf_out[0, 1]
    except Exception as e:
        # print(e)
        nux = np.nan
    pnf_in = trajectory[:, 2] - np.mean(trajectory[:, 2])
    try:
        pnf_out = pnf.naff(pnf_in, turns=1026, )
        nuy = pnf_out[0, 1]
    except Exception:
        nuy = np.nan
    return nux, nuy


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
        
    tracked_dnuxdJx = []
    tracked_dnuydJx = []
    tracked_dnuxdJy = []
    tracked_dnuydJy = []
    
    dnuxdJx = []
    dnuydJx = []
    dnuydJy = []
    
    for dp in np.linspace(-0.06, 0.04, 11):
        ring.off_momentum_optics(delta=dp)
        betax = ring.elements[0].betax
        betay = ring.elements[0].betay
        rdts = ring.driving_terms(verbose=False)
        
        nux0 = ring.nux - int(ring.nux)
        nuy0 = ring.nuy - int(ring.nuy)
        closed_orbit = ring.elements[0].closed_orbit
        dJ = 1e-8
        epsilon = np.array([(2*dJ*betax)**0.5, 0, 1e-9, 0, 0, 0])
        nux_xp, nuy_xp = track_tunes(ring, closed_orbit + epsilon)
        nux_xm, nuy_xm = track_tunes(ring, closed_orbit - epsilon)
    
        nux_x = (nux_xp + nux_xm) / 2
        nuy_x = (nuy_xp + nuy_xm) / 2
        
        tracked_dnuxdJx.append((nux_x - nux0) / dJ)
        tracked_dnuydJx.append((nuy_x - nuy0) / dJ)
    
        epsilon = np.array([1e-9, 0, (2*dJ*betay)**0.5, 0, 0, 0])
        nux_yp, nuy_yp = track_tunes(ring, closed_orbit + epsilon)
        nux_ym, nuy_ym = track_tunes(ring, closed_orbit - epsilon)
    
        nux_y = (nux_yp + nux_ym) / 2
        nuy_y = (nuy_yp + nuy_ym) / 2
        
        tracked_dnuxdJy.append((nux_y - nux0) / dJ)
        tracked_dnuydJy.append((nuy_y - nuy0) / dJ)
        
        adts = rdts.adts()
        
        dnuxdJx.append(adts[0])
        dnuydJx.append(adts[1])
        dnuydJy.append(adts[2])
    
    fig = plt.figure(figsize=(12, 4))
    plt.subplots_adjust(left=0.08, bottom=0.15, right=0.98, top=0.93, wspace=0.3, hspace=0.4)
    plt.subplot(1, 3, 1)
    plt.scatter(np.linspace(-6, 4, 11), tracked_dnuxdJx, label='tracked')
    plt.scatter(np.linspace(-6, 4, 11), dnuxdJx, label='NDT')
    plt.xlabel('$\\delta$ [%]')
    plt.ylabel('dnux/dJx')
    plt.legend()
    plt.subplot(1, 3, 2)
    
    plt.scatter(np.linspace(-6, 4, 11), tracked_dnuydJx, label='tracked')
    plt.scatter(np.linspace(-6, 4, 11), tracked_dnuxdJy, label='tracked')
    plt.scatter(np.linspace(-6, 4, 11), dnuydJx, label='NDT')
    plt.legend()
    
    plt.xlabel('$\\delta$ [%]')
    plt.ylabel('dnux/dJy')
    plt.subplot(1, 3, 3)
    plt.scatter(np.linspace(-6, 4, 11), tracked_dnuydJy, label='tracked')
    plt.scatter(np.linspace(-6, 4, 11), dnuydJy, label='NDT')
    plt.xlabel('$\\delta$ [%]')
    plt.ylabel('dnuy/dJy')
    plt.legend()
    plt.show()
    
    # !!! twiss parameters in the elements were changed by the method off_momentum_optics(),
    # so it's important to re-initialize the data.
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

    ssr.symplectic_track(particle=[1e-3, 0, 0, 0, 0, 0], lattice=ring, n_turns=100, record=True)
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

