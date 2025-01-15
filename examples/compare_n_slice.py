import time

import numpy as np
import matplotlib.pyplot as plt
from example_features import generate_ring
import simplestoragering as ssr
import PyNAFF as pnf


def Bend_slices(length, ring):
    for ele in ring.elements:
        if isinstance(ele, ssr.HBend):
            ele.n_slices = max(int(ele.length / length), 1)
        elif isinstance(ele, ssr.Quadrupole) or isinstance(ele, ssr.Sextupole) or isinstance(ele, ssr.Octupole):
            ele.n_slices = int(ele.length / 0.01)
    ring.linear_optics()
    return ring


def Quad_slices(length, ring):
    for ele in ring.elements:
        if isinstance(ele, ssr.Quadrupole):
            ele.n_slices = max(int(ele.length / length), 1)
        elif isinstance(ele, ssr.HBend):
            ele.n_slices = max(int(ele.length / 0.01), 1)
        elif isinstance(ele, ssr.Sextupole) or isinstance(ele, ssr.Octupole):
            ele.n_slices = int(ele.length / 0.01)
    ring.linear_optics()
    return ring


def Sext_slices(length, ring):
    for ele in ring.elements:
        if isinstance(ele, ssr.Sextupole):
            ele.n_slices = max(int(ele.length / length), 1)
        elif isinstance(ele, ssr.HBend):
            ele.n_slices = max(int(ele.length / 0.01), 1)
        elif isinstance(ele, ssr.Quadrupole) or isinstance(ele, ssr.Octupole):
            ele.n_slices = int(ele.length / 0.01)
    ring.linear_optics()
    return ring


def frequency_analysis(trajectory):
    pnf_in = trajectory[:, 0] - np.average(trajectory[:, 0])
    pnf_out = pnf.naff(pnf_in, turns=516, )
    try:
        nux = pnf_out[0, 1]
    except Exception as e:
        print(e)
        nux = np.nan
    pnf_in = trajectory[:, 2] - np.average(trajectory[:, 2])
    try:
        pnf_out = pnf.naff(pnf_in, turns=516, )
        nuy = pnf_out[0, 1]
    except Exception:
        nuy = np.nan
    return nux, nuy


def tune_shift(ring, param_list, param_idx):
    nux_list = np.zeros(len(param_list))
    nuy_list = np.zeros(len(param_list))
    time_list = []
    for i, param in enumerate(param_list):
        particle = [1e-6, 0, 1e-6, 0, 0, 0]
        particle[param_idx] = param
        try:
            t1 = time.time()
            ssr.symplectic_track(particle, ring, n_turns=517, record=True)
            #record = true, the coordinates of particle as it passes through the Mark will be recorded.
            #record = false, faster
            t2 = time.time()
            time_list.append(t2 - t1)
            nux, nuy = frequency_analysis(ring.mark['ss'][0].data)
            nux_list[i] = nux
            nuy_list[i] = nuy
        except ssr.ParticleLost:
            nux_list[i] = np.nan
            nuy_list[i] = np.nan
    print(np.average(time_list))
    return [nux_list, nuy_list]

def tune_shift_with_momentum(ring) -> list:
    delta_list = np.arange(-0.08, 0.08, 0.005)
    [nux_list, nuy_list] = tune_shift(ring, delta_list, 5)
    return [delta_list * 100, nux_list, nuy_list, '$\\delta$ [%]']

def tune_shift_with_x(ring) -> list:
    x_list = np.arange(-0.02, 0.02, 0.002)
    [nux_list, nuy_list] = tune_shift(ring, x_list, 0)
    return [x_list * 1000, nux_list, nuy_list, 'x [mm]']

def tune_shift_with_y(ring) -> list:
    y_list = np.arange(1e-6, 0.01, 0.001)
    [nux_list, nuy_list] = tune_shift(ring, y_list, 2)
    return [y_list * 1000, nux_list, nuy_list, 'y [mm]']


def off_momentum_tunes(ring) -> list:
    delta_list = np.linspace(-0.1, 0.1, 21)
    nux_list = np.zeros(len(delta_list))
    nuy_list = np.zeros(len(delta_list))
    time_list = []
    for i, dp in enumerate(delta_list):
        try:
            t1 = time.time()
            ring.off_momentum_optics(delta=dp)
            t2 = time.time()
            time_list.append(t2 - t1)
            nux_list[i] = ring.nux
            nuy_list[i] = ring.nuy
        except ssr.Unstable:
            nux_list[i] = np.nan
            nuy_list[i] = np.nan
    print(np.average(time_list))
    return [delta_list * 100, nux_list, nuy_list, '$\\delta$ [%]']


def compare_tune_shift():

    slice_length = [10, 0.1, 0.01, 0.001]  # 10: Only 1 slice, using tracking method in SAMM. But cannot handle k2, k3.
    x_lists = []
    nux_lists = []
    nuy_lists = []
    for le in slice_length:
        [x_list, nux_list, nuy_list, x_label] = off_momentum_tunes(Bend_slices(le, generate_ring())); print(f'done {le}')
        x_lists.append(x_list)
        nux_lists.append(nux_list)
        nuy_lists.append(nuy_list)

    plot_scatter(x_lists, nux_lists, 'nux', x_label, slice_length)
    plot_scatter(x_lists, nuy_lists, 'nuy', x_label, slice_length)
    plot_scatter(x_lists, nux_lists, 'nux differences', x_label, slice_length, differences=True)
    plot_scatter(x_lists, nuy_lists, 'nuy differences', x_label, slice_length, differences=True)


def plot_scatter(x_data, y_data, y_label, x_label, slice_length, differences=False):
    label_size = 18
    tick_size = 15
    [left, bottom, width, height] = [0.17, 0.18, 0.8, 0.75]
    fig = plt.figure(figsize=(6, 4.8))
    ax1 = fig.add_axes([left, bottom, width, height])
    for i in range(len(slice_length) - (1 if differences else 0)):
        y = y_data[i] - y_data[-1] if differences else y_data[i]
        ax1.scatter(x_data[-1] if differences else x_data[i], y, label=f'slice = {slice_length[i]} m')
    ax1.set_ylabel(y_label, fontsize=label_size)
    ax1.set_xlabel(x_label, fontsize=label_size)
    plt.legend(fontsize=tick_size)
    ax1.grid('on')
    ax1.xaxis.set_tick_params(labelsize=tick_size)
    ax1.yaxis.set_tick_params(labelsize=tick_size)
    plt.show()


if __name__ == '__main__':
    compare_tune_shift()
