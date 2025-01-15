# -*- coding: utf-8 -*-
"""
There are some functions to visualize lattice data conveniently.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from .Sextupole import Sextupole
from .Quadrupole import Quadrupole
from .HBend import HBend
from .Octupole import Octupole
from .components import Element
from .CSLattice import CSLattice
from .DrivingTerms import compute_driving_terms
import numpy as np


def get_col(ele_list, parameter: str) -> np.ndarray:
    """Get parameter in each element of ele_list, return a numpy array."""

    # Mapping of parameter names to their corresponding attributes or computations
    param_map = {
        's': lambda ele: ele.s,
        'closed_orbit_delta': lambda ele: ele.closed_orbit[5],
        'closed_orbit_x': lambda ele: ele.closed_orbit[0],
        'closed_orbit_y': lambda ele: ele.closed_orbit[2],
        'closed_orbit_px': lambda ele: ele.closed_orbit[1],
        'closed_orbit_py': lambda ele: ele.closed_orbit[3],
        'closed_orbit_z': lambda ele: ele.closed_orbit[4],
        'betax': lambda ele: ele.betax,
        'betay': lambda ele: ele.betay,
        'alphax': lambda ele: ele.alphax,
        'alphay': lambda ele: ele.alphay,
        'etax': lambda ele: ele.etax,
        'etaxp': lambda ele: ele.etaxp,
        'psix': lambda ele: ele.psix,
        'psiy': lambda ele: ele.psiy,
        '100etax': lambda ele: 100 * ele.etax,
        'length': lambda ele: ele.length,
    }

    # Check if the parameter is valid
    if parameter not in param_map:
        raise ValueError(f"Cannot get {parameter} data. Parameter not recognized.")

    # Use the mapping to extract the data
    col_data = [param_map[parameter](ele) for ele in ele_list]

    return np.array(col_data)


def get_layout(ele_list):
    """get layout data of magnets"""
    s = []
    current_s = 0
    magnet = []
    s.append(current_s)
    for ele in ele_list:
        current_s = current_s + ele.length
        s.append(current_s)
        s.append(current_s)
        magnet.append(ele.type())
        magnet.append(ele.type())
    del s[-1]
    return s, magnet


def plot_lattice(lattice, parameters, with_layout=True) -> None:
    """Plot data along lattice.
    plot_lattice(lattice, parameters='betax', with_layout=True),
    plot_lattice(lattice, parameters=['betax', 'betay'], with_layout=True)

    parameters:
        's',
        'betax',
        'betay',
        'etax',
        '100etax',
        'psix',
        'psiy',
        'closed_orbit_x',
        'closed_orbit_px',
        'closed_orbit_y',
        'closed_orbit_py',
        'closed_orbit_delta',
        'etaxp'
        'alphax',
        'alphay'.
    """
    if isinstance(lattice, CSLattice):
        ele_list = lattice.elements
    elif isinstance(lattice, list) and all(isinstance(i, Element) for i in lattice):
        ele_list = lattice
    else:
        raise Exception('lattice should be CSLattice or list[Element].')
    if with_layout:
        return plot_with_layout(ele_list, parameters)
    else:
        return plot_without_layout(ele_list, parameters)


def plot_without_layout(ele_list, parameters):
    s = get_col(ele_list, 's')
    if isinstance(parameters, list):
        for para in parameters:
            plt.plot(s, get_col(ele_list, para), label=para)
    elif isinstance(parameters, str):
        plt.plot(s, get_col(ele_list, parameters), label=parameters)
    else:
        raise Exception('plot error')
    plt.legend()
    plt.show()


def plot_with_layout(ele_list, parameters):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    s = get_col(ele_list, 's')
    if isinstance(parameters, list):
        for para in parameters:
            ax2.plot(s, get_col(ele_list, para), label=para)
    elif isinstance(parameters, str):
        ax2.plot(s, get_col(ele_list, parameters), label=parameters)
    else:
        raise Exception('plot error')
    plot_layout_in_ax(ele_list, ax1)
    ax1.set_xlim(s[0], s[-1])
    ax2.yaxis.set_ticks_position('left')
    plt.legend()
    plt.show()


def plot_layout_in_ax(ele_list: list, ax: Axes, ratio=0.03) -> None:
    """Draw the layout in matplotlib.axes.Axes and the layout will appear at the bottom of the figure. ratio controls the height."""
    current_s = 0
    i = 0
    while i < len(ele_list):
        if isinstance(ele_list[i], Quadrupole):
            begin_s = current_s
            layout_s = [current_s, current_s]
            current_s += ele_list[i].length
            layout_s += [(begin_s + current_s) / 2, current_s, current_s]
            draw_data = ele_list[i].k1 / abs(ele_list[i].k1) if ele_list[i].k1 != 0 else 0
            layout_data = [0, 1, 1 + 0.3 * draw_data, 1, 0]
            ax.fill(layout_s, layout_data, color='#EE3030')
        elif isinstance(ele_list[i], HBend):
            layout_s = [current_s, current_s]
            current_s += ele_list[i].length
            layout_s += [current_s, current_s]
            layout_data = [0, 1, 1, 0]
            if ele_list[i].h >= 0:
                ax.fill(layout_s, layout_data, color='#3d3dcd')
            else:
                ax.fill(layout_s, layout_data, color='#00eeee')
        elif isinstance(ele_list[i], Sextupole):
            begin_s = current_s
            layout_s = [current_s, current_s]
            current_s += ele_list[i].length
            layout_s += [(begin_s + current_s) / 2, current_s, current_s]
            draw_data = ele_list[i].k2 / abs(ele_list[i].k2) if ele_list[i].k2 != 0 else 0
            layout_data = [0, 1, 1 + 0.3 * draw_data, 1, 0]
            ax.fill(layout_s, layout_data, color='#3dcd3d')
        elif isinstance(ele_list[i], Octupole):
            layout_s = [current_s, current_s]
            current_s += ele_list[i].length
            layout_s += [current_s, current_s]
            layout_data = [0, 1, 1, 0]
            ax.fill(layout_s, layout_data, color='#8B3A3A')
        else:
            current_s += ele_list[i].length
        i += 1
    ax.set_ylim(0, 1/ratio)
    ax.set_yticks([])


def plot_with_background(s, y: dict, lattice):
    """plot with layout background."""
    layout_s, layout_data = get_layout(lattice)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    for k, v in y.items():
        ax2.plot(s, v, label=k)
    plt.legend()
    ax1.fill(layout_s, layout_data, color='#cccccc')
    plt.show()


def plot_resonance_line_in_ax(ax: Axes, order: int = 3, refnux: float=None, refnuy: float=None):
    """plot the resonance lines in matplotlib.axes.Axes, the plot area is [int(refnux), 1 + int(refnux)] x [int(refnuy), 1 + int(refnuy)]"""

    third_color = '#808080'  #
    fourth_color = '#a0a0a0'  #
    fifth_color = '#dddddd'  #
    third_width = 1.8
    fourth_width = 1
    fifth_width = 0.8
    Qx_int = 0 if refnux is None else int(refnux)
    Qy_int = 0 if refnuy is None else int(refnuy)
    order = 3 if order < 3 else order
    line1, = ax.plot([Qx_int + 1, Qx_int + 1], [Qy_int, Qy_int + 1], c='#000000', linewidth=3, label='1')
    ax.plot([Qx_int, Qx_int], [Qy_int, Qy_int + 1], c='#000000', linewidth=6)
    ax.plot([Qx_int, Qx_int + 1], [Qy_int, Qy_int], c='#000000', linewidth=6)
    ax.plot([Qx_int, Qx_int + 1], [Qy_int + 1, Qy_int + 1], c='#000000', linewidth=6)
    # 2nd black
    line2, = ax.plot([0.5 + Qx_int, 0.5 + Qx_int], [0. + Qy_int, 1 + Qy_int], c='#202020', linewidth=1.8, label='2')  # 2 nuy
    ax.plot([0. + Qx_int, 1 + Qx_int], [0.5 + Qy_int, 0.5 + Qy_int], c='#202020', linewidth=1.8)  # 2 nux
    # 3rd red
    line3, = ax.plot([0. + Qx_int, 1.0 + Qx_int], [0.5 + Qy_int, 1.0 + Qy_int], c=third_color, linewidth=third_width, label='3')  # nux - 2 nuy
    ax.plot([2 / 3 + Qx_int, 2 / 3 + Qx_int], [0 + Qy_int, 1 + Qy_int], c=third_color, linewidth=third_width)  # 3 nux
    ax.plot([1 / 3 + Qx_int, 1 / 3 + Qx_int], [0 + Qy_int, 1 + Qy_int], c=third_color, linewidth=third_width)  # 3 nuy
    ax.plot([0. + Qx_int, 1.0 + Qx_int], [1 + Qy_int, 0.5 + Qy_int], c=third_color, linewidth=third_width)  # nux + 2 nuy
    ax.plot([0. + Qx_int, 1.0 + Qx_int], [0. + Qy_int, 0.5 + Qy_int], c=third_color, linewidth=third_width)  # nux - 2 nuy
    ax.plot([0. + Qx_int, 1.0 + Qx_int], [0.5 + Qy_int, 0.0 + Qy_int], c=third_color, linewidth=third_width)  # nux + 2 nuy
    lines = [line1, line2, line3]
    if order >= 4:
        # 4th
        line4, = ax.plot([0.5 + Qx_int, 1.0 + Qx_int], [1 + Qy_int, 0.50 + Qy_int], c=fourth_color, linewidth=fourth_width, label='4')  # 2 nux + 2 nuy
        lines.append(line4)
        ax.plot([0.   + Qx_int, 1.0 + Qx_int], [0 + Qy_int, 1 + Qy_int], c=fourth_color, linewidth=fourth_width)  # 2 nux - 2 nuy
        ax.plot([0.   + Qx_int, 1.0 + Qx_int], [1 + Qy_int, 0.0 + Qy_int], c=fourth_color, linewidth=fourth_width)  # 2 nux + 2 nuy
        ax.plot([0.5  + Qx_int, 1.0 + Qx_int], [0 + Qy_int, 0.50 + Qy_int], c=fourth_color, linewidth=fourth_width)  # 2 nux - 2 nuy
        ax.plot([0.   + Qx_int, 0.5 + Qx_int], [0.5 + Qy_int, 0.0 + Qy_int], c=fourth_color, linewidth=fourth_width)  # 2 nux + 2 nuy
        ax.plot([0.   + Qx_int, 0.5 + Qx_int], [0.5 + Qy_int, 1.0 + Qy_int], c=fourth_color, linewidth=fourth_width)  # 2 nux - 2 nuy
        ax.plot([0.   + Qx_int, 1   + Qx_int], [0.25 + Qy_int, 0.250 + Qy_int], c=fourth_color, linewidth=fourth_width)
        ax.plot([0.   + Qx_int, 1   + Qx_int], [0.75 + Qy_int, 0.750 + Qy_int], c=fourth_color, linewidth=fourth_width)
        ax.plot([0.25 + Qx_int, 0.25 + Qx_int], [0. + Qy_int, 1.0 + Qy_int], c=fourth_color, linewidth=fourth_width)
        ax.plot([0.75 + Qx_int, 0.75 + Qx_int], [0. + Qy_int, 1.0 + Qy_int], c=fourth_color, linewidth=fourth_width)
    if order >= 5:
        # 5th
        line5, = ax.plot([0 + Qx_int, 2 / 3 + Qx_int], [0 + Qy_int, 1 + Qy_int], c=fifth_color, linewidth=fifth_width, label='5')  # 2 nux - 3 nuy
        lines.append(line5)
        ax.plot([0 + Qx_int, 1 + Qx_int], [0 + Qy_int, 0.25 + Qy_int], c=fifth_color, linewidth=fifth_width)  # nux - 4 nuy
        ax.plot([0 + Qx_int, 1 + Qx_int], [0.25 + Qy_int, 0 + Qy_int], c=fifth_color, linewidth=fifth_width)  # nux + 4 nuy
        ax.plot([0 + Qx_int, 1 / 3 + Qx_int], [0.5 + Qy_int, 0 + Qy_int], c=fifth_color, linewidth=fifth_width)  #
        ax.plot([0 + Qx_int, 1 + Qx_int], [0.5 + Qy_int, 0.25 + Qy_int], c=fifth_color, linewidth=fifth_width)  #
        ax.plot([0 + Qx_int, 1 / 3 + Qx_int], [0.5 + Qy_int, 1 + Qy_int], c=fifth_color, linewidth=fifth_width)
        ax.plot([0 + Qx_int, 1 + Qx_int], [0.5 + Qy_int, 0.75 + Qy_int], c=fifth_color, linewidth=fifth_width)  #
        ax.plot([0 + Qx_int, 1 + Qx_int], [0.25 + Qy_int, 0.5 + Qy_int], c=fifth_color, linewidth=fifth_width)  #
        ax.plot([0 + Qx_int, 1 + Qx_int], [0.75 + Qy_int, 0.5 + Qy_int], c=fifth_color, linewidth=fifth_width)  #
        ax.plot([0 + Qx_int, 1 + Qx_int], [0.75 + Qy_int, 1 + Qy_int], c=fifth_color, linewidth=fifth_width)  #
        ax.plot([0 + Qx_int, 1 + Qx_int], [1 + Qy_int, 0.75 + Qy_int], c=fifth_color, linewidth=fifth_width)  #
        ax.plot([0.2 + Qx_int, 0.2 + Qx_int], [0. + Qy_int, 1.0 + Qy_int], c=fifth_color, linewidth=fifth_width)
        ax.plot([0.4 + Qx_int, 0.4 + Qx_int], [0. + Qy_int, 1.0 + Qy_int], c=fifth_color, linewidth=fifth_width)
        ax.plot([0.6 + Qx_int, 0.6 + Qx_int], [0. + Qy_int, 1.0 + Qy_int], c=fifth_color, linewidth=fifth_width)
        ax.plot([0.8 + Qx_int, 0.8 + Qx_int], [0. + Qy_int, 1.0 + Qy_int], c=fifth_color, linewidth=fifth_width)
        ax.plot([1 / 3 + Qx_int, 1 + Qx_int], [0. + Qy_int, 1.0 + Qy_int], c=fifth_color, linewidth=fifth_width)
        ax.plot([0. + Qx_int, 2 / 3 + Qx_int], [1. + Qy_int, .0 + Qy_int], c=fifth_color, linewidth=fifth_width)
        ax.plot([2 / 3 + Qx_int, 1 + Qx_int], [0. + Qy_int, 0.50 + Qy_int], c=fifth_color, linewidth=fifth_width)
        ax.plot([1 / 3 + Qx_int, 1 + Qx_int], [1. + Qy_int, .0 + Qy_int], c=fifth_color, linewidth=fifth_width)
        ax.plot([2 / 3 + Qx_int, 1 + Qx_int], [1. + Qy_int, 0.50 + Qy_int], c=fifth_color, linewidth=fifth_width)
    ax.set_xlim(Qx_int, Qx_int + 1.2)
    ax.set_xticks([Qx_int, Qx_int + 0.2, Qx_int + 0.4, Qx_int + 0.6, Qx_int + 0.8, Qx_int + 1])
    ax.set_ylim(Qy_int, Qy_int + 1)
    line_legend = ax.legend(handles=lines, loc='lower right')
    ax.add_artist(line_legend)


def plot_RDTs_along_ring(lattice: CSLattice, RDT_type='f', show=True) -> dict[np.ndarray]:
    """plot RDT fluctuation along the ring.

    RDT_type: 'f' for natural fluctuation and RDT f_jklm;
              'h' for buildup fluctuation and RDT h_jklm.
    if not show, return the plotting data.
    """
    assert RDT_type == 'f' or RDT_type == 'h', 'RDT_type should be "f" or "h".'
    label_size = 15
    tick_size = 12
    num_elements = len(lattice.elements)
    betax = []
    betay = []
    psix = []
    psiy = []
    k2l = []
    k3l = []
    s = []
    for i, ele in enumerate(lattice.elements):
        if ele.k2 or ele.k3:
            ele_slices = ele.slice(4)
        else:
            ele_slices = [ele]
        for elele in ele_slices:
            betax.append(elele.betax)
            betay.append(elele.betay)
            psix.append(elele.psix)
            psiy.append(elele.psiy)
            k2l.append(elele.length * (elele.k2 + elele.k3 * elele.closed_orbit[0]) / (1 + lattice.delta))
            k3l.append(elele.k3 * elele.length / (1 + lattice.delta))
            s.append(elele.s)
    betax = np.array(betax)
    betay = np.array(betay)
    psix  = np.array(psix )
    psiy  = np.array(psiy )
    period_phix = psix[-1]
    period_phiy = psiy[-1]
    betax = (betax[1:] + betax[:-1]) / 2
    betay = (betay[1:] + betay[:-1]) / 2
    psix  = ( psix[1:]  + psix[:-1]) / 2
    psiy  = ( psiy[1:]  + psiy[:-1]) / 2
    rdts = compute_driving_terms(betax, betay, psix, psiy, k2l[:-1], k3l[:-1], period_phix, period_phiy, verbose=False)
    if RDT_type == 'f':
        plot_data = rdts.natural_fluctuation()
        for k in plot_data:
            plot_data[k] = np.hstack((plot_data[k][-1], plot_data[k]))
        plot_data['s'] = s
    elif RDT_type == 'h':
        plot_data = rdts.buildup_fluctuation()
        for k in plot_data:
            plot_data[k] = np.hstack((0, plot_data[k]))
        plot_data['s'] = s
    if not show:
        return plot_data
    fig = plt.figure(figsize=(7, 8))
    ax1 = fig.add_axes([0.1, 0.55, 0.85, 0.4])
    ax2 = fig.add_axes([0.1, 0.07, 0.85, 0.4])
    ax11 = ax1.twinx()
    ax22 = ax2.twinx()
    plot_layout_in_ax(lattice.elements, ax11)
    plot_layout_in_ax(lattice.elements, ax22)
    line1, = ax1.plot(plot_data['s'], np.abs(plot_data[f'{RDT_type}2100']), label=f'{RDT_type}2100')
    line2, = ax1.plot(plot_data['s'], np.abs(plot_data[f'{RDT_type}3000']), label=f'{RDT_type}3000')
    line3, = ax1.plot(plot_data['s'], np.abs(plot_data[f'{RDT_type}1011']), label=f'{RDT_type}1011')
    line4, = ax1.plot(plot_data['s'], np.abs(plot_data[f'{RDT_type}1002']), label=f'{RDT_type}1002')
    line5, = ax1.plot(plot_data['s'], np.abs(plot_data[f'{RDT_type}1020']), label=f'{RDT_type}1020')
    ax1.legend(handles=[line1, line2, line3, line4, line5], fontsize=tick_size)
    line9, =  ax2.plot(plot_data['s'], np.abs(plot_data[f'{RDT_type}3100']), label=f'{RDT_type}3100')
    line10, = ax2.plot(plot_data['s'], np.abs(plot_data[f'{RDT_type}4000']), label=f'{RDT_type}4000')
    line11, = ax2.plot(plot_data['s'], np.abs(plot_data[f'{RDT_type}2011']), label=f'{RDT_type}2011')
    line12, = ax2.plot(plot_data['s'], np.abs(plot_data[f'{RDT_type}1120']), label=f'{RDT_type}1120')
    line13, = ax2.plot(plot_data['s'], np.abs(plot_data[f'{RDT_type}2002']), label=f'{RDT_type}2002')
    line14, = ax2.plot(plot_data['s'], np.abs(plot_data[f'{RDT_type}2020']), label=f'{RDT_type}2020')
    line15, = ax2.plot(plot_data['s'], np.abs(plot_data[f'{RDT_type}0031']), label=f'{RDT_type}0031')
    line16, = ax2.plot(plot_data['s'], np.abs(plot_data[f'{RDT_type}0040']), label=f'{RDT_type}0040')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax1.set_xlim(0, plot_data['s'][-1])
    ax2.set_xlim(0, plot_data['s'][-1])
    ax1.set_xlabel('s [m]', fontsize=label_size)
    ax2.set_xlabel('s [m]', fontsize=label_size)
    ax1.xaxis.set_tick_params(labelsize=tick_size)
    ax1.yaxis.set_tick_params(labelsize=tick_size)   
    ax2.xaxis.set_tick_params(labelsize=tick_size)
    ax2.yaxis.set_tick_params(labelsize=tick_size)   
    ax2.legend(handles=[line9, line10, line11, line12, line13, line14, line15, line16], fontsize=tick_size, ncol=2)
    plt.show()
    return plot_data


def template():
    """two templates for using pyplot."""
    x = [1] * 5
    y = [1, 2, 3, 4, 5]
    label_size = 18
    tick_size = 15
    [left, bottom, width, height] = [0.16, 0.15, 0.65, 0.8]

    fig = plt.figure(figsize=(6, 4.2))
    ax1 = fig.add_axes([left, bottom, width, height])
    cmp = ax1.scatter(x, y, c=y, cmap='jet', marker='s', s=10)
    ax1.set_ylabel('y', fontsize=label_size)
    ax1.set_xlabel('x', fontsize=label_size)
    ax1.set_xticks([0, 1, 2])
    ax1.xaxis.set_tick_params(labelsize=tick_size)
    ax1.yaxis.set_tick_params(labelsize=tick_size)
    cb = fig.colorbar(cmp, cax=fig.add_axes([left + width + 0.01, bottom, 0.02, height]))
    cb.set_label('color', fontsize=label_size)
    cb.ax.tick_params(labelsize=tick_size, direction='in')
    cb.ax.set_yticks([1, 3, 5])
    cb.ax.set_yticklabels(['$1^2$', '3', '$\\geq$5'])
    plt.show()

    [left, bottom, right, top] = [0.16, 0.1, 0.8, 0.9]
    plt.subplots(figsize=(6, 8))
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=0.25, hspace=0.4)
    plt.suptitle(f'template', fontsize=label_size)
    plt.figtext(0.02, 0.5, 'vertical text', fontsize=label_size, verticalalignment='center', rotation='vertical')
    plt.subplot(2, 1, 1)
    cmp = plt.scatter(x, y, c=y, cmap='jet', marker='*', s=10)
    plt.ylim(0, )
    plt.xlabel('x [mm]', fontsize=tick_size)
    plt.ylabel('y [mm]', fontsize=tick_size)
    plt.text(
        0.2,
        0.8,
        f'Fig. A',
        transform=plt.gca().transAxes,
        size=tick_size,
        horizontalalignment="left",
    )
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=tick_size)
    ax.yaxis.set_tick_params(labelsize=tick_size)
    cb1 = plt.colorbar(cmp)
    cb1.set_label('color', fontsize=label_size)
    cb1.ax.tick_params(labelsize=tick_size, direction='in')
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(0, 1, 11), np.linspace(0, 1, 11) ** 2, label='$x^2$')
    plt.grid('on')
    plt.xlabel('x', fontsize=tick_size)
    plt.ylabel('y', fontsize=tick_size)
    plt.legend(fontsize=tick_size)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=tick_size)
    ax.yaxis.set_tick_params(labelsize=tick_size)
    cb2 = plt.colorbar(cmp, cax=plt.axes([right + 0.05, bottom, 0.02, top - bottom]))
    cb2.set_label('color', fontsize=label_size)
    cb2.ax.tick_params(labelsize=tick_size, direction='in')
    plt.show()
