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
from .exceptions import UnfinishedWork
from .components import Element
from .CSLattice import CSLattice
import numpy as np


def get_col(ele_list, parameter: str) -> np.ndarray:
    """get parameter in each element of ele_list, return a np.ndarray"""
    def __get_closed_orbit_delta():
        __col = []
        for ele in ele_list:
            __col.append(ele.closed_orbit[5])
        return __col

    def __get_closed_orbit_z():
        __col = []
        for ele in ele_list:
            __col.append(ele.closed_orbit[4])
        return __col

    def __get_closed_orbit_x():
        __col = []
        for ele in ele_list:
            __col.append(ele.closed_orbit[0])
        return __col

    def __get_closed_orbit_y():
        __col = []
        for ele in ele_list:
            __col.append(ele.closed_orbit[2])
        return __col

    def __get_closed_orbit_px():
        __col = []
        for ele in ele_list:
            __col.append(ele.closed_orbit[1])
        return __col

    def __get_closed_orbit_py():
        __col = []
        for ele in ele_list:
            __col.append(ele.closed_orbit[3])
        return __col

    def __get_s():
        __col = []
        for ele in ele_list:
            __col.append(ele.s)
        return __col

    def __get_betax():
        __col = []
        for ele in ele_list:
            __col.append(ele.betax)
        return __col

    def __get_betay():
        __col = []
        for ele in ele_list:
            __col.append(ele.betay)
        return __col

    def __get_alphax():
        __col = []
        for ele in ele_list:
            __col.append(ele.alphax)
        return __col

    def __get_alphay():
        __col = []
        for ele in ele_list:
            __col.append(ele.alphay)
        return __col

    def __get_etax():
        __col = []
        for ele in ele_list:
            __col.append(ele.etax)
        return __col

    def __get_etaxp():
        __col = []
        for ele in ele_list:
            __col.append(ele.etaxp)
        return __col

    def __get_psix():
        __col = []
        for ele in ele_list:
            __col.append(ele.psix)
        return __col

    def __get_psiy():
        __col = []
        for ele in ele_list:
            __col.append(ele.psiy)
        return __col

    # assert isinstance(lattice, CSLattice) or isinstance(lattice, SlimRing)
    if parameter == 's':
        col_data = __get_s()
    elif parameter == 'closed_orbit_delta':
        col_data = __get_closed_orbit_delta()
    elif parameter == 'closed_orbit_x':
        col_data = __get_closed_orbit_x()
    elif parameter == 'closed_orbit_y':
        col_data = __get_closed_orbit_y()
    elif parameter == 'closed_orbit_px':
        col_data = __get_closed_orbit_px()
    elif parameter == 'closed_orbit_py':
        col_data = __get_closed_orbit_py()
    elif parameter == 'closed_orbit_z':
        col_data = __get_closed_orbit_z()
    elif parameter == 'betax':
        col_data = __get_betax()
    elif parameter == 'betay':
        col_data = __get_betay()
    elif parameter == 'alphax':
        col_data = __get_alphax()
    elif parameter == 'alphay':
        col_data = __get_alphay()
    elif parameter == 'etax':
        col_data = __get_etax()
    elif parameter == '100etax':
        col_data = __get_etax()
        col_data = [100 * i for i in col_data]
    elif parameter == 'etaxp':
        col_data = __get_etaxp()
    elif parameter == 'psix':
        col_data = __get_psix()
    elif parameter == 'psiy':
        col_data = __get_psiy()
    else:
        raise UnfinishedWork(f'cannot get {parameter} data')
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


def plot_lattice(lattice, parameters, with_layout=True):
    """plot_lattice(lattice, 'betax', True)
    plot_lattice(lattice, ['betax', 'betay'], True)
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


def plot_layout_in_ax(ele_list: list, ax: Axes, ratio=0.03):
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


def plot_RDTs_along_ring(lattice, driving_terms_plot_data):
    """plot"""
    labelsize = 15
    ticksize = 12
    fig = plt.figure(figsize=(7, 8))
    ax1 = fig.add_axes([0.1, 0.55, 0.85, 0.4])
    ax2 = fig.add_axes([0.1, 0.07, 0.85, 0.4])
    ax11 = ax1.twinx()
    ax22 = ax2.twinx()
    plot_layout_in_ax(lattice.elements, ax11)
    plot_layout_in_ax(lattice.elements, ax22)
    line1, = ax1.plot(driving_terms_plot_data['s'], np.abs(driving_terms_plot_data['h21000']), label='h21000')
    line2, = ax1.plot(driving_terms_plot_data['s'], np.abs(driving_terms_plot_data['h30000']), label='h30000')
    line3, = ax1.plot(driving_terms_plot_data['s'], np.abs(driving_terms_plot_data['h10110']), label='h10110')
    line4, = ax1.plot(driving_terms_plot_data['s'], np.abs(driving_terms_plot_data['h10020']), label='h10020')
    line5, = ax1.plot(driving_terms_plot_data['s'], np.abs(driving_terms_plot_data['h10200']), label='h10200')
    ax1.legend(handles=[line1, line2, line3, line4, line5], fontsize=ticksize)
    line9, = ax2.plot(driving_terms_plot_data['s'],  np.abs(driving_terms_plot_data['h31000']), label='h31000')
    line10, = ax2.plot(driving_terms_plot_data['s'], np.abs(driving_terms_plot_data['h40000']), label='h40000')
    line11, = ax2.plot(driving_terms_plot_data['s'], np.abs(driving_terms_plot_data['h20110']), label='h20110')
    line12, = ax2.plot(driving_terms_plot_data['s'], np.abs(driving_terms_plot_data['h11200']), label='h11200')
    line13, = ax2.plot(driving_terms_plot_data['s'], np.abs(driving_terms_plot_data['h20020']), label='h20020')
    line14, = ax2.plot(driving_terms_plot_data['s'], np.abs(driving_terms_plot_data['h20200']), label='h20200')
    line15, = ax2.plot(driving_terms_plot_data['s'], np.abs(driving_terms_plot_data['h00310']), label='h00310')
    line16, = ax2.plot(driving_terms_plot_data['s'], np.abs(driving_terms_plot_data['h00400']), label='h00400')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.set_xlim(0, driving_terms_plot_data['s'][-1])
    ax2.set_xlim(0, driving_terms_plot_data['s'][-1])
    ax1.set_xlabel('s [m]', fontsize=labelsize)
    ax2.set_xlabel('s [m]', fontsize=labelsize)
    ax2.legend(handles=[line9, line10, line11, line12, line13, line14, line15, line16], fontsize=ticksize, ncol=2)
    plt.show()
