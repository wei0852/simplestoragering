# -*- coding: utf-8 -*-
"""
this file is unnecessary, I use these functions to quickly visualize lattice data when developing my code.
"""

import matplotlib.pyplot as plt
from .slimlattice import SlimRing
from .cslattice import CSLattice
from .exceptions import UnfinishedWork


def get_col(lattice, parameter: str):
    """get column data"""
    def __get_closed_orbit_delta():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.closed_orbit[5])
        return __col

    def __get_closed_orbit_z():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.closed_orbit[4])
        return __col

    def __get_closed_orbit_x():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.closed_orbit[0])
        return __col

    def __get_closed_orbit_y():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.closed_orbit[2])
        return __col

    def __get_closed_orbit_px():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.closed_orbit[1])
        return __col

    def __get_closed_orbit_py():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.closed_orbit[3])
        return __col

    def __get_s():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.s)
        return __col

    def __get_betax():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.betax)
        return __col

    def __get_betay():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.betay)
        return __col

    def __get_alphax():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.alphax)
        return __col

    def __get_alphay():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.alphay)
        return __col

    def __get_etax():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.etax)
        return __col

    def __get_etaxp():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.etaxp)
        return __col

    def __get_nux():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.nux)
        return __col

    def __get_nuy():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.nuy)
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
    elif parameter == 'nux':
        col_data = __get_nux()
    elif parameter == 'nuy':
        col_data = __get_nuy()
    else:
        raise UnfinishedWork(f'cannot get {parameter} data')
    return col_data


def get_layout(lattice):
    """get layout data of magnets"""
    s = []
    current_s = 0
    magnet = []
    s.append(current_s)
    for ele in lattice.elements:
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
    if with_layout:
        return plot_with_layout(lattice, parameters)
    else:
        return plot_without_layout(lattice, parameters)


def plot_without_layout(lattice, parameters):
    s = get_col(lattice, 's')
    if isinstance(parameters, list):
        for para in parameters:
            plt.plot(s, get_col(lattice, para), label=para)
    elif isinstance(parameters, str):
        plt.plot(s, get_col(lattice, parameters), label=parameters)
    else:
        raise Exception('plot error')
    plt.legend()
    plt.show()


def plot_with_layout(lattice, parameters):
    layout_s, layout_data = get_layout(lattice)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    s = get_col(lattice, 's')
    if isinstance(parameters, list):
        for para in parameters:
            ax2.plot(s, get_col(lattice, para), label=para)
    elif isinstance(parameters, str):
        ax2.plot(s, get_col(lattice, parameters), label=parameters)
    else:
        raise Exception('plot error')
    plt.legend()
    ax1.fill(layout_s, layout_data, color='#cccccc')
    plt.show()


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
