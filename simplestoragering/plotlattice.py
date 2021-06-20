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

    def __get_etax():
        __col = []
        for ele in lattice.ele_slices:
            __col.append(ele.etax)
        return __col

    assert isinstance(lattice, CSLattice) or isinstance(lattice, SlimRing)
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
    elif parameter == 'etax':
        col_data = __get_etax()
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
    layout_s, layout_data = get_layout(lattice)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    for k, v in y.items():
        ax2.plot(s, v, label=k)
    plt.legend()
    ax1.fill(layout_s, layout_data, color='#cccccc')
    plt.show()

