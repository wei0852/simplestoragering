"""Dynamic aperture searches on grids and radial lines."""
import numpy as np
import matplotlib.pyplot as plt
from ..track import symplectic_track, radiation_track
from ..CSLattice import CSLattice
from ..exceptions import ParticleLost, Unstable


__all__ = ["XDeltaGrid", "XYGrid", "NLine", "XDeltaLines"]


class XDeltaGrid:
    """generate 2d grid on x-delta plane.

    generate a grid in the range of (-xmax, xmax)x(-delta_max, delta_max).
    then use XDeltaGrid.search() to track particles in this grid.

    attributes:
    delta: momentum deviation.
    data: particle data with 5 columns, (x, dp, is_lost, survive_turn, lost_location)"""
    def __init__(self, xmax=None, nx=None, delta_max=None, ndelta=None, y=1e-6, xlist=None, delta_list=None) -> None:
        self.xrange = np.linspace(-xmax, xmax, nx * 2 + 1) if xlist is None else xlist
        self.delta_range = np.linspace(-delta_max, delta_max, ndelta * 2 + 1) if delta_list is None else delta_list
        self.y = y
        self.data = None

    def search(self, lattice: CSLattice, n_turns=100, radiation=False):
        """"""
        self.data = np.zeros((len(self.xrange) * len(self.delta_range), 5))
        i = 0
        for dp in self.delta_range:
            for x in self.xrange:
                try:
                    if radiation is True:
                        radiation_track([x, 0, self.y, 0, 0, dp], lattice, n_turns, record = False)
                    else:
                        symplectic_track([x, 0, self.y, 0, 0, dp], lattice, n_turns, record = False)
                    self.data[i, :] = [x, dp, 0, np.inf, np.inf]
                    i += 1
                except ParticleLost as p:
                    self.data[i, :3] = [x, dp, 1]
                    self.data[i, 3] = int(p.location / lattice.length)
                    self.data[i, 4] = p.location % (lattice.length / lattice.n_periods)
                    i += 1

    def save_data(self):
        pass


class XYGrid:
    """generate 2d grid on xy plane.

    generate a grid in the range of (-xmax, xmax)x(1e-6, ymax).
    then use XYGrid.search() to track particles in this grid.

    attributes:
    delta: momentum deviation.
    data: particle data with 5 columns, (x, y, is_lost, survive_turn, lost_location)"""
    def __init__(self, xmax=None, nx=None, ymax=None, ny=None, delta=0, xlist=None, ylist=None) -> None:
        self.xrange = np.linspace(-xmax, xmax, nx * 2 + 1) if xlist is None else xlist
        self.yrange = np.linspace(1e-6, ymax, ny + 1) if ylist is None else ylist
        self.delta = delta
        self.data = None

    def search(self, lattice: CSLattice, n_turns=100, radiation=False):
        """"""
        self.data = np.zeros((len(self.xrange) * len(self.yrange), 5))
        i = 0
        for y in self.yrange:
            for x in self.xrange:
                try:
                    if radiation is True:
                        radiation_track([x, 0, y, 0, 0, self.delta], lattice, n_turns, record = False)
                    else:
                        symplectic_track([x, 0, y, 0, 0, self.delta], lattice, n_turns, record = False)
                    self.data[i, :] = [x, y, 0, np.inf, np.inf]
                    i += 1
                except ParticleLost as p:
                    self.data[i, :3] = [x, y, 1]
                    self.data[i, 3] = int(p.location / lattice.length)
                    self.data[i, 4] = p.location % (lattice.length / lattice.n_periods)
                    i += 1

    def save_data(self):
        pass


class NLine:
    """n-line mode to find dynamic aperture.

    generate n_lines, search along the lines, and split n_splits times once the particle is lost for preciser result.
    similar to ELEGANT.

    Params:
        n_lines: int = 5,
        xmax: float = 0.01,
        ymax: float = 0.01,
        n_points: int = 10,
        n_splits: int = 0,
        split_fraction=0.5,
        delta=0, momentum deviation.
        verbose=True, print details.

    Attributes:
        aperture, np.ndarray, (n_lines, 2) shape.
        area, m^2
        n_lines
        n_splits
        nx
        split_fraction
        xmax
        ymax
        delta
        verbose
    """
    def __init__(self,
                 n_lines: int = 5,
                 xmax: float = 0.01,
                 ymax: float = 0.01,
                 n_points: int = 10,
                 n_splits: int = 0,
                 split_fraction=0.5,
                 delta=0,
                 verbose=True) -> None:
        assert n_lines >= 3, 'n_lines at least 5.'
        self.aperture = np.zeros((n_lines, 2))
        self.area = 0
        self.n_lines = n_lines
        self.n_splits = n_splits
        self.nx = n_points
        self.split_fraction = split_fraction
        self.xmax = xmax
        self.ymax = ymax
        self.delta = delta
        self.verbose = verbose

    def search(self, lattice: CSLattice, n_turns=100, radiation=False):
        for i, theta in enumerate(np.linspace(-np.pi / 2, np.pi / 2, self.n_lines)):
            if self.verbose:
                print(f'start line {i+1}...')
            xy0 = np.zeros(2) + 1e-6
            xymax = np.array([self.xmax * np.sin(theta), self.ymax * np.cos(theta)])
            self.aperture[i, :] = self._search_line(xy0, xymax, self.nx, self.n_splits, n_turns, lattice, radiation)

        area = 0
        for i in range(self.aperture.shape[0] - 1):
            area += abs(self.aperture[i, 0] * self.aperture[i + 1, 1] - self.aperture[i, 1] * self.aperture[i+1, 0])
        self.area = area / 2

    def _search_line(self, xy0, xymax, nx, n_splits, n_turns, lattice, radiation):
        xy = np.linspace(xy0, xymax, nx + 1)
        for i in range(nx):
            try:
                if radiation is True:
                    rin = np.array([xy[i+1, 0], 0, xy[i+1, 1], 0, 0, self.delta])
                    radiation_track(rin, lattice, n_turns, record=False)
                else:
                    symplectic_track([xy[i+1, 0], 0, xy[i+1, 1], 0, 0, self.delta], lattice, n_turns, record=False)
            except ParticleLost as p:
                xy0 = xy[i, :]
                nx = int(1 / self.split_fraction)
                xymax = xy[i+1, :]
                if n_splits > 0:
                    return self._search_line(xy0, xymax, nx, n_splits - 1, n_turns, lattice, radiation)
                else:
                    if self.verbose:
                        print(f'    Particle lost at ({xymax[0]*1e3:.1f}, {xymax[1]*1e3:.1f}) mm.')
                    return xy0
        if self.verbose:
            print('    Particle survived.')
        return xymax

    def save(self, filename=None, header=None):
        """header: String that will be written at the beginning of the file."""
        filename = 'DynamicAperture.csv' if filename is None else filename
        if header is None:
            header = ''
        else:
            header += '\n'
        header += f'{self.n_lines} lines, search range: ({self.xmax}, {self.ymax}), n_points: {self.nx}, number of split: {self.n_splits}, split fraction: {self.split_fraction}\n'
        header += f'x y area={self.area}'
        np.savetxt(filename, self.aperture, fmt='%10.6f', comments='#', header=header)

    def show(self, ax: plt.Axes=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.aperture[:, 0] * 1000, self.aperture[:, 1] * 1000, c='k')
        ax.grid('on')
        ax.set_xlabel('$x$ [mm]', fontsize=15)
        ax.set_ylabel('$y$ [mm]', fontsize=15)
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        if ax is None:
            plt.tight_layout()
            plt.show()


class XDeltaLines:
    """line search mode to track off-momentum horizontal dynamic aperture.

    For each delta value, track along two lines (positive and negative directions) with increasing amplitude starting at the off-momentum closed-orbit.
    The vertical amplitude is y=1e-6.
    """
    def __init__(self,
                 delta_list: np.ndarray,
                 xmax: float = 0.01,
                 n_points: int = 10,
                 n_splits: int = 0,
                 split_fraction=0.5) -> None:
        """
        Args:
            delta_list (np.ndarray): An array of delta values for which the dynamic aperture will be calculated.
            xmax (float, optional): The maximum search range for the horizontal coordinate. Defaults to 0.01.
            n_points (int, optional): The number of points in the search range. Defaults to 10.
            n_splits (int, optional): The number of times to split the search range when a particle is lost. Defaults to 0.
            split_fraction (float, optional): The fraction used to split the search range. Defaults to 0.5.
        """

        self.aperture = np.zeros((delta_list.shape[0], 4))
        self.delta_list = delta_list
        self.n_splits = n_splits
        self.n_points = n_points
        self.split_fraction = split_fraction
        self.r_max = xmax
        self.verbose = None

    def search(self, lattice: CSLattice, n_turns=100, radiation=False, verbose=True):
        """
        Search for the off-momentum horizontal dynamic aperture for each delta value.

        Args:
            lattice (CSLattice): The lattice object for which the dynamic aperture is calculated.
            n_turns (int, optional): The number of turns to track the particle. Defaults to 100.
            verbose (bool, optional): Whether to print detailed information during the search. Defaults to True.
        """
        self.verbose = verbose
        for i, delta in enumerate(self.delta_list):
            self.aperture[i, 0] = delta
            try:
                lattice.off_momentum_optics(delta=delta)
                betax = lattice.elements[0].betax
                alphax = lattice.elements[0].alphax
                cox = lattice.elements[0].closed_orbit[0]
                cop = lattice.elements[0].closed_orbit[1]
                self.aperture[i, 3] = cox
            except Unstable:
                if self.verbose:
                    print(f'Cannot find closed orbit at delta={delta*100:.2f}%.')
                self.aperture[i, 1] = np.nan
                self.aperture[i, 2] = np.nan
                self.aperture[i, 3] = np.nan
                continue
            if self.verbose:
                print(f'Start searching delta={delta*100:.2f}% ......')
            r = self.__search_delta(cox, 0, self.r_max, self.n_points, self.n_splits, n_turns, lattice, delta, radiation)
            self.aperture[i, 1] = r + cox
            r = self.__search_delta(cox, 0, -self.r_max, self.n_points, self.n_splits, n_turns, lattice, delta, radiation)
            self.aperture[i, 2] = r + cox

    def __search_delta(self, cox, x0, x_max, nx, n_splits, n_turns, lattice, dp, radiation):
        x = np.linspace(x0, x_max, nx+1)
        for i in range(nx):
            try:
                if radiation is True:
                    radiation_track([cox + x[i + 1], 0.0, 1e-6, 0, 0, dp], lattice, n_turns, record=False)
                else:
                    symplectic_track([cox + x[i + 1], 0.0, 1e-6, 0, 0, dp], lattice, n_turns, record=False)
            except ParticleLost:
                x0 = x[i]
                nx = int(1 / self.split_fraction)
                x_max = x[i+1]
                if n_splits > 0:
                    return self.__search_delta(cox, x0, x_max, nx, n_splits - 1, n_turns, lattice, dp, radiation)
                else:
                    if self.verbose:
                        print(f'    Particle lost at {x_max * 1e3:.1f} mm.')
                    return x0
        if self.verbose:
            print('    Particle survived.')
        return x_max

    def show(self, ax: plt.Axes=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.aperture[:, 0] * 100, self.aperture[:, 1] * 1000, c='k')
        ax.plot(self.aperture[:, 0] * 100, self.aperture[:, 2] * 1000, c='k')
        ax.plot(self.aperture[:, 0] * 100, self.aperture[:, 3] * 1000, c='C1')
        ax.grid('on')
        ax.set_xlabel('$\\delta$ [%]', fontsize=15)
        ax.set_ylabel('$x$ [mm]', fontsize=15)
        # plt.legend(fontsize=ticksize)
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        if ax is None:
            plt.tight_layout()
            plt.show()

    def save(self, filename=None, header=None):
        """
        Save the calculated dynamic aperture data to a file.

        Args:
            filename (str, optional): The name of the file to save the data. If None, it defaults to 'HorizontalDynamicAperture.csv'.
            header (str, optional): String that will be written at the beginning of the file.

        Note:
            The header: String that will be written at the beginning of the file.
        """
        filename = 'HorizontalDynamicAperture.csv' if filename is None else filename
        if header is None:
            header = ''
        else:
            header += '\n'
        header += f'search range: {self.r_max}, n_points: {self.n_points}, number of split: {self.n_splits}, split fraction: {self.split_fraction}\n'
        header += f'delta,  xmax,  xmin,  xco'
        np.savetxt(filename, self.aperture, delimiter=',', fmt='%10.6f', comments='#', header=header)

