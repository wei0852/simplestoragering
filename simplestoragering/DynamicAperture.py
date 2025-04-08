# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from .track import symplectic_track
from .CSLattice import CSLattice
from .exceptions import ParticleLost, Unstable
from .components import Mark


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

    def search(self, lattice: CSLattice, n_turns=100, with_rf=False):
        """"""
        if with_rf:
            raise Exception('Unfinished. 4D tracking only.')
        self.data = np.zeros((len(self.xrange) * len(self.delta_range), 5))
        i = 0
        for dp in self.delta_range:
            for x in self.xrange:
                try:
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

    def search(self, lattice: CSLattice, n_turns=100, with_rf=False):
        """"""
        if with_rf:
            raise Exception('Unfinished. 4D tracking only.')
        self.data = np.zeros((len(self.xrange) * len(self.yrange), 5))
        i = 0
        for y in self.yrange:
            for x in self.xrange:
                try:
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

    def search(self, lattice: CSLattice, n_turns=100, with_rf=False):
        if with_rf:
            raise Exception('Unfinished. 4D tracking only.')
        for i, theta in enumerate(np.linspace(-np.pi / 2, np.pi / 2, self.n_lines)):
            if self.verbose:
                print(f'start line {i+1}...')
            xy0 = np.zeros(2) + 1e-6
            xymax = np.array([self.xmax * np.sin(theta), self.ymax * np.cos(theta)])
            self.aperture[i, :] = self._search_line(xy0, xymax, self.nx, self.n_splits, n_turns, lattice)

        area = 0
        for i in range(self.aperture.shape[0] - 1):
            area += abs(self.aperture[i, 0] * self.aperture[i + 1, 1] - self.aperture[i, 1] * self.aperture[i+1, 0])
        self.area = area / 2

    def _search_line(self, xy0, xymax, nx, n_splits, n_turns, lattice):
        xy = np.linspace(xy0, xymax, nx + 1)
        for i in range(nx):
            try:
                symplectic_track([xy[i+1, 0], 0, xy[i+1, 1], 0, 0, self.delta], lattice, n_turns, record=False)
            except ParticleLost as p:
                xy0 = xy[i, :]
                nx = int(1 / self.split_fraction)
                xymax = xy[i+1, :]
                if n_splits > 0:
                    return self._search_line(xy0, xymax, nx, n_splits - 1, n_turns, lattice)
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

    def search(self, lattice: CSLattice, n_turns=100, verbose=True):
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
            r = self.__search_delta(cox, 0, self.r_max, self.n_points, self.n_splits, n_turns, lattice, delta)
            self.aperture[i, 1] = r + cox
            r = self.__search_delta(cox, 0, -self.r_max, self.n_points, self.n_splits, n_turns, lattice, delta)
            self.aperture[i, 2] = r + cox

    def __search_delta(self, cox, x0, x_max, nx, n_splits, n_turns, lattice, dp):
        x = np.linspace(x0, x_max, nx+1)
        for i in range(nx):
            try:
                symplectic_track([cox + x[i + 1], 0.0, 1e-6, 0, 0, dp], lattice, n_turns, record=False)
            except ParticleLost:
                x0 = x[i]
                nx = int(1 / self.split_fraction)
                x_max = x[i+1]
                if n_splits > 0:
                    return self.__search_delta(cox, x0, x_max, nx, n_splits - 1, n_turns, lattice, dp)
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


class LocalMomentumAperture(object):
    """local momentum aperture.

    similar to ELEGANT.

    Params:
        ds: float or a dictionary {'Drift': float, 'HBend': float, 'Quadrupole': float, 'Sextupole': float, 'Octupole': float}.
        delta_negative_start: float = 0.0,
        delta_positive_start: float = 0.0,
        delta_negative_limit: float = -0.05,
        delta_positive_limit: float = 0.05,
        delta_step = 0.01,
        n_splits: int = 1,
        split_fraction=0.5,
        verbose=True, print details.
        initial_x=1e-6,
        initial_y=1e-6.

    Attributes:
        s: np.array.
        min_delta: np.array.
        max_delta: np.array.
    """

    def __init__(self, lattice: CSLattice, ds=None,
            delta_negative_start=0, delta_positive_start=0, delta_negative_limit=-0.1, delta_positive_limit=0.1, delta_step=0.01,
            n_splits=1, split_fraction=0.5, verbose=True, initial_x=1e-6, initial_y=1e-6):
        if ds is None:
            ds = 0.1
            ele_slices = lattice.slice_elements(ds, ds, ds, ds, ds)
        elif isinstance(ds, float):
            ele_slices = lattice.slice_elements(ds, ds, ds, ds, ds)
        elif isinstance(ds, dict):
            ele_slices = lattice.slice_elements(ds['Drift'], ds['HBend'], ds['Quadrupole'], ds['Sextupole'], ds['Octupole'])
        else:
            raise Exception("ds: float or a dictionary {'Drift': float, 'HBend': float, 'Quadrupole': float, 'Sextupole': float, 'Octupole': float}")
        self.newlattice = CSLattice(ele_slices[:-1])
        self.newlattice.linear_optics()
        self.n_periods = lattice.n_periods
        self.delta_negative_start=delta_negative_start
        self.delta_positive_start=delta_positive_start
        self.delta_negative_limit=delta_negative_limit
        self.delta_positive_limit=delta_positive_limit
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.delta_step=delta_step
        self.n_splits=n_splits
        self.split_fraction=split_fraction
        self.verbose=verbose
        nums_ = len(self.newlattice.elements)
        self.s = np.zeros(nums_)
        self.max_delta = np.zeros(nums_)
        self.min_delta = np.zeros(nums_)

    def search(self, n_turns=100, parallel=False):
        if parallel:
            pass
        else:
            nums_ = len(self.s) - 1
            for i in range(nums_):
                sub_lattice = self.newlattice.elements[i:-1] + self.newlattice.elements[:i]
                self.s[i] = sub_lattice[0].s
                if isinstance(sub_lattice[0], Mark) and (i > 0):
                    self.min_delta[i] = self.min_delta[i-1]
                    self.max_delta[i] = self.max_delta[i-1]
                else:
                    if self.verbose:
                        print(f'Start search MA at s={sub_lattice[0].s:.3f} m ({sub_lattice[0].name})............... {i + 1} / {nums_}')
                    self.min_delta[i] = self._search_one_position(n_turns * self.n_periods, sub_lattice, 
                                              self.delta_negative_start, self.delta_negative_limit, -self.delta_step, self.n_splits)
                    self.max_delta[i] = self._search_one_position(n_turns * self.n_periods, sub_lattice, 
                                              self.delta_positive_start, self.delta_positive_limit, self.delta_step, self.n_splits)
            self.s[-1] = self.newlattice.elements[-1].s
            self.min_delta[-1] = self.min_delta[0]
            self.max_delta[-1] = self.max_delta[0]


    def _search_one_position(self, n_turns, sub_lattice, delta_init, delta_end, delta_step, n_splits):
        delta_survive = delta_init
        for dp in np.arange(delta_init, delta_end + delta_step, delta_step):
            try:
                symplectic_track([self.initial_x, 0, self.initial_y, 0, 0, dp], sub_lattice, n_turns, record=False)
                delta_survive = dp
            except ParticleLost:
                if n_splits > 0:
                    return self._search_one_position(n_turns, sub_lattice, delta_survive, dp, delta_step * self.split_fraction, n_splits - 1)
                else:
                    break
        if self.verbose:
            print(f'    Particle survived at delta={delta_survive * 100:.2f}%.')
        return delta_survive
    
    def save(self, filename=None, header=None):
        """header: String that will be written at the beginning of the file."""
        filename = 'LocalMA.csv' if filename is None else filename
        if header is None:
            header = ''
        else:
            header += '\n'
        header += f'search range: ({self.delta_negative_start}, {self.delta_negative_limit}) + ({self.delta_positive_start}, {self.delta_positive_limit}), step: {self.delta_step}, '
        header += f'number of split: {self.n_splits}, split fraction: {self.split_fraction}, initial x={self.initial_x}, initial y = {self.initial_y}\n'
        header += 's delta_min delta_max'
        ma_data = np.vstack((self.s, self.min_delta, self.max_delta)).T
        np.savetxt(filename, ma_data, fmt='%10.6f', comments='#', header=header)


def is_point_in_polygon(point, polygon):
    num = len(polygon)
    x, y = point
    j = num - 1
    odd_nodes = False
    for i in range(num):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if yi < y <= yj or yj < y <= yi:
            if xi + (y - yi) / (yj - yi) * (xj - xi) < x:
                odd_nodes = not odd_nodes
        j = i
    return odd_nodes


class DynamicAperturePolyhedron:
    """dynamic aperture polyhedron in (x, px, delta) space.
    
    	arXiv:2406.14407 [physics.acc-ph]
    """

    def __init__(self, x_max, n_lines, n_points, n_splits=0, split_fraction=0.5,
                 delta_list_p=None, delta_list_m=None):
        self.n_lines = n_lines
        self.n_points = n_points
        self.n_splits = n_splits
        self.split_fraction = split_fraction
        self.r_max = x_max
        self.verbose = None
        self.delta_list_p = delta_list_p
        self.delta_list_m = delta_list_m
        self.DA = np.zeros((len(self.delta_list_p) + len(self.delta_list_m), self.n_lines, 3))  # x, px, delta
        self.twiss = np.zeros((len(self.delta_list_p) + len(self.delta_list_m), 4))  # fp_x, fp_px, beta_x, alpha_x

    def search(self, lattice, n_turns=100, verbose=True) -> None:
        self.verbose = verbose
        for i, dp in enumerate(np.append(self.delta_list_p, self.delta_list_m)):
            try:
                lattice.off_momentum_optics(delta=dp)
                betax = lattice.elements[0].betax
                alphax = lattice.elements[0].alphax
                cox = lattice.elements[0].closed_orbit[0]
                cop = lattice.elements[0].closed_orbit[1]
                self.twiss[i] = [cox, cop, betax, alphax]
            except Unstable:
                self.twiss[i] = np.nan
                self.DA[i] = np.nan
                self.DA[i, :, 2] = dp
                continue

            for j, phi in enumerate(np.linspace(0, 2*np.pi, self.n_lines+1)[:-1]):
                if self.verbose:
                    print(f'Start searching delta={dp*100:.2f}%, {j+1} of {self.n_lines} lines ......')
                r = self.__search_normal_line(cox, cop, betax, alphax, 0, self.r_max, phi, self.n_points,
                                                  self.n_splits, n_turns, lattice, dp)  # r = sqrt(2 Jx beta_x)
                x = r*np.cos(phi)
                px = -(r/betax) * (np.sin(phi) + alphax * np.cos(phi))
                self.DA[i, j, :] = [x + cox, px + cop, dp]

    def __search_normal_line(self, cox, cop, betax, alphax, r0, r_max, phi, nx, n_splits, n_turns, lattice, dp):
        jx_max = r_max ** 2 / 2 / betax
        jx_0 = r0 ** 2 / 2 / betax
        jx_list = np.linspace(jx_0, jx_max, nx+1)
        x = (2 * betax * jx_list) ** 0.5 * np.cos(phi)
        px = -(2 * jx_list / betax) ** 0.5 * (np.sin(phi) + alphax * np.cos(phi))
        for i in range(nx):
            try:
                symplectic_track([cox + x[i + 1], cop + px[i + 1], 1e-6, 0, 0, dp], lattice, n_turns, record=False)
            except ParticleLost:
                r0 = (jx_list[i] * 2 * betax) ** 0.5
                nx = int(1 / self.split_fraction)
                r_max = (jx_list[i+1] * 2 * betax) ** 0.5
                if n_splits > 0:
                    return self.__search_normal_line(cox, cop, betax, alphax, r0, r_max, phi, nx, n_splits - 1,
                                                         n_turns, lattice, dp)
                else:
                    if self.verbose:
                        print(f'    Particle lost at {r_max*1e3:.1f} mm.')
                    return r0
        if self.verbose:
            print('    Particle survived.')
        return r_max

    def fast_Touschek_tracking(self, lattice) -> np.ndarray:
        num_ele = len(lattice.elements)
        s = np.zeros(num_ele)
        delta_max = np.zeros(num_ele)
        delta_min = np.zeros(num_ele)
        for i in range(len(lattice.elements)):
            sub_lattice = lattice.elements[i:]
            s[i] = sub_lattice[0].s
            dmax = dmin = 0
            jp = len(self.delta_list_p)
            for j, dp in enumerate(self.delta_list_p):
                try:
                    x_f = symplectic_track([1e-6, 0, 1e-6, 0, 0, dp], sub_lattice, 1, record=False)
                except ParticleLost:
                    break
                if is_point_in_polygon(x_f[:2], self.DA[j, :, :2]):
                    dmax = max(dmax, dp)
                else:
                    break
            for j, dp in enumerate(self.delta_list_m):
                try:
                    x_f = symplectic_track([1e-6, 0, 1e-6, 0, 0, dp], sub_lattice, 1, record=False)
                except ParticleLost:
                    break
                if is_point_in_polygon(x_f[:2], self.DA[j+jp, :, :2]):
                    dmin = min(dmin, dp)
                else:
                    break
            delta_max[i] = dmax
            delta_min[i] = dmin
        return np.vstack((s, delta_min, delta_max)).T

    def save(self, file_name, header=None):
        header = None if header is None else header + '\n'
        np.savetxt(f'{file_name}_DA_polyhedron.csv', header=f'{header} x,  px,  delta', X=self.DA.reshape(self.DA.shape[0] * self.DA.shape[1], self.DA.shape[2]))
        np.savetxt(f'{file_name}_off_momentum_optics.csv', header=f'{header} closed_orbit_x, closed_orbit_px, beta_x, alpha_x', X=self.twiss)
