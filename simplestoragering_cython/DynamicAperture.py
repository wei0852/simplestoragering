# -*- coding: utf-8 -*-
import numpy as np
from .track import symplectic_track
from .CSLattice import CSLattice
from .exceptions import ParticleLost


class Grid:

    def particles(self):
        """generate particles to track"""
        pass

    def run(self):
        pass


# def parallel_search_inner(vv):
#     set_ref_energy(3500)
#     try:
#         symplectic_track([vv[0], 0, vv[1], 0, 0, vv[2]], vv[4], vv[3], record = False)
#         return [vv[0], vv[1], 0, np.inf, np.inf]
#     except ParticleLost as p:
#         return [vv[0], vv[1], 1, p.location, p.location]


class XDeltaGrid(Grid):
    pass


class XYGrid:
    """generate 2d grid on xy plane.

    generate a grid in the range of (-xmax, xmax)x(0, ymax).
    then use XYGrid.search() to track particles in this grid.

    attributes:
    delta: momentum deviation.
    data: particle data with 5 columns, (x, y, is_lost, survive_turn, lost_location)"""
    def __init__(self, xmax=None, nx=None, ymax=None, ny=None, delta=0, xlist=None, ylist=None) -> None:
        self.xrange = np.linspace(-xmax, xmax, nx * 2 + 1) if xlist is None else xlist
        self.yrange = np.linspace(0, ymax, ny + 1) if ylist is None else ylist
        self.delta = delta
        self.data = None

    def search(self, lattice: CSLattice, n_turns=100, with_rf=False):
        """"""
        if with_rf:
            raise Exception('Unfinished. 4D tracking only.')
        self.data = np.zeros((len(self.xrange) * len(self.yrange), 5))
        area_count = 0
        i = 0
        for y in self.yrange:
            for x in self.xrange:
                try:
                    symplectic_track([x, 0, y, 0, 0, self.delta], lattice, n_turns, record = False)
                    self.data[i, :] = [x, y, 0, np.inf, np.inf]
                    i += 1
                    area_count += 1
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
    Warning: the code of tracking is not precise enough!!!

    Params:
        n_lines: int = 5,
        xmax: float = 0.01,
        ymax: float = 0.01,
        n_points: int = 10,
        n_splits: int = 0,
        split_fraction=0.5,
        delta=0, momentum deviation.
        verbose=True,

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
        power
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
            xy0 = np.zeros(2)
            xymax = np.array([self.xmax * np.sin(theta), self.ymax * np.cos(theta)])
            self.aperture[i, :] = self._search_line(xy0, xymax, self.nx, self.n_splits, n_turns, lattice)

        area = 0
        for i in range(self.aperture.shape[0] - 1):
            area += abs(self.aperture[i, 0] * self.aperture[i + 1, 1] - self.aperture[i, 1] * self.aperture[i+1, 0])
        self.area = area / 2

    def _search_line(self, xy0, xymax, nx, n_splits, n_turns, lattice):
        is_lost = False
        xy = np.linspace(xy0, xymax, nx + 1)
        for i, xymax in enumerate(xy):
            try:
                symplectic_track([xy[i+1, 0], 0, xy[i+1, 1], 0, 0, self.delta], lattice, n_turns, record=False)
            except ParticleLost as p:
                xy0 = xy[i, :]
                nx = int(1 / self.split_fraction)
                xymax = xy[i+1, :]
                is_lost = True
                if n_splits > 0:
                    return self._search_line(xy0, xymax, nx, n_splits - 1, n_turns, lattice)
        if is_lost:
            if self.verbose:
                print(f'    Particle lost at ({xymax[0]*1e3:.1f}, {xymax[1]*1e3:.1f}) mm.')
            return xy0
        else:
            if self.verbose:
                print('    Particle survived.')
            return xymax


class DynamicAperture(object):
    """find dynamic aperture of lattice"""

    def __init__(self, lattice, grid) -> None:
        pass

    def search(self):
        pass
