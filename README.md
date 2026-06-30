# SimpleStorageRing
SimpleStorageRing is a python project to simulate single particle dynamics in a simple storage ring and calculate lattice data.

You can sort the components to generate a ring, and then calculate the data of storage ring by Courant-Snyder method. 


## Installation

Download from PyPi:

```command
pip install -U simplestoragering
```

or download the source files and then compile them using Cython:

----------------------
```command
git clone https://github.com/wei0852/simplestoragering.git
cd simplestoragering
pip install .
```

## Usage

see `example_features.py`.

set reference particle (electron) energy first:

```
set_ref_energy(energy_MeV: float)
```
or
```
set_ref_gamma(gamma: float)
```
### components
A storage ring consists of many magnets to guide electron beam,

* `Mark`
* `Drift`
* `HBend`, horizontal bend.
* `Quadrupole`
* `Sextupole`
* `Octupole`

$k_n = \dfrac{q}{P_0} \dfrac{\partial^n B_y}{\partial x^n}$. All the components are child classes of the Element class. 

### Courant-Snyder lattice
`CSLattice(ele_list: list[Element], n_periods: int)`, lattice solved by C-S method.

## Functions

`symplectic_track(particle, lattice, n_turns: int, record = True)`

`track_4d_closed_orbit(lattice: CSLattice, delta, resdl_limit=1e-16, matrix_precision=1e-9)`

`output_opa_file(lattice: CSLattice, file_name=None)`

`output_elegant_file(lattice: CSLattice, filename=None, new_version=True)`

`chromaticity_correction(lattice: CSLattice, sextupole_name_list: list, target: list=None, initial_k2=None, update_sext=True, verbose=True)`

`adjust_tunes(lattice: CSLattice, quadrupole_name_list: list, target: list, iterations=5, initialize=True)`

## DA & LMA

* XYGrid()
* XDeltaGrid()
* NLine()
* LocalMomentumAperture()

and some functions:

`plot_layout_in_ax(ele_list: list, ax: Axes, ratio=0.03)`

`plot_resonance_line_in_ax(ax: Axes, order: int = 3, refnux: float=None, refnuy: float=None)`

`plot_lattice(ele_list, parameter: str or list[str], with_layout=True)`

or use `get_col(ele_list, parameter)` to get a column of the parameter along the element list.

