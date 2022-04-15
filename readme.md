# SimpleStorageRing
SimpleStorageRing is a python project to simulate simple storage ring and calculate lattice data.

You can sort the components to generate a lattice or a ring, and then calculate the data of storage ring by 
**slim method** or Courant-Snyder method. 


-------------------------
## components
A storage ring consists of many magnets to guide electron beam,

* `Drift(name: str, length: float)`, drift has no magnetic field.
* `HBend(name: str, length: float, theta: float, theta_in: float, theta_out: float)`, horizontal bend.
* `Quadrupole(name: str, length: float, k1: float)`
* `Sextupole(name: str, length: float, k2: float, n_slices: int)`, $k_2 = \dfrac{q}{P_0} \dfrac{\partial^2 B_y}{\partial x^2}$
* `Mark(name: str)`, marker

All the components are child classes of the Element class.


## slim method
A. Chao, “SLIM Formalism — Orbital Motion”， Lecture notes on special topics in accelerator 
physics

## Courant-Snyder

`CSLattice(element_list: list, periods_number: int = 1,  coupling: float = 0.00)`

CSLattice class has some methods to calculate twiss parameters, storage ring parameters, resonance driving terms and higer order chromaticity

There are some auxiliary functions in plotlib.py to plot twiss parameters.
