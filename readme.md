# SimpleStorageRing
SimpleStorageRing is a python project to simulate simple storage ring and calculate lattice data.

You can sort the components to generate a lattice or a ring, and then calculate the data of storage ring by 
**slim method** or Courant-Snyder method. 

------------------
已知存在的问题：
存在二四极组合磁铁的时候，校正色品有误差。

-------------------------
## components
A storage ring consists of many magnets to guide electron beam,

* `Drift(name: str, length: float, n_slices=1)`, drift has no magnetic field.
* `HBend(name: str, length: float, theta: float, theta_in: float, theta_out: float, n_slices=3)`, horizontal bend.
* `Quadrupole(name: str, length: float, k1: float, n_slices: int)`
* `Sextupole(name: str, length: float, k2: float, n_slices: int)`, $k_2 = \dfrac{q}{P_0} \dfrac{\partial^2 B_y}{\partial x^2}$

All the components are child classes of the Element class. The following five necessary methods must be rewritten in each subclass:

* matrix: 6X6 transfer matrix.
* damping_matrix: 6X6 transfer matrix.Considering the radiation effect, the energy will be reduced in components.
* closed_orbit_matrix: 7X7 transfer matrix to calculate the closed orbit. The extra dimension shows the effect of radiation.
* symplectic_track: symplectic track.
* real_track: considering the effect of radiation, the energy will be reduced.

## slim method
A. Chao, “SLIM Formalism — Orbital Motion”， Lecture notes on special topics in accelerator 
physics

## Courant-Snyder
