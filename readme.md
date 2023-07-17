# SimpleStorageRing
SimpleStorageRing is a python project to simulate simple storage ring and calculate lattice data,
which can calculate 
* linear optics, 
* **driving terms** (using the RDT fluctuation data, the number of iterations for calculating the crossing terms is reduced from $N(N+1)/2$ to $N$, and the calculation speed is greatly increased). 
* **fluctuation of driving terms**
* higher-order chromaticities (by calculating the tunes of off-momentum closed orbit).

### Here is an [**example**](https://github.com/wei0852/simplestoragering/blob/master/Example.ipynb) of calculating a 7BA lattice.

There is a Cython version, which needs to be compiled first.
```command
python setup.py build_ext --inplace
```
Cython translates the code to C and compile it. And the calculation will be faster.  
reference: http://docs.cython.org/en/stable/src/quickstart/build.html

-------------------------
## components
A storage ring consists of many magnets to guide electron beam,

* `Mark(name: str)`
* `Drift(name: str, length: float)`
* `HBend(name: str, length: float, theta: float, theta_in: float, theta_out: float, n_slices: int)`, horizontal bend.
* `Quadrupole(name: str, length: float, k1: float, n_slices: int = 4)`
* `Sextupole(name: str, length: float, k2: float, n_slices: int)`
* `Octupole(name: str = None, length: float = 0, k3: float = 0, n_slices: int = 1)`

$k_n = \dfrac{q}{P_0} \dfrac{\partial^n B_y}{\partial x^n}$. All the components are child classes of the Element class. 

Attributes:

* magnet data
    * name: str, 
    * length: float, 
    * type: str, type of component.
    * h: float, curvature of the reference trajectory.
    * theta_in, theta_out: float, edge angle of bend.
    * k1, k2, k3: float, multipole strength.
    * n_slices: int, slice magnets for RDTs calculation and particle tracking (and higher-order chromaticities).
    * matrix: 6x6 np.ndarray, transport matrix.
* data about beam and lattice.
    * s: float, location in a line or a ring.
    * closed_orbit: list[6]
    * betax, alphax, gammax, psix, betay, alphay, gammay, psiy, etax, etaxp, etay, etayp: float, twiss parameters and dispersion.
    * nux, nuy: float, psix / 2$\pi$, psiy / 2$\pi$
    * curl_H: float, dispersion H-function (curly-H function) $\mathcal{H}_x=\gamma_x \eta_x^2 + 2\alpha_x \eta_x \eta_x' + \beta_x\eta_x'^2$.

and methods:

* copy(), return a same component without data about beam or lattice. 
* slice(n_slices: int) -> list[Elements]
* linear_optics() -> np.array([i1, i2, i3, i4, i5, xix, xiy]), 
                           np.array([betax, alphax, gammax, betay, alphay, gammay, etax, etaxp, etay, etayp, psix, psiy])

## Courant-Snyder lattice
`CSLattice(ele_list: list[Element], n_periods: int, coupling: float=0.0)`, lattice solved by cs method.

Attributes:
* length: float
* n_periods: int, number of periods
* angle, abs_angle: float
* elements: list of Elements
* mark: Dictionary of Mark in lattice. The key is the name of Mark, and the value is a list of Mark with the same name.
* initialize twiss
    * twiss_x0, twiss_y0: np.ndarray, [$\beta$, $\alpha$, $\gamma$]
    * eta_x0, eta_y0: np.ndarray, [$\eta$, $\eta'$]. eta_y0=[0,0] because the coupled motion has not be considered yet.
* nux, nuy: Tunes
* xi_x, xi_y: Chromaticities
* natural_xi_x, natural_xi_y: Natural chromaticities
* I1, I2, I3, I4, I5: radiation integrals.
* Jx, Jy, Js: horizontal / vertical / longitudinal damping partition number
* sigma_e: natural energy spread
* emittance: natural emittance
* U0: energy loss [MeV]
* f_c: frequency
* tau_s, tau_x, tau_y: longitudinal / horizontal / vertical damping time
* alpha: Momentum compaction
* etap: phase slip factor

Methods:
* set_initial_twiss(betax, alphax, betay, alphay, etax, etaxp, etay, etayp)
  
    If periodic solutions are not used ( linear_optics(periodicity=False) ).
* linear_optics(periodicity=True, line_mode=False)
* driving_terms(printout=True) -> DrivingTerms
* driving_terms_plot_data() -> dictionary
* higher_order_chromaticity(printout=True, order=3, delta=1e-3, matrix_precision=1e-9, resdl_limit=1e-16)
  
  compute higher order chromaticity with the tunes of 4d off-momentum closed orbit.
  Return {'xi2x': float, 'xi2y': float, 'xi3x': float, 'xi3y': float}
* slice_elements(drift_maxlength=10.0, bend_maxlength=10.0, quad_maxlength=10.0, sext_maxlength=10.0)
 
   slice elements to obtain smooth curves of optical functions.

Use `help()` for more details.
## Functions

`symplectic_track(particle, lattice, n_turns: int, record = True)`, **Unfinished**, only 4-d track.

`track_4d_closed_orbit(lattice: CSLattice, delta, resdl_limit=1e-16, matrix_precision=1e-9)`

`chromaticity_correction(lattice: CSLattice, sextupole_name_list: list, target: list=None, initial_k2=None, update_sext=True, printout=True)`
**incomplete**: cannot limit the strength of sextupole.

`output_opa_file(lattice: CSLattice, file_name=None)`

`output_elegant_file(lattice: CSLattice, filename=None, new_version=True)`

and some functions to visualize lattice data quickly.

`plot_layout_in_ax(ele_list: list, ax: Axes, ratio=0.03)`

`plot_resonance_line_in_ax(ax: Axes, order: int = 3, refnux: float=None, refnuy: float=None)`

`plot_lattice(ele_list, parameter: str or list[str], with_layout=True)`

