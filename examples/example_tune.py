"""Tune analysis: tune shift with position, action and momentum deviation."""
import numpy as np
import matplotlib.pyplot as plt

import simplestoragering as ssr
from example_lattice import generate_ring


if __name__ == '__main__':
    ring = generate_ring()
    ring.linear_optics()
    print(f'linear tune: nux = {ring.nux:.4f}, nuy = {ring.nuy:.4f}\n')

    n_turns = 1024

    # tune vs initial horizontal / vertical position
    x_scan = ssr.tune_shift_with_x(ring, np.linspace(-0.004, 0.004, 9),
                                   n_turns=n_turns, verbose=True)
    x_scan.show()
    y_scan = ssr.tune_shift_with_y(ring, np.linspace(0, 0.004, 9), n_turns=n_turns)

    # tune vs betatron action (detuning with amplitude)
    J_list = np.linspace(1e-9, 1e-6, 9)
    amp_x = ssr.tune_shift_with_amplitude(ring, J_list, plane='x', n_turns=n_turns)
    amp_y = ssr.tune_shift_with_amplitude(ring, J_list, plane='y', n_turns=n_turns)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for a, scan in [(ax[0, 0], x_scan), (ax[0, 1], y_scan),
                    (ax[1, 0], amp_x), (ax[1, 1], amp_y)]:
        a.scatter(scan.param, scan.nux, label='$\\nu_x$')
        a.scatter(scan.param, scan.nuy, label='$\\nu_y$')
        a.set_xlabel(scan.xlabel)
        a.set_ylabel('tunes')
        a.legend()
        a.grid('on')
    plt.tight_layout()
    plt.show()

    # the analytic amplitude detuning terms from driving terms
    print(f'ADTS (dQxx, dQxy, dQyy) = {ring.adts()}')

    # tune vs momentum deviation (chromaticity / nonlinear tune shift)
    delta = ssr.tune_shift_with_delta(ring, np.arange(-0.02, 0.021, 0.005), n_turns=n_turns)
    delta.save(filename='example_tune_delta.csv')
    delta.show()
