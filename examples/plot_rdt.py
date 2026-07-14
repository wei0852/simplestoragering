from pathlib import Path
import sys

cwd = Path.cwd().resolve()
project_root = next(
    p for p in (cwd, *cwd.parents)
    if (p / "simplestoragering").exists() and (p / "examples").exists()
)

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "examples"))

print(f"Project root: {project_root}")

import matplotlib.pyplot as plt
from simplestoragering.objectives import ring_averaged_rdt
import simplestoragering as ssr
import numpy as np
from example_lattice import generate_ring

def plot_rdt():
    ring = generate_ring()
    ring.linear_optics()  # the number of periods of cells is 14. calculate the ring data using one cell.
    rdts = ring.driving_terms()

    print(ring_averaged_rdt(rdts))

    buildup = rdts.buildup_fluctuation()

    plot_data = ssr.plot_RDTs_along_ring(ring, 'h', show=False)
    
    for k in ['h2100', 'h3000', 'h1011', 'h1002', 'h1020', 'h3100', 'h4000', 'h2011', 'h1120', 'h2002', 'h2020', 'h0031', 'h0040']:
        plt.figure()
        plt.scatter(buildup['s'], np.abs(buildup[k]), color='red', label='buildup fluctuation')
        plt.step(buildup['s'], np.abs(buildup[k]), color='red', where='pre')
        plt.plot(plot_data['s'], np.abs(plot_data[k]), 'b:')
        # plt.scatter(rdts.s[0], 0, color='k')
        plt.xlabel('s [m]')
        plt.ylabel(f'|{k}|')
        plt.title(f'RDT {k} along the ring')
        plt.grid()
        ssr.plot_layout_in_ax(ring.elements, plt.gca().twinx())
        plt.show()

    natural = rdts.natural_fluctuation()

    plot_data = ssr.plot_RDTs_along_ring(ring, 'f', show=False)
    
    for k in ['f2100', 'f3000', 'f1011', 'f1002', 'f1020', 'f3100', 'f4000', 'f2011', 'f1120', 'f2002', 'f2020', 'f0031', 'f0040']:
        plt.figure()
        plt.scatter(natural['s'], np.abs(natural[k]), color='red', label='natural fluctuation')
        plt.step(natural['s'], np.abs(natural[k]), color='red', where='pre')
        plt.plot(plot_data['s'], np.abs(plot_data[k]), 'b:')
        plt.xlabel('s [m]')
        plt.ylabel(f'|{k}|')
        plt.title(f'RDT {k} along the ring')
        plt.grid()
        ssr.plot_layout_in_ax(ring.elements, plt.gca().twinx())
        plt.show()


if __name__ == '__main__':
    plot_rdt()
