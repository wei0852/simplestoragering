from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(['simplestoragering_cython/components.pyx',
    'simplestoragering_cython/line_matrix.pyx',
    'simplestoragering_cython/Drift.pyx',
    'simplestoragering_cython/HBend.pyx',
    'simplestoragering_cython/Quadrupole.pyx',
    'simplestoragering_cython/Sextupole.pyx',
    'simplestoragering_cython/Octupole.pyx',
    'simplestoragering_cython/CSLattice.pyx',
    'simplestoragering_cython/c_functions.pyx',
    'simplestoragering_cython/globalvars.pyx',
    'simplestoragering_cython/track.pyx']),
    include_dirs=[np.get_include()],
)