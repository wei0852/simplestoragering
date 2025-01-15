from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(name='*',
              sources=['simplestoragering/*.pyx'
                       ],
            include_dirs=[np.get_include(), './simplestoragering/atintegrators/'])]

setup(
    ext_modules=cythonize(extensions))