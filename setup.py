from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(name='*',
              sources=['simplestoragering/*.pyx'
                       ],
            include_dirs=[np.get_include(), './simplestoragering/atintegrators/'])]

setup(
    packages=find_packages(include=['simplestoragering', 'simplestoragering.*']),
    ext_modules=cythonize(extensions))
