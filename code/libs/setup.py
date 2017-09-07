from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name = "ESN-Core",
    ext_modules = cythonize('cython_esn.pyx'),
    include_dirs=[np.get_include()]
)
