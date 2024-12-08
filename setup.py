from setuptools import setup
from Cython.Build import cythonize
import numpy

setup (
    ext_modules = cythonize(( 'simulation_functions.pyx', 'old_sim_functions.pyx' )),
    include_dirs=[numpy.get_include()]
)