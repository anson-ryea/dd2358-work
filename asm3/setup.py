from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "mandelbrot_cy",
    ["mandelbrot_cy.pyx"],
    include_dirs=[np.get_include()],
)

setup(
    ext_modules=cythonize(ext, compiler_directives={"language_level": "3"}),
)
