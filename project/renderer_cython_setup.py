from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# 1. Get the numpy include path
numpy_include = numpy.get_include()
print(f"Checking NumPy include path: {numpy_include}")

# 2. Verify it actually exists (helpful for debugging)
if not os.path.exists(os.path.join(numpy_include, 'numpy', 'arrayobject.h')):
    print("WARNING: Could not find arrayobject.h in the include path!")

ext = Extension(
    "renderer_cython_fun",           # Matches your .pyx filename
    sources=["renderer_cython_fun.pyx"],
    include_dirs=[numpy_include],    # Direct path to headers
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
)

setup(
    name="CythonRenderer",
    ext_modules=cythonize(ext, annotate=True),
    packages=[],
)
