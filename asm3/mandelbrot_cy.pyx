# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Cython-optimized Mandelbrot set computation.
Uses typed memoryviews and C-level types for analysis and speed.
"""
import numpy as np
cimport numpy as cnp

# Typed mandelbrot: avoid Python complex; use two doubles for analysis/optimization
cdef int mandelbrot(double cr, double ci, int max_iter) nogil:
    """Computes iterations before divergence. Uses real arithmetic (no Python complex)."""
    cdef double zr = 0.0
    cdef double zi = 0.0
    cdef double zr2, zi2
    cdef int n
    for n in range(max_iter):
        if zr * zr + zi * zi > 4.0:  # |z|^2 > 4 <=> |z| > 2
            return n
        zr2 = zr * zr - zi * zi + cr
        zi2 = 2.0 * zr * zi + ci
        zr, zi = zr2, zi2
    return max_iter


def mandelbrot_set(int width, int height, double x_min, double x_max,
                   double y_min, double y_max, int max_iter=100):
    """Generates the Mandelbrot set using typed memoryviews (no NumPy indexing in hot loop)."""
    # Allocate buffers and get typed memoryviews
    cdef double[:] x_vals = np.linspace(x_min, x_max, width, dtype=np.float64)
    cdef double[:] y_vals = np.linspace(y_min, y_max, height, dtype=np.float64)
    cdef double[:, :] image = np.zeros((height, width), dtype=np.float64)

    cdef int i, j
    with nogil:
        for i in range(height):
            for j in range(width):
                image[i, j] = mandelbrot(x_vals[j], y_vals[i], max_iter)

    return np.asarray(image)
