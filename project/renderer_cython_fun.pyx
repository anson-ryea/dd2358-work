# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log, cos, sin, floor

cdef (double, double, double, double) transfer_function_cython(double x) nogil:
    """Inlined C-version of the transfer function."""
    cdef double r, g, b, a
    r = (1.0 * exp(-((x - 9.0)**2) / 1.0) + 0.1 * exp(-((x - 3.0)**2) / 0.1) + 0.1 * exp(-((x + 3.0)**2) / 0.5))
    g = (1.0 * exp(-((x - 9.0)**2) / 1.0) + 1.0 * exp(-((x - 3.0)**2) / 0.1) + 0.1 * exp(-((x + 3.0)**2) / 0.5))
    b = (0.1 * exp(-((x - 9.0)**2) / 1.0) + 0.1 * exp(-((x - 3.0)**2) / 0.1) + 1.0 * exp(-((x + 3.0)**2) / 0.5))
    a = (0.6 * exp(-((x - 9.0)**2) / 1.0) + 0.1 * exp(-((x - 3.0)**2) / 0.1) + 0.01 * exp(-((x + 3.0)**2) / 0.5))
    return r, g, b, a

cdef double trilinear_interp_cython(double[:, :, :] data, double x, double y, double z) nogil:
    """Manual trilinear interpolation for high-speed voxel lookups."""
    cdef int x0 = <int>floor(x)
    cdef int y0 = <int>floor(y)
    cdef int z0 = <int>floor(z)
    cdef int x1 = x0 + 1
    cdef int y1 = y0 + 1
    cdef int z1 = z0 + 1

    # Bounds check for the interpolation box
    if x0 < 0 or x1 >= data.shape[0] or y0 < 0 or y1 >= data.shape[1] or z0 < 0 or z1 >= data.shape[2]:
        return 0.0

    cdef double xd = x - x0
    cdef double yd = y - y0
    cdef double zd = z - z0

    cdef double c00 = data[x0, y0, z0] * (1 - xd) + data[x1, y0, z0] * xd
    cdef double c01 = data[x0, y0, z1] * (1 - xd) + data[x1, y0, z1] * xd
    cdef double c10 = data[x0, y1, z0] * (1 - xd) + data[x1, y1, z0] * xd
    cdef double c11 = data[x0, y1, z1] * (1 - xd) + data[x1, y1, z1] * xd

    cdef double c0 = c00 * (1 - yd) + c10 * yd
    cdef double c1 = c01 * (1 - yd) + c11 * yd

    return c0 * (1 - zd) + c1 * zd

def render_angle_cython(double[:, :, :] datacube, int N, double angle):
    """Fused Rotation, Interpolation, and Rendering Loop."""
    cdef int Nx = datacube.shape[0]
    cdef int Ny = datacube.shape[1]
    cdef int Nz = datacube.shape[2]

    cdef double[:, :, :] image = np.zeros((N, N, 3), dtype=np.float64)
    cdef double r, g, b, a, val
    cdef double qx, qy, qz, qyR, qzR
    cdef double ix, iy, iz
    cdef int i, j, k

    # Scaling factors to map camera space to datacube index space
    cdef double center_N = N / 2.0
    cdef double center_x = Nx / 2.0
    cdef double center_y = Ny / 2.0
    cdef double center_z = Nz / 2.0

    for i in range(N):      # Depth (front to back)
        for j in range(N):  # Vertical
            for k in range(N): # Horizontal
                # 1. Map camera grid to coordinates
                qx = k - center_N
                qy = j - center_N
                qz = i - center_N

                # 2. Rotate coordinates
                qyR = qy * cos(angle) - qz * sin(angle)
                qzR = qy * sin(angle) + qz * cos(angle)

                # 3. Convert to array indices
                ix = qx + center_x
                iy = qyR + center_y
                iz = qzR + center_z

                # 4. Trilinear Interpolation
                val = trilinear_interp_cython(datacube, ix, iy, iz)

                if val <= 1e-10:
                    continue

                # 5. Blend Volume
                r, g, b, a = transfer_function_cython(log(val))
                image[j, k, 0] = a * r + (1.0 - a) * image[j, k, 0]
                image[j, k, 1] = a * g + (1.0 - a) * image[j, k, 1]
                image[j, k, 2] = a * b + (1.0 - a) * image[j, k, 2]

    return np.asarray(image).clip(0.0, 1.0)
