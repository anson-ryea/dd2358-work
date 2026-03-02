"""
Export 2D grid time series to VTK Legacy format for ParaView.
One file per time step; ParaView can open the series as an animation.
"""
import os
import numpy as np


def write_vtk_structured_points(path, grid, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0)):
    """
    Write a single 2D grid to a VTK Legacy ASCII file (STRUCTURED_POINTS).
    grid: 2D array (height, width); will be written as 3D with size (height, width, 1).
    """
    h, w = grid.shape
    n_points = h * w
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("wildfire state\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write("DIMENSIONS {} {} 1\n".format(w, h))
        f.write("ORIGIN {} {} {}\n".format(origin[0], origin[1], origin[2]))
        f.write("SPACING {} {} {}\n".format(spacing[0], spacing[1], spacing[2]))
        f.write("POINT_DATA {}\n".format(n_points))
        f.write("SCALARS state int 1\n")
        f.write("LOOKUP_TABLE default\n")
        # VTK order: x varies fastest (columns), then y (rows), then z. So row-major.
        flat = np.asarray(grid, dtype=np.int32).flatten(order="C")
        for i in range(0, n_points, 10):
            chunk = flat[i : i + 10]
            f.write(" ".join(map(str, chunk)) + "\n")
    return path


def export_wildfire_to_vtk(grids, out_dir="vtk_output", prefix="forest"):
    """
    Export a list of 2D grids to VTK files: forest_000.vtk, forest_001.vtk, ...
    grids: list of 2D numpy arrays (one per time step)
    Returns: list of file paths
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, grid in enumerate(grids):
        name = "{}_{:05d}.vtk".format(prefix, i)
        path = os.path.join(out_dir, name)
        write_vtk_structured_points(path, grid)
        paths.append(path)
    return paths
