import numpy as np
import random

# Constants (must match notebook)
GRID_SIZE = 800
FIRE_SPREAD_PROB = 0.3
BURN_TIME = 3
DAYS = 60
EMPTY, TREE, BURNING, ASH = 0, 1, 2, 3


def initialize_forest():
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
    forest[x, y] = BURNING
    burn_time[x, y] = 1
    return forest, burn_time


def get_neighbors(x, y):
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors


def run_one_simulation(_):
    """One wildfire run (no plotting). Returns list of burning-tree counts per day."""
    forest, burn_time = initialize_forest()
    fire_spread = []
    for day in range(DAYS):
        new_forest = forest.copy()
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1
        forest = new_forest.copy()
        fire_spread.append(int(np.sum(forest == BURNING)))
        if np.sum(forest == BURNING) == 0:
            break
    return fire_spread


def run_simulation_with_grids(grid_size=None, max_days=None):
    """
    Run one wildfire simulation and return the forest grid at each time step.
    Used for VTK export (ParaView). Optionally use a smaller grid for faster export.
    Returns: list of 2D numpy arrays (height, width), one per day until fire out or max_days.
    """
    n = grid_size or GRID_SIZE
    days = max_days if max_days is not None else DAYS
    forest = np.ones((n, n), dtype=np.int32)
    burn_time = np.zeros((n, n), dtype=np.int32)
    x0, y0 = random.randint(0, n - 1), random.randint(0, n - 1)
    forest[x0, y0] = BURNING
    burn_time[x0, y0] = 1
    grids = [forest.copy()]

    for day in range(1, days):
        new_forest = forest.copy()
        for x in range(n):
            for y in range(n):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < n and 0 <= ny < n and forest[nx, ny] == TREE:
                            if random.random() < FIRE_SPREAD_PROB:
                                new_forest[nx, ny] = BURNING
                                burn_time[nx, ny] = 1
        forest = new_forest.copy()
        grids.append(forest.copy())
        if np.sum(forest == BURNING) == 0:
            break
    return grids
