import numpy as np
from scipy.ndimage import measurements
import math
import random
import os
import sys

from constants import *

def diff_grid(grid1, grid2):
    diff = np.abs(grid1 - grid2)
    return np.sum(diff)

def generate_grid(n_filled, thresh, n_var, n_size):
    grid = np.zeros((n_size, n_size))
    # Dump a bunch of water into the grid
    choices = np.random.permutation((n_size - 2*thresh) * (n_size - 2*thresh))
    to_fill = 0
    choice_i = 0
    n_to_fill = n_filled + random.randint(-n_var, n_var)
    while to_fill < n_to_fill:
        x = choices[choice_i]//(n_size - 2*thresh) + thresh
        y = choices[choice_i] % (n_size - 2*thresh) + thresh
        if grid[x, y] == 0:
            grid[x, y] = 1
            to_fill += 1
        choice_i += 1

    # Keep the largest connected component
    labels, n_clusters = measurements.label(grid)
    sizes = np.bincount(labels.flatten())
    sizes[0] = 0  # don't count solid regions
    max_cluster = np.argmax(sizes)
    for i in range(n_size * n_size):
        x, y = (i//n_size, i % n_size)
        if labels[x, y] != max_cluster:
            grid[x, y] = 0
    return grid

def alt_generate_grid(n_filled, thresh, n_var, n_size):
    grid = np.zeros((n_size, n_size))
    to_fill = 0
    n_to_fill = n_filled + random.randint(-n_var, n_var)

    # Start at a random point on the grid
    grid[n_size // 2, n_size // 2] = 1
    while to_fill < n_to_fill:
        x = random.randint(thresh, n_size - thresh - 1)
        y = random.randint(thresh, n_size - thresh - 1)
        if grid[x, y] == 0 and (grid[x-1, y] == 1 or grid[x+1, y] == 1
            or grid[x, y-1] == 1 or grid[x, y+1] == 1):
            to_fill += 1
            grid[x, y] = 1
    return grid

def square_generate_grid(n_offset, n_size):
    grid = np.zeros((n_size, n_size))
    for i in range(n_size * n_size):
        x, y = (i//n_size, i % n_size)
        if x >= n_offset and x < n_size - n_offset:
            if y >= n_offset and y < n_size - n_offset:
                grid[x, y] = 1
    return grid

def rand_sq_generate_grid(n_filled, thresh, n_var, n_size):
    n_offset = random.randint(1, 9)
    return square_generate_grid(n, n_offset, n_size)

def transform_grid(n, n_size, grid, output, GRID_DIR, save=True):
    raw = grid
    if output and save:
        path = os.path.join(GRID_DIR, 'raw_%04d.csv'%n)
        np.savetxt(path, grid, fmt='%n', delimiter=',')

    # Transforms the grid to control translation/rotation/symmetry
    ux, uy = measurements.center_of_mass(grid)
    shift_x = int(round((n_size - 1)/2 - ux))
    shift_y = int(round((n_size - 1)/2 - uy))
    grid = np.roll(grid, (shift_x, shift_y), axis=(0, 1))
    # Ensure grid is surrounded by a solid border
    while (np.sum(grid[0, :]) > 0 or np.sum(grid[n_size - 1, :]) > 0 or
        np.sum(grid[:, 0]) > 0 or np.sum(grid[:, n_size - 1]) > 0):
        grid[0, :] = 0
        grid[n_size - 1, :] = 0
        grid[:, 0] = 0
        grid[:, n_size - 1] = 0
        ux, uy = measurements.center_of_mass(grid)
        shift_x = int(round((n_size - 1)/2 - ux))
        shift_y = int(round((n_size - 1)/2 - uy))
        grid = np.roll(grid, (shift_x, shift_y), axis=(0, 1))
    shift = grid
    if output and save:
        path = os.path.join(GRID_DIR, 'shift_%04d.csv'%n)
        np.savetxt(path, grid, fmt='%n', delimiter=',')

    # Flip the grid based on its orientation
    ux, uy = measurements.center_of_mass(grid)
    weights_x = np.zeros((n_size, n_size))
    for i in range(n_size):
        weights_x[i, :] = abs(i - ux) ** 2
    weights_y = np.ones((n_size, n_size))
    for i in range(n_size):
        weights_y[:, i] = abs(i - uy) ** 2
    var_x = np.average(grid, weights=weights_x)
    var_y = np.average(grid, weights=weights_y)
    if var_x > var_y:
        grid = np.rot90(grid)
    rot = grid
    if output and save:
        path = os.path.join(GRID_DIR, 'rot_%04d.csv'%n)
        np.savetxt(path, grid, fmt='%n', delimiter=',')

    # Flip the grid to a corner
    ux, uy = measurements.center_of_mass(grid)
    if ux > (n_size - 1)/2:
        grid = np.fliplr(grid)
    if uy > (n_size - 1)/2:
        grid = np.flipud(grid)
    if save:
        path = os.path.join(GRID_DIR, 'grid_%04d.csv'%n)
        np.savetxt(path, grid, fmt='%i', delimiter=',')
    return raw, shift, rot, grid

def validate_grid(grid, thresh, n_size):
    if grid.shape != (n_size, n_size):
        grid = np.reshape(grid, (n_size, n_size))
    labels, n_clusters = measurements.label(grid)
    if n_clusters != 1: # don't ruin connected components
        return False
    # don't intrude on the boundary 
    grid[thresh:n_size-thresh,thresh:n_size-thresh] = 0
    return np.amax(grid) == 0

def generate_set(grid_dir=GRID_DIR):
    # method = alt_generate_grid
    os.makedirs(grid_dir, exist_ok=True)
    for i in range(1000):
        transform_grid(i, GRID_SIZE, generate_grid(
            N_FILLED, THRESH, N_VAR, GRID_SIZE), False, grid_dir+'1', True)
    for i in range(1000):
        transform_grid(i, GRID_SIZE, alt_generate_grid(
            N_FILLED, THRESH, N_VAR, GRID_SIZE), False, grid_dir+'2', True)
    for i in range(9):
        transform_grid(i, GRID_SIZE, square_generate_grid(
            i + 1, GRID_SIZE), False, grid_dir+'3', True)


if __name__ == '__main__':
    generate_set(MC_SRC_GRID_DIR)



