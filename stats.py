import numpy as np
from scipy.ndimage import measurements
import math
import os
import sys

from constants import *

def aggregate_grids(n_grids, grid_dir, set_dir, set_name):
    gridset = np.zeros(n_grids, GRID_SIZE * GRID_SIZE)
    for i in range(n_grids):
        path = os.path.join(grid_dir, 'grid_%04d.csv'%i)
        grid = np.loadtxt(path, dtype='int', delimiter=',')
        gridset[i,:] = grid.reshape((1, GRID_SIZE*GRID_SIZE))
    path = os.path.join(set_dir, set_name + '.csv')
    np.savetxt(path, gridset, delimiter=',')

def aggregate_stats(n_grids, grid_dir, result_dir, hys_dir, hys_name,
                    xfunc, yfunc):
    hys = np.zeros((n_grids, 2))
    for i in range(n_grids):
        path = os.path.join(grid_dir, 'grid_%04d.csv'%i)
        grid = np.loadtxt(path, dtype='int', delimiter=',')
        path = os.path.join(result_dir, 'density_%04d.csv'%i)
        den = np.loadtxt(path, dtype='float', delimiter=',', skiprows=1)
        hys[i,0] = xfunc(grid, den)
        hys[i,1] = yfunc(grid, den, xfunc)
    path = os.path.join(hys_dir, hys_name + '.csv')
    np.savetxt(path, hys, delimiter=',')

def aggregate_feature(n_sets, grid_name, result_name, feature_dir,
                      feature_name, xfunc):
    for i in range(1, n_sets + 1):
        path = os.path.join(feature_dir, grid_name + '%02d.csv'%i)
        gridset = np.genfromtxt(path, delimiter=',')
        path = os.path.join(feature_dir, result_name + '%02d.csv'%i)
        resultset = np.genfromtxt(path, delimiter=',')
        n_grids = gridset.shape[0]
        feature = np.zeros((n_grids, 2))
        for j in range(n_grids):
            feature[j,0] = xfunc(np.reshape(gridset[j,:], (GRID_SIZE, GRID_SIZE)), None)
            feature[j,1] = resultset[j,1]
        path = os.path.join(feature_dir, feature_name + '%02d.csv'%i)
        np.savetxt(path, feature, delimiter=',')

def pore_size(grid, den):
    return np.sum(grid)

def sqrt_pore_size(grid, den):
    return sqrt(pore_size(grid, den))

def perimeter(grid, den):
    count = 0
    for i in range(n_size):
        for j in range(n_size):
            if grid[i, j] == 1 and (grid[i-1, j] == 0 or grid[i, j-1] == 0 or 
                grid[i+1, j] == 0 or grid[i, j+1] == 0):
                count += 1
    return count

def perimeter_over_area(grid, den):
    return perimeter(grid, den) / pore_size(grid, den)

def polar_first_moment(grid, den):
    ux, uy = measurements.center_of_mass(grid)
    moment = 0
    for i in range(n_size):
        for j in range(n_size):
            if grid[i, j] == 1:
                moment += math.sqrt((i - ux)**2 + (j - uy)**2)
    return moment

def moment_of_inertia(grid, den):
    ux, uy = measurements.center_of_mass(grid)
    moment = 0
    for i in range(n_size):
        for j in range(n_size):
            if grid[i, j] == 1:
                moment += (i - ux)**2 + (j - uy)**2
    return moment

def hysteresis(grid, den, xfunc):
    rev = np.flipud(den)
    return np.sum(rev[0:N_ADSORP,1] - den[0:N_ADSORP,1])

def sorpdiff(grid, den, xfunc):
    return den[N_ADSORP,1] - den[N_ITER,1]

def linedist(grid, den, xfunc):
    dist = np.absolute(den[0:N_ADSORP,1] - (den[0:N_ADSORP,0] * STEP_SIZE))
    return np.sum(dist)

def loglinedist(grid, den, xfunc):
    dist = np.absolute(den[0:N_ADSORP,1] - (den[0:N_ADSORP,0] * STEP_SIZE))
    return np.log(np.sum(dist))

def hysteresis_over(grid, den, xfunc):
    return hysteresis(grid, den) / xfunc(grid, den)

def find_max_indices(n_top, gridset, resultset):
    print(resultset.shape)
    indices = np.argpartition(resultset[:,1], -n_top, axis=None)[-n_top:]
    return indices

def find_min_indices(n_top, gridset, resultset):
    print(resultset.shape)
    indices = np.argpartition(resultset[:,1], n_top, axis=None)[:n_top]
    return indices

# TODO: convert this into a dict
FEATURES = [
    # ('a', pore_size, hysteresis, 'Pore Size', 'Hysteresis'),
    # ('b', pore_size, hysteresis_over, 'Pore Size', 'Hysteresis/Pore Size'),
    # ('c', pore_size, hysteresis, '(Pore Size)^.5', 'Hysteresis'),
    # ('d', pore_size, hysteresis_over, '(Pore Size)^.5', 'Hysteresis/(Pore Size)^.5'),
    # ('e', perimeter, hysteresis, 'Perimeter', 'Hysteresis'),
    # ('f', perimeter_over_area, hysteresis, 'Perimeter/Area', 'Hysteresis'),
    # ('g', polar_first_moment, hysteresis, 'Polar First Moment', 'Hysteresis'),
    # ('h', polar_first_moment, hysteresis_over, 'Polar First Moment', 'Hysteresis/Polar First Moment'),
    # ('i', moment_of_inertia, hysteresis, 'Moment of Inertia', 'Hysteresis'),
    # ('j', moment_of_inertia, hysteresis_over, 'Moment of Inertia', 'Hysteresis/Moment of Inertia'),
    # ('k', pore_size, sorpdiff, 'Pore Size', 'Difference'),
    ('l', pore_size, linedist, 'Pore Size', 'Difference'),
    ('m', pore_size, loglinedist, 'Pore Size', 'Log Difference'),
]

def aggregate_all(hys_dir=HYS_DIR, grid_base_dir=GRID_DIR, result_base_dir=RESULT_DIR):
    os.makedirs(hys_dir, exist_ok=True)
    for dataset in DATASETS:
        grid_dir = grid_base_dir + dataset[0]
        result_dir = result_base_dir + dataset[0]
        n_grids = dataset[1]
        for feature in FEATURES:
            hys_name = 'hys' + dataset[0] + feature[0]
            aggregate_stats(n_grids, grid_dir, result_dir, hys_dir, hys_name,
                            feature[1], feature[2])


if __name__ == '__main__':
    aggregate_all(MC_HYS_DIR, MC_SRC_GRID_DIR, MC_SRC_RESULT_DIR)
    # aggregate_feature(N_STEPS, 'train_grids', 'train_results', TRAIN_SET_DIR,
    #                   'data_m', pore_size)
    # aggregate_feature(N_STEPS, 'gridset', 'resultset', RESULT_SET_DIR,
    #                   'data_m', pore_size)



