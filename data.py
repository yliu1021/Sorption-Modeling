import os
import glob
from random import shuffle

import numpy as np
import matplotlib.pyplot as plt

from constants import *


def fetch_grids_from_step(base_dir, step):
    step_dir = os.path.join(base_dir, 'step{}'.format(step))
    grid_dir = os.path.join(step_dir, 'grids')
    grid_files = glob.glob(os.path.join(grid_dir, 'grid_*.csv'))
    grid_files.sort()
    return [np.genfromtxt(grid_file, delimiter=',') for grid_file in grid_files]


def fetch_density_from_step(base_dir, step):
    step_dir = os.path.join(base_dir, 'step{}'.format(step))
    density_dir = os.path.join(step_dir, 'results')
    density_files = glob.glob(os.path.join(density_dir, 'density_*.csv'))
    density_files.sort()
    return [np.genfromtxt(density_file, delimiter=',', skip_header=1,
                          max_rows=N_ADSORP) for density_file in density_files]


def make_generator_input(amount, boost_dim, as_generator=False):
    n = N_ADSORP
    def gen_diffs(mean, var, _n=n, up_to=1):
        diffs = np.clip(np.exp(np.random.normal(mean, var, _n)), -10, 10)
        return diffs / np.sum(diffs) * up_to

    def gen_func():
        anchor = np.random.uniform(0, 1)
        x = np.random.uniform(0.05, 0.95)
        ind = int(n*x)
        f_1 = np.insert(np.cumsum(gen_diffs(0, 3, ind, anchor)), 0, 0)
        f_2 = np.insert(np.cumsum(gen_diffs(0, 3, n - ind - 2, 1-anchor)), 0, 0) + anchor
        f = np.concatenate((f_1, np.array([anchor]), f_2))
        f[-1] = 1.0
        return f

    def sample_rand_input(size):
        latent_codes = np.clip(np.random.normal(loc=0.5, scale=0.25, size=(size, boost_dim)), 0, 1)
        artificial_curves = np.array([np.diff(gen_func()) for _ in range(size)])
        return [artificial_curves, latent_codes]
            
    def gen():
        while True:
            out = sample_rand_input(amount)
            yield out, out

    if as_generator:
        return gen()
    else:
        return sample_rand_input(amount)


def get_all_data_files(matching=None, get_all_files=False):
    all_files = list()
    if matching:
        base_dirs = glob.glob(matching)
        base_dirs.append('generative_model_seed_grids')
    else:
        base_dirs = glob.glob('generative_model_*')
    print('Indexing files')
    for base_dir in base_dirs:
        data_files = list()
        if get_all_files:
            step_dirs = glob.glob(os.path.join(base_dir, 'step*'))
        else:
            step_dirs = glob.glob(os.path.join(base_dir, 'step_[0-9]*'))
            step_dirs.extend(glob.glob(os.path.join(base_dir, 'step[0-9]*'))) # Backward compatability
        if len(step_dirs) == 0:
            continue
        for step_dir in step_dirs:
            grid_dir = os.path.join(step_dir, 'grids')
            density_dir = os.path.join(step_dir, 'results')
            grid_files = glob.glob(os.path.join(grid_dir, 'grid_*.csv'))
            grid_files.sort()
            density_files = glob.glob(os.path.join(density_dir, 'density_*.csv'))
            density_files.sort()
            all_files.append((grid_files, density_files))
    return all_files


_cached_grids = dict()
_cached_densities = dict()
def get_all_data(matching=None):
    data = list()
    all_files = get_all_data_files(matching=matching)
    
    num_files = len(all_files)
    for i, (grid_files, density_files) in enumerate(all_files):
        print('\rLoading {}/{}                    '.format(i+1, num_files), end='')
        grids = list()
        densities = list()
        for grid_file, density_file in zip(grid_files, density_files):
            if grid_file in _cached_grids:
                print('\rLoading {}/{} - found cache'.format(i+1, num_files), end='')
                grids.append(_cached_grids[grid_file])
            else:
                print('\rLoading {}/{} - caching'.format(i+1, num_files), end='')
                grid = np.genfromtxt(grid_file, delimiter=',')
                grids.append(grid)
                _cached_grids[grid_file] = grid
            if density_file in _cached_densities:
                print('\rLoading {}/{} - found cache'.format(i+1, num_files), end='')
                densities.append(_cached_densities[density_file])
            else:
                print('\rLoading {}/{} - caching'.format(i+1, num_files), end='')
                density = np.genfromtxt(density_file, delimiter=',', skip_header=1, max_rows=N_ADSORP)
                densities.append(density)
                _cached_densities[density_file] = density
        data.extend(zip(grids, densities))
    print()

    shuffle(data)
    all_grids, all_densities = zip(*data)
    all_grids = np.array(all_grids)
    all_densities = np.array(all_densities)
    all_diffs = np.diff(all_densities, axis=1, append=1.0)
    all_metrics = all_diffs[:, :, 1]
    return (all_grids, all_metrics)


if __name__ == '__main__':
    grids, metrics = get_all_data()
    plt.hist(metrics, bins=100)
    plt.show()
    data = list(zip(grids,metrics))
    data.sort(key=lambda x: x[1])
    
    for grid, metric in data:
        print(metric)
        plt.pcolor(grid, cmap='Greys', vmin=0.0, vmax=1.0)
        print(grid)
        plt.show()