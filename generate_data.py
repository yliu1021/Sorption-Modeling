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


def get_grids_and_densities_from_stepdir(dir):
    grid_dir = os.path.join(dir, 'grids')
    density_dir = os.path.join(dir, 'results')
    grid_files = glob.glob(os.path.join(grid_dir, 'grid_*.csv'))
    density_files = glob.glob(os.path.join(density_dir, 'density_*.csv'))
    return zip(grid_files, density_files)


def get_files_from_base_dir(dir):
    data_files = list()
    step_dirs = glob.glob(os.path.join(dir, 'step*'))
    for step in step_dirs:
        data_files.extend(get_grids_and_densities_from_stepdir(step))
    return data_files


def get_generator(batch_size=64, val_split=0.3):
    data_files = list()

    base_dirs = glob.glob('generative_model_*')
    for base_dir in base_dirs:
        data_files.extend(get_files_from_base_dir(base_dir))
    
    shuffle(data_files)
    val_split = int(val_split * len(data_files))
    val_data_files, data_files = data_files[:val_split], data_files[val_split:]

    def batch(x):
        # batch list x
        batched_x = list()
        for i in range(0, len(x), batch_size):
            batched_x.append(x[i:i+batch_size])
        return batched_x
    
    val_data_files, data_files = batch(val_data_files), batch(data_files)
    
    def make_generator(x):
        # convert list x into a generator (with some preprocessing)
        def preprocess(data):
            grid_file, density_file = data
            grid = np.genfromtxt(grid_file, delimiter=',')
            density = np.genfromtxt(density_file, delimiter=',', skip_header=1, max_rows=N_ADSORP)
            density[:, 0] /= N_ADSORP
            metric = np.sum(np.absolute(density[:, 1] - density[:, 0]), axis=0) / 20.0
            return grid, metric
        
        for batch in x:
            b = map(preprocess, batch)
            b = zip(*b)
            yield b
    
    val_generator = make_generator(val_data_files)
    train_generator = make_generator(data_files)
    
    return ((train_generator, len(data_files)), (val_generator, len(val_data_files)))


def get_all_data_files():
    all_files = list()
    base_dirs = glob.glob('generative_model_*')
    print('Indexing files')
    for base_dir in base_dirs:
        data_files = list()
        step_dirs = glob.glob(os.path.join(base_dir, 'step*'))
        for step_dir in step_dirs:
            grid_dir = os.path.join(step_dir, 'grids')
            density_dir = os.path.join(step_dir, 'results')
            grid_files = glob.glob(os.path.join(grid_dir, 'grid_*.csv'))
            grid_files.sort()
            density_files = glob.glob(os.path.join(density_dir, 'density_*.csv'))
            density_files.sort()
            all_files.append((grid_files, density_files))
    return all_files


def get_all_data():
    data = list()
    all_files = get_all_data_files()
    
    num_files = len(all_files)
    for i, (grid_files, density_files) in enumerate(all_files):
        print('\rLoading {}/{}'.format(i+1, num_files), end='')
        grids = [np.genfromtxt(grid_file, delimiter=',') for grid_file in grid_files]
        densities = [np.genfromtxt(density_file, delimiter=',', skip_header=1,
                                   max_rows=N_ADSORP) for density_file in density_files]
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
        plt.pcolor(grid, cmap='Greys')
        print(grid)
        plt.show()