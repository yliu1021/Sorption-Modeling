import os
import glob
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from constants import *


base_dir = 'generative_model_2'

def press(event):
    if event.key != 'q':
        exit(0)


def show_grids(v):
    density_files = glob.glob(os.path.join(base_dir, 'step{}/results/density*.csv'.format(v)))
    grid_files = glob.glob(os.path.join(base_dir, 'step{}/grids/grid*.csv'.format(v)))
    density_files.sort()
    grid_files.sort()
    for density_file, grid_file in cycle(zip(density_files, grid_files)):
        df = pd.read_csv(density_file, index_col=0)
        density = df['0'][0:N_ADSORP]
        # target_density = np.linspace(0.0, 1.0, num=40)
        target_density = np.arange(40) * STEP_SIZE
        metric = (np.sum(np.absolute(density - target_density)) / 20.0)
        
        grid = np.genfromtxt(grid_file, delimiter=',')
    
        fig = plt.figure(1, figsize=(6, 8))
        fig.canvas.mpl_connect('key_press_event', press)
        fig.suptitle('{}, {}'.format('/'.join(grid_file.split('/')[-3:]), '/'.join(density_file.split('/')[-3:])))

        ax = plt.subplot(211)
        ax.pcolor(grid, cmap='Greys')
        ax.set_aspect('equal')

        ax = plt.subplot(212)
        ax.plot(df.index[0:N_ADSORP], df['0'][0:N_ADSORP])
        ax.plot(np.linspace(0, N_ADSORP, num=N_ADSORP), np.linspace(0, 1, num=N_ADSORP))
        ax.legend(['Metric: {:.4f}'.format(metric), 'Target'])
        ax.set_aspect(N_ADSORP)
        
        plt.show()
                          

def show_all_grids():
    steps = base_dir
    step_dirs = glob.glob(os.path.join(steps, 'step*'))
    step_dirs.sort(key=lambda x: int(x.split('/')[-1][4:]))
    metrics = list()
    for step_dir in step_dirs:
        step_num = int(step_dir.split('/')[-1][4:])
        print('loading step {}'.format(step_num))
        density_dir = os.path.join(step_dir, 'results')
        density_files = glob.glob(os.path.join(density_dir, 'density_*.csv'))
        if len(density_files) == 0:
            metrics.append([0])
            continue
        density_files.sort()
        densities = [np.genfromtxt(density_file, delimiter=',', skip_header=1,
                          max_rows=N_ADSORP) for density_file in density_files]
        densities = np.array(densities)
        densities[:, :, 0] /= N_ADSORP
        metric = (np.sum(np.absolute(densities[:, :, 1] - densities[:, :, 0]), axis=1) / 20.0)
        metrics.append(metric)
    plt.violinplot(metrics, showmeans=True, showextrema=True)
    plt.title('Metric distribution over train steps')
    plt.xlabel('Train step')
    plt.ylabel('Metric distribution')
    plt.ylim(0, 1)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("v", nargs="?", help="Show the grids/results of step v",
                        type=int, default=-1)
    args = parser.parse_args()
    v = args.v
    if v >= 0:
        show_grids(v)
    else:
        show_all_grids()