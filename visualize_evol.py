import os
import glob
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from constants import *

# from simul_dft import *

base_dir = 'cpp/evol_iter_grids/evol_iter_grids_1'

def press(event):
    if event.key != 'q':
        exit(0)


def show_grids():
    density_files = glob.glob(os.path.join(base_dir, 'density*.csv'))
    grid_files = glob.glob(os.path.join(base_dir, 'grid*.csv'))
    density_files.sort()
    grid_files.sort()
    for density_file, grid_file in cycle(zip(density_files, grid_files)):
        df = pd.read_csv(density_file, index_col=0)
        density = df['0'][0:N_ADSORP]

        # # Linear curve
        # target_density = np.arange(40) * STEP_SIZE

        # # Heaviside step function
        # target_density = np.arange(40) * STEP_SIZE # heaviside
        # target_density = target_density - 0.9 # heaviside
        # target_density = np.heaviside(target_density, 0.5)

        # # Multi-step function
        # target_density = np.arange(40) * STEP_SIZE
        # target_density = np.piecewise(target_density, [target_density<=0.3, np.logical_and(target_density>0.3, target_density<=0.6), target_density>0.6], [0.3, 0.6, 1])

        # Circle functions
        # target_density = np.genfromtxt("../../../../Desktop/circle_up.csv", delimiter=",")
        # target_density = np.genfromtxt("../../../../Desktop/circle_down.csv", delimiter=",")

        grid = np.genfromtxt(grid_file, delimiter=',')
        # density = run_dft_fast(np.reshape(grid, 400))

        metric = (np.sum(np.absolute(density - target_density)) / 20.0)
        
    
        fig = plt.figure(1, figsize=(6, 8))
        fig.canvas.mpl_connect('key_press_event', press)
        fig.suptitle('{}, {}'.format('/'.join(grid_file.split('/')[-3:]), '/'.join(density_file.split('/')[-3:])))

        ax = plt.subplot(211)
        ax.pcolor(grid, cmap='Greys', vmin=0, vmax=1)
        ax.set_aspect('equal')

        ax = plt.subplot(212)
        ax.plot(df.index[0:N_ADSORP], df['0'][0:N_ADSORP])
        ax.plot(np.linspace(0, N_ADSORP, num=N_ADSORP), target_density)
        ax.legend(['Metric: {:.4f}'.format(metric), 'Target'])
        ax.set_aspect(N_ADSORP)
        
        plt.show()

show_grids()