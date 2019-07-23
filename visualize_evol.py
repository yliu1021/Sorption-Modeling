import os
import glob
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from constants import *


base_dir = 'cpp/evol_iter_grids/'

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

show_grids()