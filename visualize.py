import os
import glob
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from constants import *


def press(event):
    if event.key != 'q':
        exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("v", help="Show the grids/results of step v",
                        type=int)
    args = parser.parse_args()
    v = args.v
    density_files = glob.glob('generative_model/step{}/results/density*.csv'.format(v))
    grid_files = glob.glob('generative_model/step{}/grids/grid*.csv'.format(v))
    density_files.sort()
    grid_files.sort()
    for density_file, grid_file in cycle(zip(density_files, grid_files)):
        df = pd.read_csv(density_file, index_col=0)
    
        grid = np.genfromtxt(grid_file, delimiter=',')
    
        fig = plt.figure(1)
        fig.canvas.mpl_connect('key_press_event', press)

        ax = plt.subplot(211)
        ax.title.set_text('Grid {}'.format(grid_file))
        ax.pcolor(grid, cmap='Greys')
        ax.set_aspect('equal')

        ax = plt.subplot(212)
        ax.title.set_text('Density {}'.format(density_file))
        ax.plot(df.index[0:N_ADSORP], df['0'][0:N_ADSORP])
        ax.plot(np.linspace(0, N_ADSORP, num=N_ADSORP), np.linspace(0, 1, num=N_ADSORP))
        ax.set_aspect(N_ADSORP)
        plt.show()
