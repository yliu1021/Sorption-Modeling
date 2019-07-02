import os
import glob
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from constants import *


def press(event):
    if event.key != 'q':
        exit(0)


v = '0'
density_files = glob.glob('predict_mc/results{}/density*.csv'.format(v))
grid_files = glob.glob('predict_mc/grids{}/grid*.csv'.format(v))
density_files.sort()
grid_files.sort()
for density_file, grid_file in zip(density_files, grid_files):
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
    plt.show()
