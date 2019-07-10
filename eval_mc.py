import numpy as np
import math
import random
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from constants import *

def plot_stats(statset, path, param):
    fig, ax = plt.subplots()
    xvals = np.reshape(np.arange(N_MC_ITER+1), (N_MC_ITER+1,))
    ax.plot(xvals[:-1], statset[:-1,param])
    ax.set(#xlabel='Relative Humidity (%)', ylabel='Density',
         title=('Old', 'New', 'Diff', 'Prob', 'Sigma', 'Avg Prob')[param])
    # plt.legend(('Old', 'New', 'Diff', 'Prob', 'Sigma', ), loc='lower right')
    fig.savefig(path)
    plt.close(fig)

def animate_grids(gridset):
    fig, ax = plt.subplots()
    ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))
    grid = gridset[0].reshape((GRID_SIZE, GRID_SIZE))
    quad = ax.pcolormesh(grid, cmap='Greys')
    def update(grid):
        quad.set_array(grid)
        return (quad,)
    ani = animation.FuncAnimation(fig, update, frames=gridset, 
                                  interval=20, blit=True, repeat=False)
    plt.show(fig)

def plot_stat_summary():
    statset = np.loadtxt(MC_GRID_DIR + '/results.csv', dtype='float', delimiter=',')
    for i in range(N_RES_PARAMS):
        plot_stats(statset, MC_GRID_DIR + '/results%02d.png'%i, i)

def plot_animation():
    gridset = np.loadtxt(MC_GRID_DIR + '/gridset.csv', dtype='float', delimiter=',')
    animate_grids(gridset)


if __name__ == '__main__':
    plot_animation()


