import os
import glob
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import imageio
import tqdm
import pdb

from constants import *

from simul_dft import *


# base_dir = 'cpp/evol_iter_grids/empty_steepest/'
# base_dir = 'swarm_grids/swarm_1pore'
base_dir = 'generative_model_2/step19/'
STOP_ITER = 12000

def press(event):
    if event.key != 'q':
        exit(0)

def make_gif():
    images = []
    for filename in sorted(os.listdir('evol_animation')):
        if filename[0] == 'g':
                images.append(imageio.imread('evol_animation/'+filename))
    imageio.mimsave('evol_animation/animated_evol.gif', images, duration=0.04)

def show_grids():
    target_grid = np.zeros(400)+1
#     target_grid[180] = 1
#     for x in range(20):
#         for y in range(20):
#             if x == 0 or x == 19 or y == 0 or y == 19:
#                 target_grid[20*x+y] = 0
    target_density = run_dft_fast(target_grid)
    target_density = target_density[:41] * 100
#     target_density = (np.arange(41)*STEP_SIZE*100).reshape(41,1)

    density_files = glob.glob(os.path.join(base_dir, 'results/density*.csv'))
    grid_files = glob.glob(os.path.join(base_dir, 'grids/grid*.csv'))

#     density_files = glob.glob(os.path.join(base_dir, 'density*.csv'))
#     grid_files = glob.glob(os.path.join(base_dir, 'grid*.csv'))
    density_files.sort()
    grid_files.sort()

#     for grid_file in grid_files:
    for density_file, grid_file in tqdm.tqdm(zip(density_files, grid_files)):
        # df = pd.read_csv(density_file, index_col=0)
        # density = df['0'][0:N_ADSORP+1]

        # # Linear curve
        # target_density = np.arange(41) * STEP_SIZE

        # # Heaviside step function
        # target_density = np.arange(41) * STEP_SIZE # heaviside
        # target_density = target_density - 0.5 # heaviside
        # target_density = np.heaviside(target_density, 0.5)

        # # Multi-step function
        # target_density = np.arange(41) * STEP_SIZE
        # target_density = np.piecewise(target_density, [target_density<=0.3, np.logical_and(target_density>0.3, target_density<=0.6), target_density>0.6], [0.3, 0.6, 1])

        # Circle functions
        # target_density = np.genfromtxt("../../../../Desktop/circle_up.csv", delimiter=",")
        # target_density = np.genfromtxt("../../../../Desktop/circle_down.csv", delimiter=",")

        # target_density = np.genfromtxt("../../../../Desktop/1pore_den.csv", delimiter=",")
        # target_density = np.genfromtxt("../../../../Desktop/1solid_den.csv", delimiter=",")
        # target_density = np.genfromtxt("../../../../Desktop/bigpore.csv", delimiter=",")

        if int(grid_file[-8:-4]) > STOP_ITER:
            break

        grid = np.genfromtxt(grid_file, delimiter=',')
        grid = grid[:, :20]

        # Use DFT to generate densities OR 
        # density = run_dft_fast(np.reshape(grid, 400))
        # density = density[:41] * 100
        # Load pre-calculated densities
        density = np.genfromtxt(density_file, delimiter=',')
        density = (density[1:42, 1]*100).reshape(41,1)

        density[-1] = 100
        target_density[-1] = 100

        metric = (np.sum(np.absolute(density - target_density)) / (N_ADSORP+1))
    
        fig = plt.figure(1, figsize=(6, 8))
        fig.canvas.mpl_connect('key_press_event', press)
        fig.suptitle('{}, {}'.format('/'.join(grid_file.split('/')[-3:]), '/'.join(density_file.split('/')[-3:])))

        ax = plt.subplot(211)
        ax.pcolor(grid, cmap='Greys_r', vmin=0.0, vmax=1.0)
        ax.set_aspect('equal')

        ax = plt.subplot(212)
        plt.xlabel('Relative humidity (%)')
        plt.ylabel('Water content (%)')
        ax.plot(np.arange(N_ADSORP+1)*STEP_SIZE*100, density[0:N_ADSORP+1])
        # ax.plot(np.arange(N_ADSORP+1)*STEP_SIZE*100, target_density)
        plt.plot([1],[1])
        # ax.legend(['Metric: {:.4f}'.format(metric), 'Target'])
        ax.set_aspect(1)
        
        # plt.savefig('evol_animation/' + grid_file[-13:-4] + '.png')
        # plt.close()

    plt.show()

print('creating images')
show_grids()
# print('\ncreating animation')
# make_gif()