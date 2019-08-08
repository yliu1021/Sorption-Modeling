import os
import glob
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from itertools import cycle

from tensorflow.keras.models import load_model

import models
from constants import *


base_dir = 'generative_model_seed_grids'
step_index = -1
index = 0
def press(event):
    global index
    global step_index
    if event.key == 'd':
        index += 1
    elif event.key == 'a':
        index -= 1
    elif event.key == 'w':
        step_index += 1
    elif event.key == 's':
        step_index -= 1
    if event.key == 'q':
        exit(0)


def show_grids(v):
    global index
    global step_index

    all_files = list()
    step_dirs = glob.glob(os.path.join(base_dir, 'step*'))
    step_dirs.sort()
    for i, step_dir in enumerate(step_dirs):
        if 'step_{}'.format(v) in step_dir or 'step{}'.format(v) in step_dir:
            step_index = i
        grid_files = glob.glob(os.path.join(step_dir, 'grids/grid_*.csv'.format(v)))
        density_files = glob.glob(os.path.join(step_dir, 'results/density_*.csv'.format(v)))
        target_density_files = glob.glob(os.path.join(step_dir, 'target_densities/artificial_curve_*.csv'.format(v)))
        grid_files.sort()
        density_files.sort()
        target_density_files.sort()
        if len(target_density_files) == 0:
            all_step_files = list(zip(grid_files, density_files))
        else:
            all_step_files = list(zip(grid_files, density_files, target_density_files))
        all_files.append(all_step_files)

    predictor_model = None
    print('Attempting to load predictor model...')
    try:
        predictor_model = load_model(os.path.join(base_dir, 'model_saves/predictor_step_{}.hdf5'.format(v)),
                                     custom_objects={'binary_sigmoid': models.binary_sigmoid})
        print('Loaded successfully')
    except:
        print("Couldn't load predictor_model")

    fig = plt.figure(figsize=(10, 4))
    fig.canvas.mpl_connect('key_press_event', press)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    while True:
        s = step_index % len(all_files)
        i = index % len(all_files[s])
        files = all_files[s][i]
        target_density_file = None
        if len(files) == 2:
            grid_file, density_file = files
        else:
            grid_file, density_file, target_density_file = files

        df = pd.read_csv(density_file, index_col=0)
        relative_humidity = np.arange(41) * STEP_SIZE
        
        grid = np.genfromtxt(grid_file, delimiter=',')
        density = np.append(df['0'][:N_ADSORP], 1)
        target_density = None
        if target_density_file:
            target_density = np.insert(np.genfromtxt(target_density_file), 0, 0)
            target_density = np.cumsum(target_density)
        predicted_density = None
        if predictor_model:
            predicted_density = np.insert(predictor_model.predict(np.array([grid]))[0], 0, 0)
            predicted_density = np.cumsum(predicted_density)

        fig.suptitle('Step {}, Grid {}'.format(s, i))

        ax = plt.subplot(1, 2, 1)
        ax.clear()
        ax.set_title('Grid (Black = Solid, White = Pore)')
        ax.set_yticks(np.linspace(0, 20, 5))
        ax.set_xticks(np.linspace(0, 20, 5))
        ax.pcolor(1 - grid, cmap='Greys', vmin=0.0, vmax=1.0)
        ax.set_aspect('equal')

        ax = plt.subplot(1, 2, 2)
        ax.clear()
        ax.set_title('Adsorption Curve')
        ax.plot(relative_humidity, density, label='DFT')
        if predicted_density:
            ax.plot(relative_humidity, predicted_density, label='Predictor')
        if target_density:
            ax.plot(relative_humidity, target_density, label='Target')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Relative Humidity')
        ax.set_ylabel('Proportion of Pores filled')
        ax.set_aspect('equal')
        ax.legend()
        
        plt.show()
        plt.waitforbuttonpress(timeout=-1)


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