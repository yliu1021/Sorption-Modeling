import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')

from constants import *

model_name = 'vae_conditional'

# # Linear curve
target_density = np.linspace(0, 1, 40).reshape(1, 40,)
target_density = np.genfromtxt('../generative_model_3/step_0/results/density_0001.csv', delimiter=',')[:40].reshape(1, 40)


def press(event):
    if event.key != 'q':
        exit(0)

def show_grids(base_dir, save=False):
    density_files = glob.glob(os.path.join(base_dir, 'results', 'density*.csv'))
    grid_files = glob.glob(os.path.join(base_dir, 'grids', 'grid*.csv'))
    if len(density_files) != len(grid_files):
        if os.system('../cpp/fast_dft ./{}/'.format(model_name)):
            print('Failed to execute dft, no results found')
        density_files = glob.glob(os.path.join(base_dir, 'results', 'density*.csv'))
    density_files.sort()
    grid_files.sort()
    for density_file, grid_file in zip(density_files, grid_files):
        density = np.genfromtxt(density_file, delimiter=',')[:41]
        density[40] = 1

        grid = np.genfromtxt(grid_file, delimiter=',')
        grid = grid[:, :20]

        td = np.reshape(target_density, (40, ))
        td = np.append(target_density, 1)
        metric = (np.sum(np.absolute(density - td)) / 20.0)
    
        fig = plt.figure(1, figsize=(6, 8))
        fig.canvas.mpl_connect('key_press_event', press)
        fig.suptitle('{}, {}'.format('/'.join(grid_file.split('/')[-3:]), '/'.join(density_file.split('/')[-3:])))

        ax = plt.subplot(211)
        ax.pcolor(grid, cmap='Greys_r', vmin=0.0, vmax=1.0)
        ax.set_aspect('equal')

        ax = plt.subplot(212)
        ax.plot(np.linspace(0, 1, N_ADSORP+1), density)
        
        ax.plot(np.linspace(0, 1, N_ADSORP+1), td)
        plt.plot([1],[1])
        ax.legend(['Metric: {:.4f}'.format(metric), 'Target'])
        ax.set_aspect('equal')
        
        # plt.savefig('evol_animation/' + grid_file[-13:-4] + '.png')
        # plt.close()

        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", help="Show the grids/results of model name", default=model_name)
    args = parser.parse_args()
    name = args.name

    show_grids(name)