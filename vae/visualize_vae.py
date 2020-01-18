import sys
sys.path.append('..')

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from itertools import cycle
import imageio
from tensorflow.keras.models import load_model

import data
from constants import *
from simul_dft import *

model_name = 'vae_conditional'

# # Linear curve
target_density = np.linspace(0, 1, 40).reshape(1, 40,)
target_density = np.genfromtxt('../generative_model_3/step_0/results/density_0002.csv', delimiter=',')[:40].reshape(1, 40)

def press(event):
    if event.key != 'q':
        exit(0)

# def make_gif():
#     images = []
#     for filename in sorted(os.listdir('evol_animation')):
#         if filename[0] == 'g': # if grid
#                 images.append(imageio.imread('evol_animation/'+filename))
#     imageio.mimsave('evol_animation/animated_evol.gif', images, duration=0.1)

def plot_latent(models, test_data, batch_size=128, use_curve=True, save=False):
    """
    Graphs of the 2-d latent vector. Untested with higher dimensions.
    Currently shows with a linear target curve.
    """
    encoder, decoder = models
    x_test, y_test = test_data

    os.makedirs(model_name+'/grids', exist_ok=True)
    filename = os.path.join(model_name, 'vae_mean.png')

    # display a 2D plot of the digit classes in the latent space
    if use_curve:
        z_mean, _, _ = encoder.predict([x_test, y_test], batch_size=batch_size)
    else:
        z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1])
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('Latent Encoding of Test Grids')
    if save:
        plt.savefig(filename)
    plt.show()

    # display a 30x30 2D manifold of grids
    filename = os.path.join(model_name, 'grids_over_latent.png')
    n = 30
    figure = np.zeros((GRID_SIZE * n, GRID_SIZE * n))
    # linearly spaced coordinates corresponding to the 2D plot of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            if use_curve:
                x_decoded = decoder.predict([z_sample, target_density])
            else:
                x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(GRID_SIZE, GRID_SIZE)

            # digit = digit.round()

            path = os.path.join(model_name, 'grids', 'grid_{:04d}.csv'.format(i*n+j))
            np.savetxt(path, digit, fmt='%i', delimiter=',')

            # plt.figure(figsize=(10,10))
            # plt.pcolor(digit, cmap='Greys_r', vmin=0.0, vmax=1.0)
            # plt.title('{}, {}'.format(xi, yi))
            # plt.show()

            figure[i * GRID_SIZE: (i + 1) * GRID_SIZE,
                   j * GRID_SIZE: (j + 1) * GRID_SIZE] = digit

    plt.figure(figsize=(10, 10))
    start_range = GRID_SIZE // 2
    end_range = n * GRID_SIZE + start_range + 1
    pixel_range = np.arange(start_range, end_range, GRID_SIZE)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.imshow(figure, cmap='Greys_r', vmin=0.0, vmax=1.0)
    plt.title('Grids Over Latent Distribution')
    if save:
        plt.savefig(filename)
    plt.show()


def show_grids(base_dir):
    density_files = glob.glob(os.path.join(base_dir, 'results', 'density*.csv'))
    grid_files = glob.glob(os.path.join(base_dir, 'grids', 'grid*.csv'))
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


def show_grids_no_density():

    base_dir = 'vae_cnn_1'

    grid_files = glob.glob(os.path.join(base_dir, 'grid*.csv'))
    grid_files.sort()
    for i, grid_file in enumerate(grid_files):
        grid = np.genfromtxt(grid_file, delimiter=',')
        grid = grid[:, :20]
        density = run_dft_fast(np.reshape(grid, 400))
        density = density[:41]
        density[40] = 1

        metric = (np.sum(np.absolute(density - target_density)) / 20.0)

        fig = plt.figure(figsize=(10, 4))
        fig.canvas.mpl_connect('key_press_event', press)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle('Grid {}'.format(i))

        ax = plt.subplot(1, 2, 1)
        ax.clear()
        ax.set_title('Grid (Black = Solid, White = Pore)')
        ax.set_yticks(np.linspace(0, 20, 5))
        ax.set_xticks(np.linspace(0, 20, 5))
        ax.pcolor(grid, cmap='Greys_r', vmin=0.0, vmax=1.0)
        ax.set_aspect('equal')

        ax = plt.subplot(1, 2, 2)
        ax.clear()
        ax.set_title('Adsorption Curve')
        relative_humidity = np.arange(41) * STEP_SIZE
        ax.plot(relative_humidity, density, label='DFT')
        if target_density is not None:
            ax.plot(relative_humidity, target_density, label='Target')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Relative Humidity')
        ax.set_ylabel('Proportion of Pores filled')
        ax.set_aspect('equal')
        ax.legend()
       
        # # plt.savefig('visualizations/' + grid_file[-13:-4] + '.png')
        # # plt.close()

        plt.show() 
        plt.close()

# show_grids_no_density()

if __name__ == '__main__':
    encoder = load_model(model_name+'/encoder.tf')
    decoder = load_model(model_name+'/decoder.tf')
    x_test, y_test = data.get_all_data(matching='../generative_model_3')
    # x_test, y_test = data.get_all_data(matching='../generative_model_2')

    # p = np.random.permutation(len(x_test))
    # x_test  = x_test[p]
    # y_test  = y_test[p]

    # x_test = x_test[:100]
    x_test = np.reshape(x_test, [-1, GRID_SIZE, GRID_SIZE, 1])
    # y_test = y_test[:100]
    plot_latent((encoder, decoder), (x_test, y_test), use_curve=True, save=True)

    # show_grids('vae_conditional')


# print('creating images')
# show_grids_no_density()
# print('creating animation')
# make_gif()