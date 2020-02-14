
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import imageio
import sys
sys.path.append('..')
from tensorflow.keras.models import load_model

import data
import vae_models
from constants import *
from vae_options import *

# # Linear curve
# target_density = np.genfromtxt('../generative_model_3/step_0/results/density_0000.csv', delimiter=',')[:40].reshape(1, 40)

def plot_latent(model_name, encoder, test_data, batch_size=128, use_curve=True, save=False):
    """
    Graphs of the 2-d latent vector.
    Currently shows with a linear target curve.
    """
    x_test, y_test = test_data

    os.makedirs(model_name+'/grids', exist_ok=True)
    filename = os.path.join(model_name, 'vae_mean.png')

    # display a 2D plot of the latent vector
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

def plot_grids_for_curve(model_name, decoder, n=15, round_grid=True, save=False):
    """
    Graph and save the grids over the 2-d latent vector for a given curve.
    Curve is set in vae_options.py
    """
    # display a nxn 2D manifold of grids
    filename = os.path.join(model_name, 'grids_over_latent.png')
    figure = np.zeros((GRID_SIZE * n, GRID_SIZE * n))
    # linearly spaced coordinates corresponding to the 2D plot of digit classes in the latent space
    grid_x = np.linspace(-2, 2, n)
    grid_y = np.linspace(-2, 2, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict([z_sample, target_density])
            grid = x_decoded[0].reshape(GRID_SIZE, GRID_SIZE)

            if round_grid:
                grid = grid.round()

            if save:
                path = os.path.join(model_name, 'grids', 'grid_{:04d}.csv'.format(i*n+j))
                np.savetxt(path, grid, fmt='%i', delimiter=',')

            figure[i * GRID_SIZE: (i + 1) * GRID_SIZE,
                   j * GRID_SIZE: (j + 1) * GRID_SIZE] = grid

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


def make_grids(model_name, decoder, test_data, save=True):
    x_test, y_test = test_data
    z_sample = np.array([[0, 0]])

    for i, den in enumerate(y_test):
        x_decoded = decoder.predict([z_sample, den])
        grid = x_decoded[0].reshape(GRID_SIZE, GRID_SIZE)
        grid = grid.round()
        path = os.path.join(model_name, 'grids', 'grid_{:04d}.csv'.format(i))
        np.savetxt(path, grid, fmt='%i', delimiter=',')
    run_dft()


if __name__ == '__main__':
    e = load_model('vae_conditional'+'/encoder.tf')
    d = load_model('vae_conditional'+'/decoder.tf')
    x_test, y_test = data.get_all_data(matching='../generative_model_3')

    # p = np.random.permutation(len(x_test))
    # x_test  = x_test[p]
    # y_test  = y_test[p]

    # x_test = x_test[:100]
    # y_test = y_test[:100]
    # x_test = np.reshape(x_test, [-1, GRID_SIZE, GRID_SIZE, 1])


    # plot_latent(e, (x_test, y_test), save=False)
    # plot_grids(d)