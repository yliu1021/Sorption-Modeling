
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
from constants import *
from vae_options import *

# # Linear curve
target_density = np.genfromtxt('../generative_model_3/step_0/results/density_0000.csv', delimiter=',')[:40].reshape(1, 40)

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

            digit = digit.round()

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