# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py
# take a look at https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/

import sys
sys.path.append('..')

import argparse
import os
import math
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 1000
# os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

import data
from constants import *

import pdb

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# x_train, y_train = data.get_all_data(matching='vae_cnn_data')
x_train, y_train = data.get_all_data(matching='../generative_model_2')

num_test = 100
x_test = x_train[:num_test]
y_test = y_train[:num_test]
x_train = x_train[num_test:]
y_train = y_train[num_test:]

x_train = np.reshape(x_train, [-1, GRID_SIZE, GRID_SIZE, 1])
x_test = np.reshape(x_test, [-1, GRID_SIZE, GRID_SIZE, 1])

# network parameters
original_dim = GRID_SIZE * GRID_SIZE
input_shape = (GRID_SIZE, GRID_SIZE, 1)
batch_size = 128
kernel_size = 3
filters = 16
latent_dim = 2
epochs = 5 

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(2):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x1 = Dense(128, activation='relu')(latent_inputs)

curve_inputs = Input(shape=(N_ADSORP,), name='curve_inp')
x2 = Dense(128, activation='relu')(curve_inputs)

decoder_inputs = concatenate([x1, x2], axis=1)
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(decoder_inputs)

x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(2):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    filters //= 2

outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model([latent_inputs, curve_inputs], outputs, name='decoder')

# instantiate VAE model
outputs = decoder([encoder(inputs)[2], curve_inputs])
vae = Model([inputs, curve_inputs], outputs, name='vae')

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):

    target_curve = np.linspace(0, 1, 40).reshape(1, 40,)
    # target_curve = np.zeros(40).reshape(1,40,)

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1])
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 20 
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict([z_sample, target_curve])
            digit = x_decoded[0].reshape(digit_size, digit_size)

            path = os.path.join('vae_cnn_1', 'grid_{:04d}.csv'.format(i*n+j))
            np.savetxt(path, digit, fmt='%i', delimiter=',')

            # plt.figure(figsize=(10,10))
            # plt.pcolor(digit, cmap='Greys_r', vmin=0.0, vmax=1.0)
            # plt.title('{}, {}'.format(xi, yi))
            # plt.show()

            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    args = parser.parse_args()

    models = (encoder, decoder)
    data = (x_train, y_train)

    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))

    reconstruction_loss *= GRID_SIZE * GRID_SIZE
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit([x_train, y_train], epochs=epochs, batch_size=batch_size)
        # decoder.save_weights('vae_cnn_decoder.h5')
        # vae.save_weights('vae_cnn.h5')

    plot_results(models, (x_test, y_test), batch_size=batch_size, model_name="vae_cnn_1")