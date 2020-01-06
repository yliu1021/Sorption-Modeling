import os
import shutil
import glob
import sys
from multiprocessing import Pool
from random import randint, shuffle, sample
import json
import argparse
import math

import numpy as np
np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 1000
import pandas as pd
# os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse, binary_crossentropy

import data
# import models
from constants import *

import pdb

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


original_dim = GRID_SIZE * GRID_SIZE
# x_train, y_train = data.get_all_data(matching='vae_model_0')
x_train, y_train = data.get_all_data(matching='generative_model_2')

pdb.set_trace()

x_train = np.reshape(x_train, [-1, original_dim])

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 5
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # help_ = "Load h5 model trained weights"
    # parser.add_argument("-w", "--weights", help=help_)
    # help_ = "Use mse loss instead of binary cross entropy (default)"
    # parser.add_argument("-m",
    #                     "--mse",
    #                     help=help_, action='store_true')
    # args = parser.parse_args()
    models = (encoder, decoder)

    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = binary_crossentropy(inputs, outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    optimizer = Adam(10**-2)
    vae.compile(optimizer=optimizer)

    # train the autoencoder
    vae.fit(x_train, epochs=epochs, batch_size=batch_size)
    # vae.save_weights('vae_mlp_mnist.h5')