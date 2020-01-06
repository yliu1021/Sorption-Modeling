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

import data
import models
from constants import *

def start_training(**kwargs):
    base_dir = kwargs.get('base_dir', 'generative_model_default')
    step = 0
    model_save_dir = os.path.join(base_dir, 'model_saves')
    decoder_model_save_file = os.path.join(model_save_dir, 'decoder_step_{}.hdf5'.format(step))
    model_save_file = os.path.join(model_save_dir, 'vae_step_{}.hdf5'.format(step))

    train_grids, train_curves = data.get_all_data(matching=base_dir, augment_factor=20)
    train_grids = train_grids[0:1800:10]
    train_curves = train_curves[0:1800:10]

    grid_inp = Input(shape=(GRID_SIZE, GRID_SIZE), name='grid')
    curve_inp = Input(shape=(N_ADSORP,), name='curve')
    encoder_model, decoder_model = models.make_vae_encoder(**kwargs), models.make_vae_decoder(**kwargs)
    outputs = decoder_model([curve_inp, encoder_model(grid_inp)[2]])
    vae_model = Model(inputs=[grid_inp, curve_inp], outputs=outputs, name='vae')

    # lc_inp = Input(shape=(boost_dim,), name='latent_code')
    # curve_inp = Input(shape=(N_ADSORP,), name='target_curve')
    # decoder_out = decoder_model([lc_inp, curve_inp])
    # encoder_out = encoder_model(decoder_out)

    # Define our loss function and compile our model
    learning_rate = 10**-2
    optimizer = SGD(learning_rate, clipnorm=1.0)
    # optimizer = Adam(learning_rate, clipnorm=1.0)
    vae_model.compile(optimizer, loss='binary_crossentropy', metrics=['mae', models.worst_abs_loss])
    vae_model.fit(x=[train_grids, train_curves], y=train_grids, batch_size=64, epochs=50, validation_split=0.1)
    vae_model.save(model_save_file, include_optimizer=False)
    decoder_model.save(decoder_model_save_file, include_optimizer=False)

start_training(base_dir='vae_model_0')