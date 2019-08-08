import os
import shutil
import glob
import sys
from multiprocessing import Pool
from random import randint, shuffle
import json
import argparse

import numpy as np
np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 1000
import pandas as pd
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm

import data
import models
from constants import *


def KL_divergence(p, q):
    epsilon = 10 ** -7
    q += epsilon
    return np.sum(p * np.log(p / q))


def make_dirs(*dirs, exist_ok=True):
    for d in dirs:
        os.makedirs(d, exist_ok=exist_ok)


def make_generator_input(amount, boost_dim, as_generator=False):
    n = N_ADSORP
    def gen_diffs(mean, var, _n=n, up_to=1):
        diffs = np.clip(np.exp(np.random.normal(mean, var, _n)), -10, 10)
        return diffs / np.sum(diffs) * up_to

    def gen_func():
        anchor = np.random.uniform(0, 1)
        x = np.random.uniform(0.05, 0.95)
        ind = int(n*x)
        f_1 = np.insert(np.cumsum(gen_diffs(0, 3, ind, anchor)), 0, 0)
        f_2 = np.insert(np.cumsum(gen_diffs(0, 3, n - ind - 2, 1-anchor)), 0, 0) + anchor
        f = np.concatenate((f_1, np.array([anchor]), f_2))
        f[-1] = 1.0
        return f

    def sample_rand_input(size):
        latent_codes = np.clip(np.random.normal(loc=0.5, scale=0.25, size=(size, boost_dim)), 0, 1)
        artificial_curves = np.array([np.diff(gen_func()) for _ in range(size)])
        return [artificial_curves, latent_codes]
            
    def gen():
        while True:
            out = sample_rand_input(amount)
            yield out, out

    if as_generator:
        return gen()
    else:
        return sample_rand_input(amount)


def train_step(step, predictor_model, lc_model, generator_model, **kwargs):
    # Setup our directory
    # -------------------
    base_dir = kwargs.get('base_dir', 'generative_model_default')
    step_dir = os.path.join(base_dir, 'step_{}'.format(step))
    grids_dir = os.path.join(step_dir, 'grids')
    densities_dir = os.path.join(step_dir, 'results')
    target_densities_dir = os.path.join(step_dir, 'target_densities')
    model_save_dir = os.path.join(base_dir, 'model_saves')
    predictor_model_logs = os.path.join(step_dir, 'predictor_train_logs')
    generator_model_logs = os.path.join(step_dir, 'generator_model_logs')
    make_dirs(step_dir,
              grids_dir, densities_dir, target_densities_dir,
              model_save_dir, predictor_model_logs, generator_model_logs)

    predictor_save_file = os.path.join(model_save_dir, 'predictor_step_{}.hdf5'.format(step))
    generator_save_file = os.path.join(model_save_dir, 'generator_step_{}.hdf5'.format(step))

    # Train predictor on dataset
    # --------------------------
    # Get our training data
    train_grids, train_curves = data.get_all_data(matching=base_dir)
    # Define our loss function and compile our model
    loss_func = kwargs.get('loss_func', 'kullback_leibler_divergence')
    models.unfreeze(predictor_model)
    learning_rate = 10**-3
    optimizer = Adam(learning_rate, clipnorm=1.0)
    predictor_model.compile(optimizer, loss=loss_func, metrics=['mae', models.worst_abs_loss])
    # Fit our model to the dataset
    predictor_batch_size = kwargs.get('predictor_batch_size', 64)
    predictor_epochs = kwargs.get('predictor_epochs', 30)
    lr_patience = max(int(round(predictor_epochs * 0.3)), 1) # clip to at least 1
    es_patience = max(int(round(predictor_epochs * 0.4)), 1) # clip to at least 1
    predictor_model.fit(x=train_grids, y=train_curves, batch_size=predictor_batch_size,
                        epochs=predictor_epochs, validation_split=0.2,
                        callbacks=[ReduceLROnPlateau(patience=lr_patience),
                                   EarlyStopping(patience=es_patience),
                                   TensorBoard(log_dir=predictor_model_logs, histogram_freq=1,
                                               write_graph=False, write_images=True)])
    # Save our model
    predictor_model.save(predictor_save_file, include_optimizer=False)
    
    # Train generator on predictor
    # ----------------------------
    # Get our training data
    # num_curves = 10000
    num_curves = 10000
    boost_dim = kwargs.get('boost_dim', 5)
    random_curves = make_generator_input(num_curves, boost_dim, as_generator=False)
    # Create the training model
    models.freeze(predictor_model)
    lc_inp = Input(shape=(boost_dim,), name='latent_code')
    curve_inp = Input(shape=(N_ADSORP,), name='target_curve')
    generator_out = generator_model([curve_inp, lc_inp])
    predictor_out = predictor_model(generator_out)
    lc_out = lc_model(generator_out)
    training_model = Model(inputs=[curve_inp, lc_inp], outputs=[predictor_out, lc_out])
    # Define our loss function and compile our model
    loss_weights = kwargs.get('loss_weights', [1.0, 0.8])
    learning_rate = 10**-3
    optimizer = Adam(learning_rate, clipnorm=1.0)
    training_model.compile(optimizer, loss=[loss_func, 'mse'],
                           metrics={
                               'predictor_model': ['mae', models.worst_abs_loss],
                               'latent_code_model': ['mae', models.worst_abs_loss]
                           }, loss_weights=loss_weights)
    # Fit our model to the curves
    generator_batch_size = kwargs.get('generator_batch_size', 64)
    generator_epochs = kwargs.get('generator_epochs', 15)
    lr_patience = max(int(round(generator_epochs * 0.3)), 1) # clip to at least 1
    es_patience = max(int(round(generator_epochs * 0.4)), 1) # clip to at least 1
    training_model.fit(x=random_curves, y=random_curves, batch_size=generator_batch_size,
                       epochs=generator_epochs, validation_split=0.2,
                       callbacks=[ReduceLROnPlateau(patience=lr_patience),
                                  EarlyStopping(patience=es_patience),
                                  TensorBoard(log_dir=generator_model_logs, histogram_freq=1,
                                              write_graph=False, write_images=True)])
    # Save our model
    generator_model.save(generator_save_file, include_optimizer=False)
    
    # Generate new data
    # -----------------
    num_new_grids = kwargs.get('num_new_grids', 300)
    data_upscale_factor = kwargs.get('data_upscale_factor', 3)
    artificial_curves, latent_codes = make_generator_input(int(num_new_grids*data_upscale_factor), boost_dim, as_generator=False)
    generated_grids = generator_model.predict([artificial_curves, latent_codes])
    saved_grids = generated_grids.astype('int')
    for i, grid in enumerate(saved_grids):
        path = os.path.join(grids_dir, 'grid_%04d.csv'%i)
        np.savetxt(path, grid, fmt='%i', delimiter=',')

    print('Evaluating candidate grids')
    os.system('./fast_dft {}'.format(step_dir))
    
    target_densities_dir
    for i, artificial_curve in enumerate(artificial_curves):
        path = os.path.join(target_densities_dir, 'artificial_curve_%04d.csv'%i)
        np.savetxt(path, artificial_curve, fmt='%f', delimiter=',')

    # Prune data
    # ----------
    # Get the actual, target, and predicted curves
    density_files = glob.glob(os.path.join(densities_dir, 'density_*.csv'))
    density_files.sort()
    actual_densities = [np.append(np.genfromtxt(density_file, delimiter=',', skip_header=1, max_rows=N_ADSORP)[:, 1], 1) for density_file in density_files]
    target_densities = [np.cumsum(np.insert(curve_diffs, 0, 0)) for curve_diffs in artificial_curves]
    predicted_densities = [np.cumsum(np.insert(curve_diffs, 0, 0)) for curve_diffs in predictor_model.predict(generated_grids)]
    generated_grids = list(generated_grids)
    new_data = list(zip(actual_densities, target_densities, predicted_densities, generated_grids))

    # Sort the grids by some metric
    # K-nearest sampling
    num_train_samples = kwargs.get('num_train_samples', 200)
    random_train_samples = train_curves[:num_train_samples]
    
    def sort_key(x):
        actual_curve, target_curve, predicted_curve, _ = x
        actual_curve = np.diff(actual_curve)
        distance = 0
        for train_sample in random_train_samples:
            distance += KL_divergence(actual_curve, train_sample)
        return distance
    
    def generator_acc(x):
        actual_curve, target_curve, predicted_curve, _ = x
        delta_prime_err = np.sum(np.abs(actual_curve - target_curve))
        return delta_prime_err
    
    # Remove the grids that are already good
    print('Finding most dissimilar grids')
    new_data.sort(key=sort_key, reverse=True)
    generator_accuracy = sum(map(generator_acc, new_data)) / len(new_data)
    print("Generated data error metric: {}".format(generator_accuracy))
    explore_rate = kwargs.get('explore_rate', 0.95)
    explore_num = int(explore_rate * num_new_grids)
    refine_num = num_new_grids - explore_num
    new_data = new_data[:explore_num] + new_data[-refine_num:]

    # Add data back to dataset
    # ------------------------
    # Remove our tmp data
    shutil.rmtree(grids_dir)
    shutil.rmtree(densities_dir)
    shutil.rmtree(target_densities_dir)
    make_dirs(grids_dir, densities_dir, target_densities_dir)
    
    # Save new data
    print('Saving new grids')
    for i, (density, target_density, _, grid) in enumerate(new_data):
        grid_path = os.path.join(grids_dir, 'grid_%04d.csv'%i)
        density_path = os.path.join(densities_dir, 'density_%04d.csv'%i)
        target_density_path = os.path.join(target_densities_dir, 'artificial_curve_%04d.csv'%i)
        np.savetxt(grid_path, grid, fmt='%i', delimiter=',')
        np.savetxt(target_density_path, np.diff(target_density), fmt='%f', delimiter=',')

    print('Evaluating new grids')
    os.system('./fast_dft {}'.format(step_dir))

    return generator_accuracy


def start_training(**kwargs):
    # Setup our directories
    base_dir = kwargs.get('base_dir', 'generative_model_default')
    make_dirs(base_dir)
    
    # Write our training parameters
    with open(os.path.join(base_dir, 'train_parameters.json'), 'w') as f:
        json.dump(kwargs, f)

    # Make our models
    predictor_model, lc_model = models.make_predictor_model(**kwargs)
    generator_model = models.make_generator_model(**kwargs)

    train_steps = kwargs.get('train_steps', 10)
    last_accuracy = None
    for step in range(train_steps):
        last_accuracy = train_step(step, predictor_model, lc_model, generator_model, **kwargs)

    return last_accuracy


if __name__ == '__main__':
    # start_training(predictor_epochs=2, generator_epochs=2)
    start_training(predictor_epochs=30, generator_epochs=15)
