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


def KL_divergence(p, q):
    epsilon = 10 ** -7
    q += epsilon
    return np.sum(p * np.log(p / q))


def make_dirs(*dirs, exist_ok=True):
    for d in dirs:
        os.makedirs(d, exist_ok=exist_ok)
        

def train_step(step, predictor_model, lc_model, generator_model, **kwargs):
    # Setup our directory
    # -------------------
    base_dir = kwargs.get('base_dir', 'generative_model_default')
    step_dir = os.path.join(base_dir, 'step_{}'.format(step))
    grids_dir = os.path.join(step_dir, 'grids')
    densities_dir = os.path.join(step_dir, 'results')
    target_densities_dir = os.path.join(step_dir, 'target_densities')
    model_save_dir = os.path.join(base_dir, 'model_saves')
    predictor_model_logs = os.path.join(step_dir, 'predictor_model_logs')
    generator_model_logs = os.path.join(step_dir, 'generator_model_logs')
    make_dirs(step_dir,
              grids_dir, densities_dir, target_densities_dir,
              model_save_dir, predictor_model_logs, generator_model_logs)

    predictor_save_file = os.path.join(model_save_dir, 'predictor_step_{}.hdf5'.format(step))
    generator_save_file = os.path.join(model_save_dir, 'generator_step_{}.hdf5'.format(step))

    # Train predictor on dataset
    # --------------------------
    # Get our training data
    train_grids, train_curves = data.get_all_data(matching=base_dir, augment_factor=20)
    # Define our loss function and compile our model
    predictor_loss_func = kwargs.get('predictor_loss_func', 'binary_crossentropy') # or binary_crossentropy
    models.unfreeze(predictor_model)
    learning_rate = 10**-2
    optimizer = SGD(learning_rate, clipnorm=1.0)
    predictor_model.compile(optimizer, loss=predictor_loss_func, metrics=['mae', models.worst_abs_loss])
    # Fit our model to the dataset
    predictor_batch_size = kwargs.get('predictor_batch_size', 64)
    predictor_epochs = kwargs.get('predictor_epochs', 6)
    if step == 0:
        predictor_epochs += kwargs.get('predictor_first_step_epoch_boost', 10) # train more to start off
    lr_patience = max(int(round(predictor_epochs * 0.2)), 3) # clip to at least 1
    es_patience = max(int(round(predictor_epochs * 0.8)), 4) # clip to at least 1
    
    predictor_model.fit(x=train_grids, y=train_curves, batch_size=predictor_batch_size,
                        epochs=predictor_epochs, validation_split=0.1,
                        callbacks=[ReduceLROnPlateau(patience=lr_patience, factor=0.1),
                                   EarlyStopping(patience=es_patience, restore_best_weights=True),
                                   TensorBoard(log_dir=predictor_model_logs, histogram_freq=1,
                                               write_graph=False, write_images=False)])
    # Save our model
    print('Saving model', end='... ', flush=True)
    predictor_model.save(predictor_save_file, include_optimizer=False)
    print('done')

    # Train generator on predictor
    # ----------------------------
    # Get our training data
    print('Picking random curves ', end='... ', flush=True)
    num_curves = 10000
    boost_dim = kwargs.get('boost_dim', 5)
    random_curves = data.make_generator_input(num_curves, boost_dim, allow_squeeze=True, as_generator=False)
    print('Done')

    # Create the training model
    models.freeze(predictor_model)
    lc_inp = Input(shape=(boost_dim,), name='latent_code')
    curve_inp = Input(shape=(N_ADSORP,), name='target_curve')
    generator_out = generator_model([curve_inp, lc_inp])
    predictor_out = predictor_model(generator_out)
    lc_out = lc_model(generator_out)
    training_model = Model(inputs=[curve_inp, lc_inp], outputs=[predictor_out, lc_out])
    # Define our loss function and compile our model
    generator_loss_func = kwargs.get('generator_loss_func', 'binary_crossentropy') # or binary_crossentropy
    loss_weights = kwargs.get('loss_weights', [1.0, 0.6])
    learning_rate = 10**-2
    optimizer = Adam(learning_rate)
    training_model.compile(optimizer, loss=[generator_loss_func, 'mse'],
                           metrics={
                               'predictor_model': ['mae', models.worst_abs_loss],
                               'latent_code_model': ['mae', models.worst_abs_loss]
                           }, loss_weights=loss_weights)
    # Fit our model to the curves
    generator_batch_size = kwargs.get('generator_batch_size', 64)
    generator_epochs = kwargs.get('generator_epochs', 3)
    if step == 0:
        generator_epochs += kwargs.get('generator_first_step_epoch_boost', 20) # train more to start off
    lr_patience = max(int(round(generator_epochs * 0.1)), 3) # clip to at least 1
    es_patience = max(int(round(generator_epochs * 0.8)), 4) # clip to at least 1
    training_model.fit(x=random_curves, y=random_curves, batch_size=generator_batch_size,
                       epochs=generator_epochs, validation_split=0.1,
                       callbacks=[ReduceLROnPlateau(patience=lr_patience, factor=0.1),
                                  EarlyStopping(patience=es_patience),
                                  TensorBoard(log_dir=generator_model_logs, histogram_freq=1,
                                              write_graph=False, write_images=False)])
    # Save our model
    generator_model.save(generator_save_file, include_optimizer=False)
    
    # Generate new data
    # -----------------
    num_new_grids = kwargs.get('num_new_grids', 100)
    data_upscale_factor = kwargs.get('data_upscale_factor', 1.5)
    artificial_curves, latent_codes = data.make_generator_input(int(num_new_grids*data_upscale_factor), boost_dim, as_generator=False)
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
    # Sample k curves from our dataset to see how close we are to our dataset
    def generator_err(x):
        actual_curve, target_curve, predicted_curve, _ = x
        delta_prime_err = np.sum(np.abs(actual_curve - target_curve))
        return delta_prime_err
    
    def predictor_err(x):
        actual_curve, target_curve, predicted_curve, _ = x
        gamma_err = np.sum(np.abs(actual_curve - predicted_curve))
        return gamma_err
    
    def cross_err(x):
        actual_curve, target_curve, predicted_curve, _ = x
        delta_err = np.sum(np.abs(target_curve - predicted_curve))
        return delta_err

    # Evaluate our accuracies
    generator_error = np.array(list(map(generator_err, new_data))) / (N_ADSORP + 1)
    predictor_error = np.array(list(map(predictor_err, new_data))) / (N_ADSORP + 1)
    cross_error = np.array(list(map(cross_err, new_data))) / (N_ADSORP + 1)
    print('Generated data error metric: {:.3f} ± {:.3f}'.format(generator_error.mean(),
                                                                generator_error.std()))
    print('Predictor error metric: {:.3f} ± {:.3f}'.format(predictor_error.mean(),
                                                           predictor_error.std()))
    print('Cross error metric: {:.3f} ± {:.3f}'.format(cross_error.mean(),
                                                       cross_error.std()))

    # Remove the grids that are already good
    print('Finding most dissimilar grids')
    divergences = np.fromiter(map(lambda x: divergence(x[0]), new_data), dtype=float)
    divergences = divergences ** 1.2
    divergences /= np.sum(divergences)
    new_data_inds = np.random.choice(len(new_data), num_new_grids, replace=False, p=divergences)
    new_data = [new_data[i] for i in new_data_inds]

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

    return generator_error, predictor_error, cross_error


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
    generator_err_hist = list()
    predictor_err_hist = list()
    cross_err_hist = list()
    kwargs['generator_err_hist'] = generator_err_hist
    kwargs['predictor_err_hist'] = predictor_err_hist
    kwargs['cross_err_hist'] = cross_err_hist
    # TODO
    for step in range(train_steps):
        acc = train_step(step, predictor_model, lc_model, generator_model, **kwargs)
        generator_err_hist.append(acc[0])
        predictor_err_hist.append(acc[1])
        cross_err_hist.append(acc[2])

    generator_err_medians = [np.median(g) for g in generator_err_hist]
    predictor_err_medians = [np.median(p) for p in predictor_err_hist]
    cross_err_medians = [np.median(c) for c in cross_err_hist]
    steps = list(range(1, len(generator_err_medians)+1))

    plt.violinplot(generator_err_hist, showmeans=True)
    plt.plot(steps, generator_err_medians)
    plt.title('Generator error per step')
    plt.ylabel('Mean abs diff metric')
    plt.xlabel('Step number')
    plt.ylim(0, 1)
    plt.show()

    plt.violinplot(predictor_err_hist, showmeans=True)
    plt.plot(steps, predictor_err_medians)
    plt.title('Predictor error per step')
    plt.ylabel('Mean abs diff metric')
    plt.xlabel('Step number')
    plt.ylim(0, 1)
    plt.show()

    plt.violinplot(cross_err_hist, showmeans=True)
    plt.plot(steps, cross_err_medians)
    plt.title('Cross error per step')
    plt.ylabel('Mean abs diff metric')
    plt.xlabel('Step number')
    plt.ylim(0, 1)
    plt.show()

    last_3_means = 0
    last_3_means += generator_err_hist[-1].mean()
    last_3_means += generator_err_hist[-2].mean()
    last_3_means += generator_err_hist[-3].mean()
    return last_3_means / 3


if __name__ == '__main__':
    start_training(predictor_epochs=50, generator_epochs=40, train_steps=20,
                   predictor_first_step_epoch_boost=0,
                   generator_first_step_epoch_boost=0,
                   predictor_loss_func='binary_crossentropy',
                   generator_loss_func='binary_crossentropy',
                   base_dir='generative_model_new')
    # start_training(predictor_epochs=30, generator_epochs=15)
