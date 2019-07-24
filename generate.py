import os
import glob
import sys
from multiprocessing import Pool
from random import randint
from random import shuffle
import argparse

import numpy as np
np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 1000
import pandas as pd
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import tensorflow as tf
import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import Model, Sequential
from keras.models import load_model
from keras.callbacks import *
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import matplotlib.pyplot as plt
from tqdm import tqdm

from generate_data import *
from constants import *
import simul_dft as dft
#  ssh yliu1021@hoffman2.idre.ucla.edu                         
"""
We have a function T: S -> F that is the ground truth (i.e. the dft simulation),
which can take a grid in S and map it to some metric on the adsorption curve
in F (linedist).

We have a model M: S' -> F' that approximates T as a CNN
(we call this the proxy enforcer as it acts in place of the enforcer T).

We can train a generative model G: F' x C -> S  that approximates M^-1.
C are latent codes that seed the generative model (to make it parameterize its
outputs on something)
"""

def press(event):
    if event.key != 'q':
        exit(0)



base_dir = 'generative_model_target_fullsize'
os.makedirs(base_dir, exist_ok=True)

# Hyperparameters
uniform_boost_dim = 5
loss_weights = [1.0, 0.75] # weights of losses in the metric and each latent code

proxy_enforcer_epochs = 20
proxy_enforcer_epochs = 10
proxy_enforcer_batchsize = 64

generator_train_size = 10000
# generator_train_size = 128
generator_epochs = 50
generator_epochs = 20
generator_batchsize = 64
generator_train_size //= generator_batchsize

n_gen_grids = 300

gen_random_sample_rate = 0.95    # generate 80% random curves, 20% will be target curve
train_random_sample_rate = 0.95  # train on 80% random curves, 20% will be target curve


def summarize_model(model):
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))


def log_diff_loss(y_true, y_pred):
    return -K.mean(K.log(1 - K.abs(y_true - y_pred) + K.epsilon()))


def biased_loss(y_true, y_pred):
    loss = ((y_pred - y_true) ** 2) * ((y_true + 0.1) ** -0.5)
    return K.mean(loss, axis=-1)

    
def round_through(x):
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    x = (0.5 * x) + 0.5 # Gradient is steeper than regular sigmoid activation
    return K.clip(x, 0, 1)


def binary_sigmoid(x):
    return round_through(_hard_sigmoid(x))


def worst_abs_loss(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred))


def worst_mse_loss(y_true, y_pred):
    return K.max((y_true - y_pred) ** 2)


# Enforcer model
def make_proxy_enforcer_model():
    first_filter_size = 3
    
    inp = Input(shape=(GRID_SIZE, GRID_SIZE), name='proxy_enforcer_input')
    x = Lambda(lambda x: K.tile(x, [1, 3, 3]))(inp)
    x = Reshape((GRID_SIZE * 3, GRID_SIZE * 3, 1))(x)
    x = Cropping2D(cropping=(GRID_SIZE - (first_filter_size - 1), GRID_SIZE - (first_filter_size - 1)))(x)

    x = Conv2D(16, first_filter_size, padding='valid', name='conv0')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(32, 3, padding='valid', name='conv1')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(64, 3, padding='valid', name='conv2')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, 3, padding='valid', strides=2, name='conv3')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, 3, padding='valid', strides=2, name='conv4')(x)   
    x = LeakyReLU()(x)

    x = Flatten()(x)

    x = Dense(2048, name='hidden_fc_1', activation='relu')(x)
    x = Dense(2048, name='hidden_fc_2', activation='relu')(x)
    hidden = Dense(2048, name='hidden_fc_final', activation='relu')(x)

    latent_code_uni = Dense(uniform_boost_dim, name='uniform_latent_codes')(hidden)
    out = Dense(N_ADSORP, name='out', activation='softmax')(hidden)
    
    model = Model(inputs=[inp], outputs=[out], name='proxy_enforcer_model')
    lc_uni = Model(inputs=[inp], outputs=[latent_code_uni], name='uniform_latent_code_model')

    return model, lc_uni


# Generator model
def make_generator_model():
    latent_code_uni = Input(shape=(uniform_boost_dim,))

    inp = Input(shape=(N_ADSORP,))

    conc = Concatenate(axis=-1)([inp, latent_code_uni])

    Q_GRID_SIZE = GRID_SIZE // 4
    H_GRID_SIZE = GRID_SIZE // 2

    x = Dense(Q_GRID_SIZE * Q_GRID_SIZE * 64, name='fc1')(conc)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(Q_GRID_SIZE * Q_GRID_SIZE * 64, name='fc3')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(Q_GRID_SIZE * Q_GRID_SIZE * 128, name='fc4')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Reshape((Q_GRID_SIZE, Q_GRID_SIZE, 128))(x)

    x = Conv2DTranspose(128, 5, strides=1, padding='same', name='pre_deconv1')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(128, 3, strides=1, padding='same', name='pre_deconv2')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128, 3, strides=2, padding='same', name='deconv_expand1')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(64, 3, strides=2, padding='same', name='deconv_expand2')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(32, 3, strides=1, padding='same', name='post_deconv1')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    out = Conv2D(1, 3, strides=1, padding='same', activation=binary_sigmoid, name='generator_conv')(x)
    out = Reshape((GRID_SIZE, GRID_SIZE))(out)

    model = Model(inputs=[inp, latent_code_uni], outputs=[out],
                  name='generator_model')
    
    return model


def get_target_curve():
    # this curve is the one with a large pore the entire size of the grid
    density_file = 'generative_model_3/step_custom_grids/results/density_0009.csv'
    density = np.genfromtxt(density_file, delimiter=',', skip_header=1, max_rows=N_ADSORP)
    density = density[:, 1]
    return np.diff(density, append=1.0)


def make_generator_input(n_grids, use_generator=False, batchsize=generator_batchsize):
    n = N_ADSORP
    def gen_diffs(mean, var, _n=n, up_to=1):
        diffs = np.clip(np.exp(np.random.normal(mean, var, _n)), -10, 10)
        return diffs / np.sum(diffs) * up_to

    def gen_func():
        f = np.insert(np.cumsum(gen_diffs(0, 2)), 0, 0)
        return f

    def gen_func():
        anchor = np.random.uniform(0, 1)
        x = np.clip(np.random.normal(0.5, 0.4), 0.05, 0.95)
        ind = int(n*x)
        f_1 = np.insert(np.cumsum(gen_diffs(0, 3, ind, anchor)), 0, 0)
        f_2 = np.insert(np.cumsum(gen_diffs(0, 3, n - ind - 2, 1-anchor)), 0, 0) + anchor
        f = np.concatenate((f_1, np.array([anchor]), f_2))
        f[-1] = 1.0
        return f

    if use_generator:
        def gen():
            while True:
                uniform_latent_code = np.clip(np.random.normal(loc=0.5, scale=0.25, size=(batchsize, uniform_boost_dim)), 0, 1)
                artificial_metrics = list()
                num_random_samples = int(round(batchsize * train_random_sample_rate))
                for i in range(num_random_samples):
                    artificial_metrics.append(np.diff(gen_func()))
                artificial_metrics.extend([get_target_curve()] * (batchsize - num_random_samples))
                artificial_metrics = np.array(artificial_metrics)
                out = [artificial_metrics, uniform_latent_code]
                yield out, out
        return gen()
    else:
        print('Generating')
        uniform_latent_code = np.clip(np.random.normal(loc=0.5, scale=0.25, size=(n_grids, uniform_boost_dim)), 0, 1)
        artificial_metrics = list()
        num_random_samples = int(round(n_grids * gen_random_sample_rate))
        for i in range(num_random_samples):
            artificial_metrics.append(np.diff(gen_func()))
        artificial_metrics.extend([get_target_curve()] * (n_grids - num_random_samples))
        artificial_metrics = np.array(artificial_metrics)
        return artificial_metrics, uniform_latent_code


def print_weight(model):
    layer = model.layers[4]
    weights = layer.get_weights()
    print(weights)


def freeze_model(model):
    for layer in model.layers:
        layer.trainable = False


def unfreeze_model(model):
    for layer in model.layers:
        layer.trainable = True


def train_step(generator_model, proxy_enforcer_model, lc_uni, step):
    """
    1) Train M on the grids in generative_models_*
    2) Train G on M
    3) Generate random grids using G
    4) Add new grids to generative_models_*
    """
    prev_step = step - 1
    step_dir = os.path.join(base_dir, 'step{}'.format(step))
    prev_step_dir = os.path.join(base_dir, 'step{}'.format(prev_step))
    enforcer_log_dir = os.path.join(step_dir, 'enforcer_log')
    generator_log_dir = os.path.join(step_dir, 'generator_log')
    generator_model_save_loc = os.path.join(step_dir, 'generator.hdf5')
    proxy_enforcer_model_save_loc = os.path.join(step_dir, 'enforcer.hdf5')
    lc_uni_save_loc = os.path.join(step_dir, 'lc_uni.hdf5')
    os.makedirs(enforcer_log_dir, exist_ok=True)
    os.makedirs(generator_log_dir, exist_ok=True)
    os.makedirs(step_dir, exist_ok=True)
    
    # Train M
    lr = 0.001
    prev_enforcer_save_loc = os.path.join(prev_step_dir, 'enforcer.hdf5')
    if os.path.exists(prev_enforcer_save_loc):
        print('Found enforcer previous model. Loading from there')
        try:
            proxy_enforcer_model.load_weights(prev_enforcer_save_loc)
            lr *= 0.5
        except:
            print('Incompatible save')

    unfreeze_model(proxy_enforcer_model)
    optimizer = Adam(lr=lr, clipnorm=1.0)
    proxy_enforcer_model.compile(optimizer, loss='kullback_leibler_divergence', metrics=['mae', worst_abs_loss])
    summarize_model(proxy_enforcer_model)

    grids, metrics = get_all_data()
    proxy_enforcer_model.fit(x=grids, y=metrics, batch_size=proxy_enforcer_batchsize,
                             epochs=proxy_enforcer_epochs, validation_split=0.3,
                             callbacks=[ReduceLROnPlateau(patience=15),
                                        EarlyStopping(patience=40, restore_best_weights=True)])

    proxy_enforcer_model.save(proxy_enforcer_model_save_loc)

    # Train G on M
    lr = 0.001
    generator_train_generator = make_generator_input(n_grids=generator_train_size,
                                                     use_generator=True,
                                                     batchsize=generator_batchsize)

    prev_generator_save_loc = os.path.join(prev_step_dir, 'generator.hdf5')
    if os.path.exists(prev_generator_save_loc):
        print('Found generator previous model. Loading from there')
        try:
            generator_model.load_weights(prev_generator_save_loc)
            lr *= 0.5
        except:
            print('Incompatible save')

    latent_code_uni = Input(shape=(uniform_boost_dim,), name='latent_code')
    inp = Input(shape=(N_ADSORP,), name='target_metric')
    
    generator_model.compile('adam', loss='mse', metrics=['mae'])
    generator_out = generator_model([inp, latent_code_uni])
    
    freeze_model(proxy_enforcer_model)
    proxy_enforcer_out = proxy_enforcer_model(generator_out)
    latent_code_uni_out = lc_uni(generator_out)
    
    training_model = Model(inputs=[inp, latent_code_uni],
                           outputs=[proxy_enforcer_out, latent_code_uni_out])
    optimizer = Adam(lr=lr, clipvalue=5.0)
    training_model.compile(optimizer, loss=['kullback_leibler_divergence', 'mse'],
                           metrics={
                               'proxy_enforcer_model': ['mae', worst_abs_loss],
                               'uniform_latent_code_model': 'mae',
                           }, loss_weights=loss_weights)
    summarize_model(training_model)
    training_model.fit_generator(generator_train_generator, steps_per_epoch=generator_train_size,
                                 epochs=generator_epochs,
                                 callbacks=[ReduceLROnPlateau(patience=25, monitor='loss'),
                                            EarlyStopping(patience=40, restore_best_weights=True,
                                                          monitor='loss')],
                                 max_queue_size=32, shuffle=False)

    generator_model.save(generator_model_save_loc)
    lc_uni.save(lc_uni_save_loc)

    # Generate random grids using G then evaluate them
    artificial_metrics, uniform_latent_code = make_generator_input(n_grids=n_gen_grids, use_generator=False)
    
    generated_grids = generator_model.predict([artificial_metrics, uniform_latent_code])
    generated_grids = generated_grids.astype('int')

    grid_dir = os.path.join(step_dir, 'grids')
    density_dir = os.path.join(step_dir, 'results')
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(density_dir, exist_ok=True)
    print('saving generated grids')
    for i in range(n_gen_grids):
        path = os.path.join(grid_dir, 'grid_%04d.csv'%i)
        np.savetxt(path, generated_grids[i, :, :], fmt='%i', delimiter=',')
    
    print('evaluating grids')
    os.system('./fast_dft {}'.format(step_dir))
    
    print('saving artificial metrics')
    artificial_metrics_dir = os.path.join(step_dir, 'artificial_metrics')
    os.makedirs(artificial_metrics_dir, exist_ok=True)
    for i, artificial_metric in enumerate(artificial_metrics):
        path = os.path.join(artificial_metrics_dir, 'artificial_metric_%04d.csv'%i)
        np.savetxt(path, artificial_metric, fmt='%f', delimiter=',')


def visualize_enforcer(model_step=None):
    if model_step is None:
        model_step = 1
    enforcer_model, _ = make_proxy_enforcer_model()
    enforcer_model.load_weights(os.path.join(base_dir, 'step{}/enforcer.hdf5'.format(model_step)))

    all_data_files = get_all_data_files(get_all_files=True)
    all_data_files = [item for sublist in all_data_files for item in zip(*sublist)]
    shuffle(all_data_files)

    extreme_grid = None
    extreme_density = None
    extreme_pred_density = None
    extreme_metric = 1

    for grid_file, density_file in all_data_files:
        # if os.path.join(base_dir, 'step1') not in grid_file:
        #     continue
        grid = np.genfromtxt(grid_file, delimiter=',')
        density = np.genfromtxt(density_file, delimiter=',', skip_header=1, max_rows=N_ADSORP)
        density = density[:, 1]

        pred_diff = enforcer_model.predict(np.array([grid]))[0]

        pred_density = np.cumsum(pred_diff)
        metric = np.mean(np.abs(density - pred_density))
        
        if metric < extreme_metric:
            extreme_grid = grid
            extreme_density = density
            extreme_pred_density = pred_density
            extreme_metric = metric
        
        fig = plt.figure(1, figsize=(6, 8))
        fig.canvas.mpl_connect('key_press_event', press)

        ax = plt.subplot(211)
        ax.pcolor(grid, cmap='Greys', vmin=0.0, vmax=1.0)
        ax.set_aspect('equal')

        ax = plt.subplot(212)
        x = np.linspace(0, 1, N_ADSORP+1)
        ax.plot(x, np.insert(density, N_ADSORP, 1))
        ax.plot(x, np.insert(pred_density, 0, 0))
        ax.legend(['Target', 'Predicted'], loc='best')
        ax.set_aspect('equal')
        
        fig.text(0.5, 0.05, 'Mean absolute difference: {:.4f}'.format(metric), ha='center')

        plt.title('{} {}'.format(grid_file, density_file))
        plt.show()

    fig = plt.figure(1, figsize=(6, 8))
    fig.canvas.mpl_connect('key_press_event', press)

    ax = plt.subplot(211)
    ax.pcolor(extreme_grid, cmap='Greys', vmin=0.0, vmax=1.0)
    ax.set_aspect('equal')

    ax = plt.subplot(212)
    x = np.linspace(0, 1, N_ADSORP+1)
    ax.plot(x, np.insert(extreme_density, N_ADSORP, 1))
    ax.plot(x, np.insert(extreme_pred_density, 0, 0))
    ax.legend(['Target', 'Predicted'], loc='upper left')
    ax.set_aspect('equal')

    fig.text(0.5, 0.05, 'Mean absolute difference: {:.4f}'.format(extreme_metric), ha='center')

    plt.show()


def visualize_generator(step, model_step=None):
    if model_step is None:
        model_step = step
    enforcer_model, _ = make_proxy_enforcer_model()
    enforcer_model.load_weights(os.path.join(base_dir, 'step{}/enforcer.hdf5'.format(model_step)))
    # visualize_curr_step_generator(step, None)
    visualize_curr_step_generator(step, enforcer_model)


def visualize_curr_step_generator(step, enforcer_model=None):
    step_dir = os.path.join(base_dir, 'step{}'.format(step))
    grid_dir = os.path.join(step_dir, 'grids')
    density_dir = os.path.join(step_dir, 'results')
    artificial_metrics_dir = os.path.join(step_dir, 'artificial_metrics')
    
    grid_files = glob.glob(os.path.join(grid_dir, 'grid_*'))
    grid_files.sort()
    density_files = glob.glob(os.path.join(density_dir, 'density_*'))
    density_files.sort()
    artificial_metrics_files = glob.glob(os.path.join(artificial_metrics_dir, 'artificial_metric_*'))
    artificial_metrics_files.sort()
    
    for grid_file, density_file, artificial_metric_file in zip(grid_files, density_files, artificial_metrics_files):
        grid = np.genfromtxt(grid_file, delimiter=',')
        density = np.genfromtxt(density_file, delimiter=',', skip_header=1, max_rows=N_ADSORP)
        density = density[:, 1]
        artificial_metrics = np.genfromtxt(artificial_metric_file, delimiter=',')
        artificial_density = np.cumsum(artificial_metrics)

        if enforcer_model:
            pred_diff = enforcer_model.predict(np.array([grid]))[0]
            pred_density = np.cumsum(pred_diff)
        
        fig = plt.figure(1, figsize=(6, 8))
        fig.canvas.mpl_connect('key_press_event', press)

        ax = plt.subplot(211)
        ax.pcolor(grid, cmap='Greys', vmin=0.0, vmax=1.0)
        ax.set_aspect('equal')

        ax = plt.subplot(212)
        x = np.linspace(0, 1, N_ADSORP+1)
        ax.plot(x, np.insert(density, N_ADSORP, 1))
        ax.plot(x, np.insert(artificial_density, 0, 0))
        if enforcer_model:
            ax.plot(x, np.insert(pred_density, 0, 0))
            ax.legend(['Actual', 'Target for M^-1', 'Predicted by M'], loc='best')
            metric = np.mean(np.abs(artificial_density - pred_density))
            fig.text(0.5, 0.05, 'Mean absolute difference: {:.4f}'.format(metric), ha='center')
        else:
            ax.legend(['Actual', 'Target for M^-1'], loc='best')

        ax.set_aspect('equal')

        plt.show()


def generate_custom_curves(model_step):
    step_dir = os.path.join(base_dir, 'step_custom_curves')
    os.makedirs(step_dir, exist_ok=True)

    generator_model = make_generator_model()
    proxy_enforcer_model, lc_uni = make_proxy_enforcer_model()
    generator_model.load_weights(os.path.join(base_dir, 'step{}/generator.hdf5'.format(model_step)))
    proxy_enforcer_model.load_weights(os.path.join(base_dir, 'step{}/enforcer.hdf5'.format(model_step)))
    
    n_generate = 30
    artificial_metrics = list()
    for i in range(n_generate):
        diffs = np.zeros(N_ADSORP)
        ind = int(i / n_generate * N_ADSORP)
        diffs[ind] = 1.0
        artificial_metrics.append(diffs)
    artificial_metrics = np.array(artificial_metrics)
    uniform_latent_code = np.random.uniform(-0.2, 0.2, size=(n_generate, uniform_boost_dim))
    
    square_density_files = glob.glob(os.path.join(base_dir, 'step_custom_grids/results/density_*.csv'))
    square_density_files.sort()
    square_densities = [np.genfromtxt(square_density_file, delimiter=',', skip_header=1,
                                      max_rows=N_ADSORP) for square_density_file in square_density_files]
    square_densities = np.array(square_densities)
    square_metrics = np.diff(square_densities, axis=1, append=1.0)
    square_metrics = square_metrics[:, :, 1]
    new_latent_codes = np.random.uniform(-0.2, 0.2, size=(len(square_metrics), uniform_boost_dim))
    artificial_metrics = np.concatenate((artificial_metrics, square_metrics))
    uniform_latent_code = np.concatenate((uniform_latent_code, new_latent_codes))
    
    generated_grids = generator_model.predict([artificial_metrics, uniform_latent_code])
    generated_grids = generated_grids.astype('int')

    grid_dir = os.path.join(step_dir, 'grids')
    density_dir = os.path.join(step_dir, 'results')
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(density_dir, exist_ok=True)
    print('saving generated grids')
    for i in range(len(generated_grids)):
        path = os.path.join(grid_dir, 'grid_%04d.csv'%i)
        np.savetxt(path, generated_grids[i, :, :], fmt='%i', delimiter=',')
    
    print('evaluating grids')
    os.system('./fast_dft {}'.format(step_dir))
    
    print('saving artificial metrics')
    artificial_metrics_dir = os.path.join(step_dir, 'artificial_metrics')
    os.makedirs(artificial_metrics_dir, exist_ok=True)
    for i, artificial_metric in enumerate(artificial_metrics):
        path = os.path.join(artificial_metrics_dir, 'artificial_metric_%04d.csv'%i)
        np.savetxt(path, artificial_metric, fmt='%f', delimiter=',')


def generate_custom_grids(model_step):
    step_dir = os.path.join(base_dir, 'step_custom_grids')
    os.makedirs(step_dir, exist_ok=True)

    grids = list()
    for i in range(1, 11):
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        offset = i
        halfway = int(GRID_SIZE / 2)
        grid[halfway - offset : halfway + offset, halfway - offset : halfway + offset] = 1
        grids.append(grid)
        
    grid_dir = os.path.join(step_dir, 'grids')
    density_dir = os.path.join(step_dir, 'results')
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(density_dir, exist_ok=True)
    print('saving generated grids')
    for i, grid in enumerate(grids):
        path = os.path.join(grid_dir, 'grid_%04d.csv'%i)
        np.savetxt(path, grid, fmt='%i', delimiter=',')
    
    print('evaluating grids')
    os.system('./fast_dft {}'.format(step_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("startfrom", help="start training from step",
                        type=int, nargs="?", default=-1)
    args = parser.parse_args()
    
    startfrom = args.startfrom
    if startfrom != -1:
        for step in range(startfrom, 100):
            generator_model = make_generator_model()
            proxy_enforcer_model, lc_uni = make_proxy_enforcer_model()
            
            train_step(generator_model, proxy_enforcer_model, lc_uni, step=step)
            
            K.clear_session()
            del generator_model
            del proxy_enforcer_model
            del lc_uni
    else:
        prompt = """\
Options:
    1. Visualize enforcer (model M)
    2. Visualize generator (model M^-1)
    3. Generate custom curves
    4. Generate custom grids
Enter <option number> <step number>
"""
        inp = input(prompt)
        option_num, max_steps = inp.split(' ')
        option_num = int(option_num)
        if option_num == 1:
            visualize_enforcer(model_step=max_steps)
        elif option_num == 2:
            visualize_generator(max_steps, model_step=max_steps)
        elif option_num == 3:
            generate_custom_curves(max_steps)
        elif option_num == 4:
            generate_custom_grids(max_steps)
        else:
            print('Invalid option')

