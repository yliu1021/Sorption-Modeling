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


base_dir = 'generative_model_2'
os.makedirs(base_dir, exist_ok=True)

# Hyperparameters
uniform_boost_dim = 5
loss_weights = [1, 0.5] # weights of losses in the metric and each latent code

proxy_enforcer_epochs = 120
# proxy_enforcer_epochs = 1
proxy_enforcer_batchsize = 64

generator_train_size = 10000
# generator_train_size = 10
generator_epochs = 120
# generator_epochs = 1
generator_batchsize = 64
generator_train_size //= generator_batchsize

n_gen_grids = 1000

generator_train_bias = 3.0


def summarize_model(model):
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))


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
    inp = Input(shape=(GRID_SIZE, GRID_SIZE), name='proxy_enforcer_input')
    x = Reshape((GRID_SIZE, GRID_SIZE, 1), name='reshape')(inp)
    
    x = Conv2D(16, 2, padding='same', name='conv0')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(32, 2, padding='same', name='conv1')(x)
    x = LeakyReLU()(x)

    x = Conv2D(64, 2, padding='same', name='conv2')(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, 3, padding='same', strides=2, name='conv3')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, 3, padding='same', strides=2, name='conv4')(x)   
    x = LeakyReLU()(x)
    
    x = Conv2D(516, 3, padding='same', strides=2, name='conv5')(x)
    x = LeakyReLU()(x)
    
    x = Flatten()(x)

    x = Dense(2048, name='hidden_fc_1', activation='relu')(x)
    x = Dense(2048, name='hidden_fc_2', activation='relu')(x)
    hidden = Dense(2048, name='hidden_fc_final', activation='relu')(x)

    latent_code_uni = Dense(uniform_boost_dim, name='uniform_latent_codes')(hidden)
    out = Dense(1, name='out')(hidden)
    
    model = Model(inputs=[inp], outputs=[out], name='proxy_enforcer_model')
    lc_uni = Model(inputs=[inp], outputs=[latent_code_uni], name='uniform_latent_code_model')

    return model, lc_uni


# Generator model
def make_generator_model():
    latent_code_uni = Input(shape=(uniform_boost_dim,))

    inp = Input(shape=(1,))

    conc = Concatenate(axis=-1)([inp, latent_code_uni])

    Q_GRID_SIZE = GRID_SIZE // 4
    H_GRID_SIZE = GRID_SIZE // 2

    x = Dense(Q_GRID_SIZE * Q_GRID_SIZE * 4, name='fc1')(conc)
    x = Dense(Q_GRID_SIZE * Q_GRID_SIZE * 8, name='fc2')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Dense(Q_GRID_SIZE * Q_GRID_SIZE * 16, name='fc3')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Reshape((Q_GRID_SIZE, Q_GRID_SIZE, 16))(x)

    x = Conv2DTranspose(16, 2, strides=1, padding='same', name='deconv1')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(32, 2, strides=1, padding='same', name='deconv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(32, 3, strides=1, padding='same', name='deconv3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128, 3, strides=2, padding='same', name='deconv_expand_1')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(128, 3, strides=2, padding='same', name='deconv_expand_2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    out = Conv2D(1, 5, strides=1, padding='same', activation=binary_sigmoid, name='conv1')(x)
    out = Reshape((GRID_SIZE, GRID_SIZE))(out)

    model = Model(inputs=[inp, latent_code_uni], outputs=[out],
                  name='generator_model')
    
    return model


def make_generator_input(n_grids, use_generator=False, batchsize=generator_batchsize):
    if use_generator:
        while True:
            uniform_latent_code = np.random.normal(loc=0.0, scale=0.5, size=(batchsize,
                                                                             uniform_boost_dim))
            artificial_metrics = np.random.uniform(low=0.0, high=1.0, size=(batchsize,))
            out = [artificial_metrics ** generator_train_bias, uniform_latent_code]
            yield out, out
    else:
        uniform_latent_code = np.random.normal(loc=0.0, scale=0.5, size=(n_grids, uniform_boost_dim))
        artificial_metrics = np.random.uniform(low=0.0, high=1.0, size=(n_grids,))
        return (artificial_metrics ** generator_train_bias, uniform_latent_code)


def train_step(generator_model, proxy_enforcer_model, lc_uni, step):
    """
    1) Train M on the grids in our predict_mc directory.
    2) Train G on M
    3) Generate random grids using G
    4) Replace metrics with new grids in predict_mc
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
    # load the grids and densities from previous 5 steps (or less if we don't have that much)
    proxy_enforcer_model.trainable = True
    optimizer = Adam(lr=0.001, clipnorm=1.0)
    proxy_enforcer_model.compile(optimizer, loss=biased_loss, metrics=['mae', worst_abs_loss])
    summarize_model(proxy_enforcer_model)
    
    grids, metrics = get_all_data()
    proxy_enforcer_model.fit(x=grids, y=metrics, batch_size=proxy_enforcer_batchsize,
                             epochs=proxy_enforcer_epochs, validation_split=0.2,
                             callbacks=[ReduceLROnPlateau(patience=20),
                                        EarlyStopping(patience=40, restore_best_weights=True)])
    
    proxy_enforcer_model.save(proxy_enforcer_model_save_loc)

    # Train G on M
    # generate artificial training data
    generator_train_generator = make_generator_input(n_grids=generator_train_size,
                                                     use_generator=True,
                                                     batchsize=generator_batchsize)

    latent_code_uni = Input(shape=(uniform_boost_dim,), name='latent_code')
    inp = Input(shape=(1,), name='target_metric')
    generator_out = generator_model([inp, latent_code_uni])
    proxy_enforcer_model.trainable = False
    proxy_enforcer_out = proxy_enforcer_model(generator_out)
    latent_code_uni_out = lc_uni(generator_out)
    training_model = Model(inputs=[inp, latent_code_uni],
                           outputs=[proxy_enforcer_out, latent_code_uni_out])

    optimizer = Adam(lr=0.001, clipnorm=1.0)
    training_model.compile(optimizer, loss=[biased_loss, 'mse'],
                           metrics={
                               'proxy_enforcer_model': ['mae', worst_abs_loss],
                               'uniform_latent_code_model': 'mae',
                           }, loss_weights=loss_weights)
    summarize_model(training_model)
    training_model.fit_generator(generator_train_generator, steps_per_epoch=generator_train_size,
                                 epochs=generator_epochs,
                                 callbacks=[ReduceLROnPlateau(patience=10, monitor='loss'),
                                            EarlyStopping(patience=40, restore_best_weights=True,
                                                          monitor='loss')],
                                 max_queue_size=32, shuffle=False)

    generator_model.save(generator_model_save_loc)
    lc_uni.save(lc_uni_save_loc)

    # Generate random grids using G then evaluate them
    uniform_latent_code = np.random.uniform(-0.5, 0.5, size=(n_gen_grids, uniform_boost_dim))
    artificial_metrics = np.linspace(0.0, 1.0, num=n_gen_grids) ** generator_train_bias
    
    generated_grids = generator_model.predict([artificial_metrics, uniform_latent_code])
    eval_grids = np.around(generated_grids).astype('int')

    grid_dir = os.path.join(step_dir, 'grids')
    density_dir = os.path.join(step_dir, 'results')
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(density_dir, exist_ok=True)
    print('saving eval grids')
    for i in range(n_gen_grids):
        path = os.path.join(grid_dir, 'grid_%04d.csv'%i)
        np.savetxt(path, eval_grids[i, :, :], fmt='%i', delimiter=',')
    
    print('evaluating grids')
    os.system('./fast_dft {}'.format(step_dir))
    # dft.pool_arg['grid_dir'] = grid_dir
    # dft.pool_arg['result_dir'] = density_dir
    # p = Pool()
    # list(tqdm(p.imap(dft.run_dft_pool, range(n_gen_grids)), total=n_gen_grids))


def gen_and_eval_grids(step):
    step_dir = os.path.join(base_dir, 'step{}'.format(step))
    os.makedirs(step_dir, exist_ok=True)

    generator_model = make_generator_model()
    generator_model.load_weights(os.path.join(base_dir, 'step{}/generator.hdf5'.format(step)),
                                 by_name=True)
    
    uniform_latent_code = np.random.uniform(-0.2, 0.2, size=(n_gen_grids, uniform_boost_dim))
    artificial_metrics = np.linspace(0.0, 1.0, num=n_gen_grids)
    
    generated_grids = generator_model.predict([artificial_metrics, uniform_latent_code])
    eval_grids = np.around(generated_grids).astype('int')

    grid_dir = os.path.join(step_dir, 'grids')
    density_dir = os.path.join(step_dir, 'results')
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(density_dir, exist_ok=True)
    print('saving eval grids')
    for i in range(n_gen_grids):
        path = os.path.join(grid_dir, 'grid_%04d.csv'%i)
        np.savetxt(path, eval_grids[i, :, :], fmt='%i', delimiter=',')
    
    print('evaluating grids')
    dft.pool_arg['grid_dir'] = grid_dir
    dft.pool_arg['result_dir'] = density_dir
    p = Pool()
    list(tqdm(p.imap(dft.run_dft_pool, range(n_gen_grids)), total=n_gen_grids))


def visualize_grids(model_step=None):
    if model_step is None:
        model_step = 1
    generator_model = make_generator_model()
    enforcer_model, _ = make_proxy_enforcer_model()
    generator_model.load_weights(os.path.join(base_dir,
                                              'step{}/generator.hdf5'.format(model_step)),
                                 by_name=True)
    enforcer_model.load_weights(os.path.join(base_dir,
                                             'step{}/enforcer.hdf5'.format(model_step)),
                                by_name=True)

    for _ in range(5):
        points = 200
        step_points = points // 4
        latent_range = (-0.5, 0.5)
        uniform_latent_code_25 = np.random.uniform(*latent_range, size=(step_points, uniform_boost_dim))
        uniform_latent_code_50 = np.random.uniform(*latent_range, size=(step_points, uniform_boost_dim))
        uniform_latent_code_75 = np.random.uniform(*latent_range, size=(step_points, uniform_boost_dim))
        uniform_latent_code_100 = np.random.uniform(*latent_range, size=(step_points, uniform_boost_dim))        
        artificial_metrics_25 = np.linspace(0.0, 0.25, num=step_points)
        artificial_metrics_50 = np.linspace(0.25, 0.5, num=step_points)
        artificial_metrics_75 = np.linspace(0.5, 0.75, num=step_points)
        artificial_metrics_100 = np.linspace(0.75, 1.0, num=step_points)
    
        generated_grids_25 = generator_model.predict([artificial_metrics_25, uniform_latent_code_25])
        generated_grids_50 = generator_model.predict([artificial_metrics_50, uniform_latent_code_50])
        generated_grids_75 = generator_model.predict([artificial_metrics_75, uniform_latent_code_75])
        generated_grids_100 = generator_model.predict([artificial_metrics_100, uniform_latent_code_100])
        predicted_metrics_25 = np.squeeze(enforcer_model.predict(generated_grids_25))
        predicted_metrics_50 = np.squeeze(enforcer_model.predict(generated_grids_50))
        predicted_metrics_75 = np.squeeze(enforcer_model.predict(generated_grids_75))
        predicted_metrics_100 = np.squeeze(enforcer_model.predict(generated_grids_100))
    
        artificial_metrics = np.concatenate((artificial_metrics_25,
                                         artificial_metrics_50,
                                         artificial_metrics_75,
                                         artificial_metrics_100))
        predicted_metrics = np.concatenate((predicted_metrics_25,
                                         predicted_metrics_50,
                                         predicted_metrics_75,
                                         predicted_metrics_100))
        fit = np.polyfit(artificial_metrics, predicted_metrics, 1)
        
        fit_fn = np.poly1d(fit)
    
        plt.scatter(artificial_metrics_25, predicted_metrics_25)
        plt.scatter(artificial_metrics_50, predicted_metrics_50)
        plt.scatter(artificial_metrics_75, predicted_metrics_75)
        plt.scatter(artificial_metrics_100, predicted_metrics_100)
        
        x = np.linspace(0.0, 1.0, num=2)
        plt.plot(x, fit_fn(x))
        plt.plot([0, 1], [0, 1])
        plt.xlabel('artificial_metrics')
        plt.ylabel('predicted_metrics')
        plt.title('step {}'.format(model_step))
        plt.legend(['best fit y = {:.2f}x + {:.2f}'.format(fit[0], fit[1]), 'base line',
                    '0.0-0.25', '0.25-0.50', '0.50-0.75', '0.75-1.00'])
        plt.show()


def visualize_accuracy(max_steps, model_step=None):
    if model_step is None:
        model_step = step
    proxy_enforcer_model, _ = make_proxy_enforcer_model()
    proxy_enforcer_model.load_weights(os.path.join(base_dir,
                                                   'step{}/enforcer.hdf5'.format(model_step)),
                                      by_name=True)
    
    for step in range(max_steps, -1, -1):
        grid = np.array(fetch_grids_from_step(base_dir, step))
        if (grid.size == 0):
            continue
        density = np.array(fetch_density_from_step(base_dir, step))
        density[:, :, 0] /= N_ADSORP
        metric = (np.sum(np.abs(density[:, :, 1] - density[:, :, 0]), axis=1) / 20.0)

        pred = np.squeeze(proxy_enforcer_model.predict(grid))
    
        fit = np.polyfit(metric, pred, 1)
        print("Log R^2: {:.6f}".format(np.corrcoef(np.log(metric), np.log(pred))[0, 1] ** 2))
        correlation = np.corrcoef(metric, pred)[0, 1]
        mse = np.mean((pred - metric) ** 2)

        fit_fn = np.poly1d(fit)
        x = np.linspace(0, 1, num=10)
        plt.plot([0, 1], [0, 1])
        plt.plot(x, fit_fn(x), color='red')
        plt.scatter(metric, pred)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Actual metric')
        plt.ylabel('Predicted metric')
        plt.title('Metric: {:.2f} +/- {:.2f}, R^2={:.3f}, R={:.3f}, mse={:.3e}'.format(metric.mean(),
                                                                                       metric.std(),
                                                                                       correlation**2,
                                                                                       correlation,
                                                                                       mse))
        plt.legend(['base line',
                    'y = {:.2f}x + {:.2f}'.format(fit[0], fit[1]),
                    'step {}'.format(step)])
        plt.show()


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
    1. Visualize generated grids
    2. Visualize accuracy
    3. Gen and eval
Enter <option number> <step number>
"""
        inp = input(prompt)
        option_num, max_steps = inp.split(' ')
        option_num = int(option_num)
        max_steps = int(max_steps)
        if option_num == 1:
            visualize_grids(model_step=max_steps)
        elif option_num == 2:
            visualize_accuracy(max_steps, model_step=max_steps)
        elif option_num == 3:
            gen_and_eval_grids(max_steps)
        else:
            print('Invalid option')

