import os
import glob
from multiprocessing import Pool
from random import randint

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


base_dir = 'generative_model'
# Hyperparameters
# categorical_boost_dim = 2
# binary_boost_dim = 1
uniform_boost_dim = 40
# num_boost_dim = categorical_boost_dim + binary_boost_dim + uniform_boost_dim
loss_weights = [1, 0.5] # weights of losses in the metric and each latent code

proxy_enforcer_epochs = 100
proxy_enforcer_batchsize = 32

generator_train_size = 10000
generator_epochs = 100
generator_batchsize = 32

n_gen_grids = 300


def homogeneity_loss(layer):
    def loss(y_true, y_pred):
        return K.abs(y_pred - y_true) + 0.0025*K.sum(K.exp(-K.abs(2 * layer - 1)), axis=(1,2))
    return loss
    
    
def round_through(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    '''Hard sigmoid different from the more conventional form (see definition of K.hard_sigmoid).
    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    '''
    x = (0.5 * x) + 0.5 # Gradient is steeper than regular sigmoid activation
    return K.clip(x, 0, 1)


def binary_sigmoid(x):
    '''Binary hard sigmoid for training binarized neural network.
    # Reference:
    - [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, Courbariaux et al. 2016](http://arxiv.org/abs/1602.02830}
    '''
    return round_through(_hard_sigmoid(x))


def worst_abs_loss(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred))


def worst_mse_loss(y_true, y_pred):
    return K.max((y_true - y_pred) ** 2)


def make_proxy_enforcer_model():
    # shared layers
    inp = Input(shape=(GRID_SIZE, GRID_SIZE), name='proxy_enforcer_input')
    x = Reshape((GRID_SIZE, GRID_SIZE, 1), name='reshape')(inp)
    
    x = Conv2D(16, 2, padding='same', name='conv1')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(64, 2, padding='same', name='conv2')(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, 2, padding='same', name='conv3')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(256, 2, padding='same', strides=2, name='conv4')(x)   
    x = LeakyReLU()(x)
    
    x = Conv2D(516, 2, padding='same', strides=2, name='conv5')(x)
    x = LeakyReLU()(x)
    
    x = Flatten()(x)
    
    hidden = Dense(2048, name='hidden_fc')(x)
    # hidden = BatchNormalization()(hidden)
    
    # use the hidden layer for latent codes as well as output
    # latent_code_cat = Dense(categorical_boost_dim, activation='softmax', name='categorical_latent_codes')(hidden)
    # latent_code_bin = Dense(binary_boost_dim, activation='sigmoid', name='binary_latent_codes')(hidden)
    latent_code_uni = Dense(uniform_boost_dim, name='uniform_latent_codes')(hidden)
    out = Dense(1, name='out')(hidden)
    
    model = Model(inputs=[inp], outputs=[out], name='proxy_enforcer_model')
    # model.compile(loss='mean_squared_error', optimizer='adam')
    
    # lc_cat = Model(inputs=[inp], outputs=[latent_code_cat], name='categorical_latent_code_model')
    # lc_cat.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # lc_bin = Model(inputs=[inp], outputs=[latent_code_bin], name='binary_latent_code_model')
    # lc_bin.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    lc_uni = Model(inputs=[inp], outputs=[latent_code_uni], name='uniform_latent_code_model')
    # lc_uni.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    return model, lc_uni


def make_generator_model():
    # latent_code_cat = Input(shape=(categorical_boost_dim,))
    # latent_code_bin = Input(shape=(binary_boost_dim,))
    latent_code_uni = Input(shape=(uniform_boost_dim,))

    inp = Input(shape=(1,))

    conc = Concatenate(axis=-1)([inp, latent_code_uni])

    Q_GRID_SIZE = GRID_SIZE // 4
    H_GRID_SIZE = GRID_SIZE // 2

    x = Dense(Q_GRID_SIZE * Q_GRID_SIZE * 16, name='fc1', use_bias=False)(conc)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Reshape((Q_GRID_SIZE, Q_GRID_SIZE, 16))(x)

    x = Conv2DTranspose(16, 2, strides=1, padding='same', name='deconv1')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(32, 2, strides=1, padding='same', name='deconv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(32, 2, strides=1, padding='same', name='deconv3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(64, 2, strides=2, padding='same', name='deconv4')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(128, 2, strides=2, padding='same', name='deconv5')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    out = Conv2D(1, 3, strides=1, padding='valid', activation=binary_sigmoid, name='conv1')(x)
    out = ZeroPadding2D(padding=1)(out)
    out = Reshape((GRID_SIZE, GRID_SIZE))(out)

    model = Model(inputs=[inp, latent_code_uni], outputs=[out],
                  name='generator_model')
    
    return model


def make_generator_input(n_grids=10000):
    # def one_hot(i):
    #     a = np.zeros(categorical_boost_dim, dtype='float')
    #     a[i] = 1.0
    #     return a
    # categorical_latent_code = [one_hot(x % categorical_boost_dim) for x in range(n_grids)]
    # categorical_latent_code = np.array(categorical_latent_code)
    # np.random.shuffle(categorical_latent_code)
    #
    # binary_latent_code = np.random.randint(0, high=2, size=(n_grids, binary_boost_dim)).astype('float')

    # uniform_latent_code = np.random.uniform(low=0.0, high=1.0, size=(n_grids, uniform_boost_dim))
    uniform_latent_code = np.random.normal(loc=0.0, scale=1.0, size=(n_grids, uniform_boost_dim))

    artificial_metrics = np.random.uniform(low=0.0, high=1.0, size=(n_grids,))
    
    return (artificial_metrics, uniform_latent_code)


def fetch_grids_from_step(step):
    step_dir = os.path.join(base_dir, 'step{}'.format(step))
    grid_dir = os.path.join(step_dir, 'grids')
    grid_files = glob.glob(os.path.join(grid_dir, 'grid_*.csv'))
    grid_files.sort()
    return [np.genfromtxt(grid_file, delimiter=',') for grid_file in grid_files]


def fetch_density_from_step(step):
    step_dir = os.path.join(base_dir, 'step{}'.format(step))
    density_dir = os.path.join(step_dir, 'results')
    density_files = glob.glob(os.path.join(density_dir, 'density_*.csv'))
    density_files.sort()
    return [np.genfromtxt(density_file, delimiter=',', skip_header=1,
                          max_rows=N_ADSORP) for density_file in density_files]


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
    grids = list()
    densities = list()
    for s in range(step - 1, max(-1, step - 20), -1):
        print('loading from step %d' % s)
        grids.extend(fetch_grids_from_step(s))
        densities.extend(fetch_density_from_step(s))
    for i in range(len(grids) - 1):
        s = randint(i+1, len(grids) - 1)
        grids[i], grids[s] = grids[s], grids[i]
        densities[i], densities[s] = densities[s], densities[i]

    grids = np.array(grids)
    densities = np.array(densities)
    
    densities[:, :, 0] /= N_ADSORP

    metric = (np.sum(np.absolute(densities[:, :, 1] - densities[:, :, 0]), axis=1) / 20.0)
    print('Metric stats: {:.2f} ± {:.2f}'.format(metric.mean(), metric.std()))    
    
    proxy_enforcer_model.trainable = True
    optimizer = Adam(lr=0.001, clipnorm=1.0)
    proxy_enforcer_model.compile(optimizer, loss='mse', metrics=['mae', worst_abs_loss])
    proxy_enforcer_model.summary()
    proxy_enforcer_model.fit(x=grids, y=metric, batch_size=proxy_enforcer_batchsize,
                             epochs=proxy_enforcer_epochs, validation_split=0.1,
                             callbacks=[ReduceLROnPlateau(patience=30),
                                        EarlyStopping(patience=50, restore_best_weights=True)])
    proxy_enforcer_model.save_weights(proxy_enforcer_model_save_loc)

    # Train G on M
    # generate artificial training data
    (artificial_metrics,
     uniform_latent_code) = make_generator_input(n_grids=generator_train_size)
    artificial_metrics = artificial_metrics ** 1.5

    latent_code_uni = Input(shape=(uniform_boost_dim,))

    inp = Input(shape=(1,))
    
    generator_out = generator_model([inp, latent_code_uni])
    proxy_enforcer_model.trainable = False
    proxy_enforcer_out = proxy_enforcer_model(generator_out)
    latent_code_uni_out = lc_uni(generator_out)
    
    training_model = Model(inputs=[inp, latent_code_uni],
                           outputs=[proxy_enforcer_out, latent_code_uni_out])
    optimizer = Adam(lr=0.001, clipnorm=1.0)
    training_model.compile(optimizer, loss=['mse', 'mse'],
                           metrics={
                               'proxy_enforcer_model': 'mae',
                               'proxy_enforcer_model': worst_abs_loss,
                               'uniform_latent_code_model': 'mae',
                           }, loss_weights=loss_weights)
    training_model.summary()
    training_model.fit(x=[artificial_metrics, uniform_latent_code],
                       y=[artificial_metrics, uniform_latent_code],
                       batch_size=generator_batchsize, epochs=generator_epochs,
                       validation_split=0.2, callbacks=[ReduceLROnPlateau(patience=30),
                                                        EarlyStopping(patience=50, restore_best_weights=True)])

    generator_model.save_weights(generator_model_save_loc)
    lc_uni.save_weights(lc_uni_save_loc)

    # Generate random grids using G then evaluate them
    (artificial_metrics,
     uniform_latent_code) = make_generator_input(n_grids=n_gen_grids)

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


def visualize_grids(step):
    generator_model = make_generator_model()
    enforcer_model, _ = make_proxy_enforcer_model()
    generator_model.load_weights('generative_model/step{}/generator.hdf5'.format(step), by_name=True)
    enforcer_model.load_weights('generative_model/step{}/enforcer.hdf5'.format(step), by_name=True)
    step_dir = os.path.join(base_dir, 'step{}'.format(step))
    (_,
     uniform_latent_code) = make_generator_input(n_grids=n_gen_grids)
    artificial_metrics = np.linspace(0.0, 1.0, num=n_gen_grids)
    
    generated_grids = generator_model.predict([artificial_metrics, uniform_latent_code])
    predicted_metrics = enforcer_model.predict(generated_grids)
    
    plt.scatter(artificial_metrics, predicted_metrics)
    plt.plot([0, 1], [0, 1])
    plt.xlabel('artificial_metrics')
    plt.ylabel('predicted_metrics')
    plt.title('step {}'.format(step))
    plt.show()

    # for i in range(generated_grids.shape[0]):
    #     plt.pcolor(generated_grids[i], cmap='Greys')
    #     plt.title('Predicted metric: {:.4f}'.format(float(predicted_metrics[i])))
    #     plt.show()
    # exit(0)
    
    # generated_grids = np.around(generated_grids).astype('int')
    #
    # grid_dir = os.path.join(step_dir, 'grids')
    # density_dir = os.path.join(step_dir, 'results')
    # os.makedirs(grid_dir, exist_ok=True)
    # os.makedirs(density_dir, exist_ok=True)
    # print('saving new grids')
    # for i in range(n_gen_grids):
    #     path = os.path.join(grid_dir, 'grid_%04d.csv'%i)
    #     np.savetxt(path, generated_grids[i, :, :], fmt='%i', delimiter=',')
    #
    # print('simulating densities')
    # dft.pool_arg['grid_dir'] = grid_dir
    # dft.pool_arg['result_dir'] = density_dir
    # p = Pool()
    # list(tqdm(p.imap(dft.run_dft_pool, range(n_gen_grids)), total=n_gen_grids))


def visualize_accuracy(step, model_step=None):
    if model_step is None:
        model_step = step
    proxy_enforcer_model, _ = make_proxy_enforcer_model()
    proxy_enforcer_model.load_weights('generative_model/step{}/enforcer.hdf5'.format(model_step))
    grid = np.array(fetch_grids_from_step(step))
    density = np.array(fetch_density_from_step(step))
    density[:, :, 0] /= N_ADSORP
    metric = (np.sum(np.abs(density[:, :, 1] - density[:, :, 0]), axis=1) / 20.0)

    pred = np.squeeze(proxy_enforcer_model.predict(grid))
    
    fit = np.polyfit(metric, pred, 1)
    fit_fn = np.poly1d(fit)
    plt.plot(metric, fit_fn(np.linspace(0, 1, num=len(metric))), color='red')
    plt.scatter(metric, pred)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Actual metric')
    plt.ylabel('Predicted metric')
    plt.title('Metric: {:.2f} +/- {:.2f}'.format(metric.mean(), metric.std()))
    plt.legend(['y={:.2f}x + {:.2f}'.format(fit[0], fit[1]), 'step {}'.format(step)])
    plt.show()
    # exit(0)


if __name__ == '__main__':
    for i in range(9, 8, -1):
        visualize_grids(i)
    visualize_accuracy(9, model_step=9)
    for i in range(0, 9):
        visualize_accuracy(i, model_step=9)
    exit(0)

    for step in range(9, 100):
        generator_model = make_generator_model()
        proxy_enforcer_model, (lc_uni) = make_proxy_enforcer_model()
        train_step(generator_model, proxy_enforcer_model, lc_uni, step=step)

