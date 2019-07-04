import os
import glob
from multiprocessing import Pool
from random import randint

import numpy as np
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from keras.layers import *
from keras.models import Model, Sequential
from keras.models import load_model
from keras.callbacks import *
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
categorical_boost_dim = 5
binary_boost_dim = 5
uniform_boost_dim = 1
num_boost_dim = categorical_boost_dim + binary_boost_dim + uniform_boost_dim
loss_weights = [1, 0.05, 0.05, 0.2] # weights of losses in the metric and each latent code

proxy_enforcer_epochs = 200
proxy_enforcer_batchsize = 64

generator_train_size = 10000
generator_epochs = 200
generator_batchsize = 64

n_gen_grids = 1000


def make_proxy_enforcer_model():
    # shared layers
    inp = Input(shape=(GRID_SIZE, GRID_SIZE), name='proxy_enforcer_input')
    x = Reshape((GRID_SIZE, GRID_SIZE, 1), name='reshape')(inp)
    
    x = Conv2D(16, 3, padding='same', name='conv1')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    
    x = Conv2D(32, 3, padding='same', name='conv2')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)

    x = Conv2D(64, 3, padding='same', strides=2, name='conv3')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    
    x = Conv2D(128, 3, padding='same', strides=2, name='conv4')(x)   
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    
    x = Flatten()(x)
    
    hidden = Dense(1024, name='hidden_fc')(x)
    hidden = BatchNormalization()(hidden)
    
    # use the hidden layer for latent codes as well as output
    latent_code_cat = Dense(categorical_boost_dim, activation='softmax', name='categorical_latent_codes')(hidden)
    latent_code_bin = Dense(binary_boost_dim, activation='sigmoid', name='binary_latent_codes')(hidden)
    latent_code_uni = Dense(uniform_boost_dim, name='uniform_latent_codes')(hidden)
    out = Dense(1, name='out')(hidden)
    
    model = Model(inputs=[inp], outputs=[out], name='proxy_enforcer_model')
    # model.compile(loss='mean_squared_error', optimizer='adam')
    
    lc_cat = Model(inputs=[inp], outputs=[latent_code_cat], name='categorical_latent_code_model')
    # lc_cat.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    lc_bin = Model(inputs=[inp], outputs=[latent_code_bin], name='binary_latent_code_model')
    # lc_bin.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    lc_uni = Model(inputs=[inp], outputs=[latent_code_uni], name='uniform_latent_code_model')
    # lc_uni.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    return model, (lc_cat, lc_bin, lc_uni)


def make_generator_model():
    latent_code_cat = Input(shape=(categorical_boost_dim,))
    latent_code_bin = Input(shape=(binary_boost_dim,))
    latent_code_uni = Input(shape=(uniform_boost_dim,))

    inp = Input(shape=(1,))

    conc = Concatenate(axis=-1)([inp, latent_code_cat, latent_code_bin, latent_code_uni])

    Q_GRID_SIZE = GRID_SIZE // 4
    H_GRID_SIZE = GRID_SIZE // 2

    x = Dense(Q_GRID_SIZE * Q_GRID_SIZE * 64, name='fc1', use_bias=False)(conc)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Reshape((Q_GRID_SIZE, Q_GRID_SIZE, 64))(x)

    x = Conv2DTranspose(64, 3, strides=1, padding='same', name='deconv1')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(64, 3, strides=2, padding='same', name='deconv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(128, 3, strides=2, padding='same', name='deconv3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    out = Conv2D(1, 3, strides=1, padding='valid', activation='sigmoid', name='conv1')(x)
    out = ZeroPadding2D(padding=1)(out)
    out = Reshape((GRID_SIZE, GRID_SIZE))(out)

    model = Model(inputs=[inp, latent_code_cat, latent_code_bin, latent_code_uni], outputs=[out],
                  name='generator_model')
    
    return model


def make_generator_input(n_grids=10000, use_gaussian_metric=False):
    def one_hot(i):
        a = np.zeros(categorical_boost_dim, dtype='float')
        a[i] = 1.0
        return a
    categorical_latent_code = [one_hot(x % categorical_boost_dim) for x in range(n_grids)]
    categorical_latent_code = np.array(categorical_latent_code)
    np.random.shuffle(categorical_latent_code)
    
    binary_latent_code = np.random.randint(0, high=2, size=(n_grids, binary_boost_dim)).astype('float')

    uniform_latent_code = np.random.uniform(low=0.0, high=1.0, size=(n_grids, uniform_boost_dim))
    
    if use_gaussian_metric:
        artificial_metrics = np.random.normal(loc=-1.0, scale=2., size=(n_grids,))
    else:
        artificial_metrics = np.random.uniform(low=-5.0, high=2.6, size=(n_grids,))
    
    return (artificial_metrics, categorical_latent_code, binary_latent_code, uniform_latent_code)


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


def train_step(generator_model, proxy_enforcer_model, lc_cat, lc_bin, lc_uni, step):
    """
    1) Train M on the grids in our predict_mc directory.
    2) Train G on M
    3) Generate random grids using G
    4) Replace metrics with new grids in predict_mc
    """
    prev_step = step - 1
    step_dir = os.path.join(base_dir, 'step{}'.format(step))
    prev_step_dir = os.path.join(base_dir, 'step{}'.format(prev_step))
    os.makedirs(step_dir, exist_ok=True)
    
    # Train M
    # load the grids and densities from previous 5 steps (or less if we don't have that much)
    grids = list()
    densities = list()
    for s in range(step - 1, max(-1, step - 5), -1):
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

    metric = np.log(np.sum(np.absolute(densities[:, :, 1] - densities[:, :, 0]), axis=1))
    print('Metric stats: {:.2f} ± {:.2f}'.format(metric.mean(), metric.std()))    
    proxy_enforcer_model.trainable = True
    proxy_enforcer_model.compile('adam', loss='mse', metrics=['mae'])
    proxy_enforcer_model.summary()
    proxy_enforcer_model.fit(x=grids, y=metric, batch_size=proxy_enforcer_batchsize,
                             epochs=proxy_enforcer_epochs, validation_split=0.1,
                             callbacks=[ReduceLROnPlateau(patience=10),
                                        EarlyStopping(patience=25)])
                             
    # Train G on M
    # generate artificial training data
    (artificial_metrics,
     categorical_latent_code,
     binary_latent_code,
     uniform_latent_code) = make_generator_input(n_grids=generator_train_size)

    latent_code_cat = Input(shape=(categorical_boost_dim,))
    latent_code_bin = Input(shape=(binary_boost_dim,))
    latent_code_uni = Input(shape=(uniform_boost_dim,))

    inp = Input(shape=(1,))
    
    generator_out = generator_model([inp, latent_code_cat, latent_code_bin, latent_code_uni])
    proxy_enforcer_model.trainable = False
    proxy_enforcer_out = proxy_enforcer_model(generator_out)
    latent_code_cat_out = lc_cat(generator_out)
    latent_code_bin_out = lc_bin(generator_out)
    latent_code_uni_out = lc_uni(generator_out)
    
    training_model = Model(inputs=[inp, latent_code_cat, latent_code_bin, latent_code_uni],
                           outputs=[proxy_enforcer_out, latent_code_cat_out, latent_code_bin_out, latent_code_uni_out])
    training_model.compile('adam', loss=['mae', 'categorical_crossentropy', 'binary_crossentropy', 'mse'],
                           metrics={
                               'categorical_latent_code_model': 'categorical_accuracy',
                               'binary_latent_code_model': 'binary_accuracy',
                               'uniform_latent_code_model': 'mae',
                               'proxy_enforcer_model': 'mse'
                           }, loss_weights=loss_weights)
    training_model.summary()
    training_model.fit(x=[artificial_metrics, categorical_latent_code, binary_latent_code, uniform_latent_code],
                       y=[artificial_metrics, categorical_latent_code, binary_latent_code, uniform_latent_code],
                       batch_size=generator_batchsize, epochs=generator_epochs,
                       validation_split=0.2, callbacks=[ReduceLROnPlateau(patience=5),
                                                        EarlyStopping(patience=10)])

    generator_model_save_loc = os.path.join(step_dir, 'generator.hdf5')
    proxy_enforcer_model_save_loc = os.path.join(step_dir, 'enforcer.hdf5')
    lc_cat_save_loc = os.path.join(step_dir, 'lc_cat.hdf5')
    lc_bin_save_loc = os.path.join(step_dir, 'lc_bin.hdf5')
    lc_uni_save_loc = os.path.join(step_dir, 'lc_uni.hdf5')
    generator_model.save_weights(generator_model_save_loc)
    proxy_enforcer_model.save_weights(proxy_enforcer_model_save_loc)
    lc_cat.save_weights(lc_cat_save_loc)
    lc_bin.save_weights(lc_bin_save_loc)
    lc_uni.save_weights(lc_uni_save_loc)

    # Generate random grids using G then evaluate them
    (_,
     categorical_latent_code, 
     binary_latent_code,
     uniform_latent_code) = make_generator_input(n_grids=n_gen_grids)
    artificial_metrics = np.linspace(-5.0, 2.6, num=n_gen_grids)
    
    generated_grids = generator_model.predict([artificial_metrics,
                                               categorical_latent_code, 
                                               binary_latent_code,
                                               uniform_latent_code])
    generated_grids = np.around(generated_grids).astype('int')

    grid_dir = os.path.join(step_dir, 'grids')
    density_dir = os.path.join(step_dir, 'results')
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(density_dir, exist_ok=True)
    print('saving new grids')
    for i in range(n_gen_grids):
        path = os.path.join(grid_dir, 'grid_%04d.csv'%i)
        np.savetxt(path, generated_grids[i, :, :], fmt='%i', delimiter=',')
    
    dft.pool_arg['grid_dir'] = grid_dir
    dft.pool_arg['result_dir'] = density_dir
    p = Pool()
    list(tqdm(p.imap(dft.run_dft_pool, range(n_gen_grids)), total=n_gen_grids))
    

def test_model(step):
    generator_model = make_generator_model()
    generator_model.load_weights('generative_model/step{}/generator.hdf5'.format(step), by_name=True)
    step_dir = os.path.join(base_dir, 'step{}'.format(step))
    (_,
     categorical_latent_code, 
     binary_latent_code,
     uniform_latent_code) = make_generator_input(n_grids=n_gen_grids)
    artificial_metrics = np.linspace(-5.0, 3.0, num=n_gen_grids)
    
    generated_grids = generator_model.predict([artificial_metrics,
                                               categorical_latent_code, 
                                               binary_latent_code,
                                               uniform_latent_code])
    generated_grids = np.around(generated_grids).astype('int')

    grid_dir = os.path.join(step_dir, 'grids')
    density_dir = os.path.join(step_dir, 'results')
    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(density_dir, exist_ok=True)
    print('saving new grids')
    for i in range(n_gen_grids):
        path = os.path.join(grid_dir, 'grid_%04d.csv'%i)
        np.savetxt(path, generated_grids[i, :, :], fmt='%i', delimiter=',')
    
    print('simulating densities')
    dft.pool_arg['grid_dir'] = grid_dir
    dft.pool_arg['result_dir'] = density_dir
    p = Pool()
    list(tqdm(p.imap(dft.run_dft_pool, range(n_gen_grids)), total=n_gen_grids))


def test_acc():
    proxy_enforcer_model, _ = make_proxy_enforcer_model()
    proxy_enforcer_model.load_weights('generative_model/step3/enforcer.hdf5')
    grid = np.genfromtxt('generative_model/step3/grids/grid_0000.csv', delimiter=',')
    density = np.genfromtxt('generative_model/step3/results/density_0000.csv', delimiter=',',
                            skip_header=1, max_rows=N_ADSORP)
    density[:, 0] /= N_ADSORP
    metric = np.log(np.sum(np.absolute(density[:, 1] - density[:, 0]), axis=0))
    print(metric)
    grid = np.reshape(grid, (1, 20, 20))
    p = proxy_enforcer_model.predict(grid)
    print(p)


if __name__ == '__main__':
    generator_model = make_generator_model()
    proxy_enforcer_model, (lc_cat, lc_bin, lc_uni) = make_proxy_enforcer_model()

    load_from_step = 0
    if load_from_step:
        step_dir = 'generative_model/step{}/'.format(step)
        generator_model_save_loc = os.path.join(step_dir, 'generator.hdf5')
        proxy_enforcer_model_save_loc = os.path.join(step_dir, 'enforcer.hdf5')
        lc_cat_save_loc = os.path.join(step_dir, 'lc_cat.hdf5')
        lc_bin_save_loc = os.path.join(step_dir, 'lc_bin.hdf5')
        lc_uni_save_loc = os.path.join(step_dir, 'lc_uni.hdf5')
        generator_model.load_weights(generator_model_save_loc)
        proxy_enforcer_model.load_weights(proxy_enforcer_model_save_loc)
        lc_cat.load_weights(lc_cat_save_loc)
        lc_bin.load_weights(lc_bin_save_loc)
        lc_uni.load_weights(lc_uni_save_loc)

    for step in range(4, 10):
        train_step(generator_model, proxy_enforcer_model, lc_cat, lc_bin, lc_uni, step=step)

