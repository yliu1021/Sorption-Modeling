import time
import os
import glob
import sys

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np

from constants import *
from tf_dft import run_dft

import matplotlib.pyplot as plt


def area_between(y_true, y_pred):
    return K.mean(K.abs(K.cumsum(y_true, axis=-1) - K.cumsum(y_pred, axis=-1)))


def squared_area_between(y_true, y_pred):
    return K.mean(K.square(K.cumsum(y_true, axis=-1) - K.cumsum(y_pred, axis=-1)))


base_dir = './generative_model_4'
model_loc = os.path.join(base_dir, 'generator.hdf5')
log_loc = os.path.join(base_dir, 'logs')

generator_train_size = 50000
generator_epochs = 20
try:
    generator_epochs = int(sys.argv[1])
except:
    pass
generator_batchsize = 128
generator_train_size //= generator_batchsize
loss = squared_area_between
lr = 1e-6
max_var = 24
inner_loops = 30


def round_through(x):
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    x = (0.5 * x) + 0.5 # Gradient is steeper than regular sigmoid activation
    return K.clip(x, 0, 1)


def binary_sigmoid(x):
    return round_through(_hard_sigmoid(x))


def make_steps():
    curves = list()
    for i in range(N_ADSORP):
        diffs = np.zeros(N_ADSORP)
        diffs[i] = 1.0
        curves.append(diffs)
    curves = np.array(curves)
    return curves, curves


def make_generator_input(n_grids, use_generator=False, batchsize=generator_batchsize):
    if use_generator:
        def gen():
            while True:
                artificial_metrics = list()
                rand_curve_size = batchsize - N_ADSORP
                for i in range(rand_curve_size):
                    diffs = np.exp(np.random.normal(0, (i/rand_curve_size)**1.2 * max_var, N_ADSORP))
                    diffs /= np.sum(diffs, axis=0)
                    artificial_metrics.append(diffs)
                for i in range(N_ADSORP):
                    diffs = np.zeros(N_ADSORP)
                    diffs[i] = 1.0
                    artificial_metrics.append(diffs)
                artificial_metrics = np.array(artificial_metrics)
                out = artificial_metrics
                yield out, out
        return gen()
    else:
        artificial_metrics = list()
        for i in range(n_grids):
            diffs = np.exp(np.random.normal(0, np.sqrt(i/n_grids) * max_var, N_ADSORP))
            diffs /= np.sum(diffs, axis=0)
            artificial_metrics.append(diffs)
        artificial_metrics = np.array(artificial_metrics)
        return artificial_metrics, uniform_latent_code


def inverse_dft_model():
    inp = Input(shape=(N_ADSORP,), name='generator_input', batch_size=generator_batchsize)

    Q_GRID_SIZE = GRID_SIZE // 4

    x = Dense(Q_GRID_SIZE * Q_GRID_SIZE * 2048, name='fc1')(inp)
    x = LeakyReLU()(x)

    x = Reshape((Q_GRID_SIZE, Q_GRID_SIZE, 2048))(x)

    x = Conv2DTranspose(2048, 3, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(2048, 3, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(1024, 3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(512, 3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128, 3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(64, 3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)

    out = Conv2D(1, 3, strides=1, padding='same', activation=binary_sigmoid, name='generator_conv')(x)
    out = Reshape((GRID_SIZE, GRID_SIZE))(out)

    model = Model(inputs=inp, outputs=out, name='generator_model')

    return model


def dft_model():
    inp = Input(shape=(GRID_SIZE, GRID_SIZE), batch_size=generator_batchsize, name='dft_input')
    x = Lambda(lambda x: run_dft(x,
                                 batch_size=generator_batchsize,
                                 inner_loops=inner_loops))(inp)
    model = Model(inputs=inp, outputs=x, name='dft_model')

    return model


generator_train_generator = make_generator_input(n_grids=generator_train_size,
                                                 use_generator=True,
                                                 batchsize=generator_batchsize)

generator = inverse_dft_model()

# Visualization
visualize = False
see_grids = False
try:
    visualize = sys.argv[1] == 'v'
    see_grids = 's' in sys.argv
except:
    pass
if visualize:
    generator = load_model(model_loc, custom_objects={'binary_sigmoid': binary_sigmoid,
                                                      'area_between': area_between})
    # generator.load_weights(model_loc)
    relative_humidity = np.arange(41) * STEP_SIZE

    errors = list()

    c = make_steps()[0][::-1]
    grids = generator.predict(c)
    densities = run_dft(grids)
    for diffs, grid, diffs_dft in zip(c, grids, densities):
        curve = np.cumsum(np.insert(diffs, 0, 0))
        curve_dft = np.cumsum(np.insert(diffs_dft, 0, 0))

        error = np.sum(np.abs(curve - curve_dft)) / len(curve)
        # errors.append(error)

        if see_grids:
            fig = plt.figure(figsize=(10, 4))
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            ax = plt.subplot(1, 2, 1)
            ax.clear()
            ax.set_title('Grid (Black = Solid, White = Pore)')
            ax.set_yticks(np.linspace(0, 20, 5))
            ax.set_xticks(np.linspace(0, 20, 5))
            ax.pcolor(1 - grid, cmap='Greys', vmin=0.0, vmax=1.0)
            ax.set_aspect('equal')

            ax = plt.subplot(1, 2, 2)
            ax.clear()
            ax.set_title('Adsorption Curve')
            ax.plot(relative_humidity, curve, label='Target')
            ax.plot(relative_humidity, curve_dft, label='DFT')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Relative Humidity')
            ax.set_ylabel('Proportion of Pores filled')
            ax.set_aspect('equal')
            ax.legend()

            plt.show()

    curves = [next(generator_train_generator) for _ in range(5)]
    for c, _ in curves:
        print('computing...')
        grids = generator.predict(c)
        densities = run_dft(grids)
        for diffs, grid, diffs_dft in zip(c, grids, densities):
            curve = np.cumsum(np.insert(diffs, 0, 0))
            curve_dft = np.cumsum(np.insert(diffs_dft, 0, 0))

            error = np.sum(np.abs(curve - curve_dft)) / len(curve)
            errors.append(error)

            if see_grids:
                fig = plt.figure(figsize=(10, 4))
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])

                ax = plt.subplot(1, 2, 1)
                ax.clear()
                ax.set_title('Grid (Black = Solid, White = Pore)')
                ax.set_yticks(np.linspace(0, 20, 5))
                ax.set_xticks(np.linspace(0, 20, 5))
                ax.pcolor(1 - grid, cmap='Greys', vmin=0.0, vmax=1.0)
                ax.set_aspect('equal')

                ax = plt.subplot(1, 2, 2)
                ax.clear()
                ax.set_title('Adsorption Curve')
                ax.plot(relative_humidity, curve, label='Target')
                ax.plot(relative_humidity, curve_dft, label='DFT')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel('Relative Humidity')
                ax.set_ylabel('Proportion of Pores filled')
                ax.set_aspect('equal')
                ax.legend()

                plt.show()
    print('Mean: ', np.array(errors).mean())
    print('Std: ', np.array(errors).std())
    plt.hist(errors, bins=10)
    plt.title('Error Distribution')
    plt.xlabel('Abs error')
    plt.xlim(0, 1)
    plt.show()
    exit(0)

dft_model = dft_model()

generator.compile('adam', loss='mse')
inp = Input(shape=(N_ADSORP,), name='target_metric')
generator_out = generator(inp)
dft_out = dft_model(generator_out)

training_model = Model(inputs=inp, outputs=dft_out)
optimizer = Adam(lr=lr)
training_model.compile(optimizer,
                       loss=loss,
                       metrics=[area_between])
training_model.summary()

filepath = os.path.join(base_dir, 'generator_{epoch:03d}.hdf5')
save_freq=generator_train_size*5*generator_batchsize
training_model.fit_generator(generator_train_generator,
                             steps_per_epoch=generator_train_size,
                             epochs=generator_epochs,
                             max_queue_size=64, shuffle=False,
                             callbacks=[TensorBoard(log_dir=log_loc,
                                                    write_graph=True,
                                                    write_images=True),
                                        ReduceLROnPlateau(monitor='loss',
                                                          factor=0.1,
                                                          patience=100),
                                        ModelCheckpoint(filepath,
                                                        monitor='area_between',
                                                        save_best_only=True,
                                                        mode='min',
                                                        save_freq='epoch')])

generator.save(model_loc)
