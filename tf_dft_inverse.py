import time
import os
import glob

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


base_dir = './generative_model_4'
model_loc = os.path.join(base_dir, 'generator.hdf5')
log_loc = os.path.join(base_dir, 'logs')

generator_train_size = 500
generator_epochs = 25
generator_batchsize = 32
generator_train_size //= generator_batchsize

max_var = 12


def worst_abs_loss(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred))


def round_through(x):
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    x = (0.5 * x) + 0.5 # Gradient is steeper than regular sigmoid activation
    return K.clip(x, 0, 1)


def binary_sigmoid(x):
    return round_through(_hard_sigmoid(x))


def make_generator_input(n_grids, use_generator=False, batchsize=generator_batchsize):
    if use_generator:
        def gen():
            while True:
                artificial_metrics = list()
                for i in range(batchsize):
                    diffs = np.exp(np.random.normal(0, np.sqrt(i/batchsize) * max_var, N_ADSORP))
                    diffs /= np.sum(diffs, axis=0)
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

    x = Dense(Q_GRID_SIZE * Q_GRID_SIZE * 64, name='fc1')(inp)
    x = LeakyReLU()(x)

    x = Dense(Q_GRID_SIZE * Q_GRID_SIZE * 64, name='fc3')(x)
    x = LeakyReLU()(x)

    x = Dense(Q_GRID_SIZE * Q_GRID_SIZE * 128, name='fc4')(x)
    x = LeakyReLU()(x)

    x = Reshape((Q_GRID_SIZE, Q_GRID_SIZE, 128))(x)

    x = Conv2DTranspose(128, 5, strides=1, padding='same', name='pre_deconv1')(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(128, 3, strides=1, padding='same', name='pre_deconv2')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128, 3, strides=2, padding='same', name='deconv_expand1')(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(64, 3, strides=2, padding='same', name='deconv_expand2')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(32, 3, strides=1, padding='same', name='post_deconv1')(x)
    x = LeakyReLU()(x)

    out = Conv2D(1, 3, strides=1, padding='same', activation=binary_sigmoid, name='generator_conv')(x)
    out = Reshape((GRID_SIZE, GRID_SIZE))(out)

    model = Model(inputs=inp, outputs=out, name='generator_model')
    
    return model


def dft_model():
    inp = Input(shape=(GRID_SIZE, GRID_SIZE), batch_size=generator_batchsize, name='dft_input')
    x = Lambda(lambda x: run_dft(x, batch_size=generator_batchsize))(inp)
    model = Model(inputs=inp, outputs=x, name='dft_model')
    
    return model


generator_train_generator = make_generator_input(n_grids=generator_train_size,
                                                 use_generator=True,
                                                 batchsize=generator_batchsize)

# Visualization
visualize=False
if visualize:
    grid_tf = tf.compat.v1.placeholder(tf.float32, shape=[generator_batchsize, GRID_SIZE, GRID_SIZE], name='input_grid')
    density_tf = run_dft(grid_tf)
    sess = K.get_session()
    generator = load_model(model_loc, custom_objects={'binary_sigmoid': binary_sigmoid})
    relative_humidity = np.arange(41) * STEP_SIZE
    for c, _ in generator_train_generator:
        grids = generator.predict(c)
        densities = sess.run(density_tf, feed_dict={grid_tf: grids})

        for diffs, grid, diffs_dft in zip(c, grids, densities):
            curve = np.cumsum(np.insert(diffs, 0, 0))
            curve_dft = np.cumsum(np.insert(diffs_dft, 0, 0))
        
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
    exit(0)

generator = inverse_dft_model()
dft_model = dft_model()

generator.compile('adam', loss='mse')
inp = Input(shape=(N_ADSORP,), name='target_metric')
generator_out = generator(inp)
dft_out = dft_model(generator_out)

training_model = Model(inputs=inp, outputs=dft_out)
optimizer = SGD(lr=0.001, clipnorm=1.0)
training_model.compile(optimizer, loss='categorical_crossentropy', metrics=['mae', worst_abs_loss])
training_model.summary()

training_model.fit_generator(generator_train_generator, steps_per_epoch=generator_train_size,
                             epochs=generator_epochs,
                             max_queue_size=32, shuffle=False,
                             callbacks=[TensorBoard(log_dir=log_loc,
                                                    write_graph=True, write_grads=True,
                                                    write_images=True)])

generator.save(model_loc)