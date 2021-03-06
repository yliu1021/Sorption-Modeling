import time
import os
import glob
import sys
from random import shuffle, random, randint

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np

from constants import *
from tf_dft import run_dft, run_dft_pad

import matplotlib.pyplot as plt


def show_grid(grid, target_curve, dft_curve):
    relative_humidity = np.arange(41) * STEP_SIZE

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
    ax.plot(relative_humidity, target_curve, label='Target')
    ax.plot(relative_humidity, dft_curve, label='DFT')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Relative Humidity')
    ax.set_ylabel('Proportion of Pores filled')
    ax.set_aspect('equal')
    ax.legend()

    error = np.sum(np.abs(target_curve - dft_curve)) / len(target_curve)
    plt.title(f'Error: {error:.3f}')


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def area_between(y_true, y_pred):
    return K.mean(K.abs(K.cumsum(y_true, axis=-1) - K.cumsum(y_pred, axis=-1)))


def squared_area_between(y_true, y_pred):
    return K.mean(K.square(K.cumsum(y_true, axis=-1) - K.cumsum(y_pred, axis=-1)))


base_dir = './generative_model_benchmark_par'
os.makedirs(base_dir, exist_ok=True)
model_loc = os.path.join(base_dir, 'generator.hdf5')
log_loc = os.path.join(base_dir, 'logs')

generator_train_size = 64000
generator_epochs = 100
try:
    generator_epochs = int(sys.argv[1])
except:
    pass
generator_batchsize = 64
generator_train_size //= generator_batchsize
loss = squared_area_between
loss = area_between
lr = 1e-1
max_var = 24
stepwise_prop = 0.2
inner_loops = 100


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
    n = N_ADSORP
    def gen_diffs(mean, var, _n=n, up_to=1):
        diffs = np.clip(np.exp(np.random.normal(mean, var, _n)), -10, 10)
        diffs = np.exp(np.random.normal(mean, var, _n))
        return diffs / np.sum(diffs) * up_to

    def gen_func():
        if random() < stepwise_prop:
            f = np.zeros(N_ADSORP + 1)
            i = randint(1, N_ADSORP)
            f[-i:] = 1.0
            return f
        else:
            anchor = np.random.uniform(0, 1)
            x = np.random.uniform(0.05, 0.95)
            ind = int(n*x)
            f_1 = np.insert(np.cumsum(gen_diffs(0, 4, ind, anchor)), 0, 0)
            f_2 = np.insert(np.cumsum(gen_diffs(0, 4, n - ind - 2, 1-anchor)), 0, 0) + anchor
            f = np.concatenate((f_1, np.array([anchor]), f_2))
            f[-1] = 1.0
            return f

    if use_generator:
        def gen():
            while True:
                funcs = [gen_func() for _ in range(batchsize)]
                artificial_curves = np.array([np.diff(f) for f in funcs])
                yield artificial_curves, artificial_curves
        return gen
    else:
        funcs = [gen_func() for _ in range(batchsize)]
        artificial_curves = np.array([np.diff(f) for f in funcs])
        return artificial_curves, artificial_curves


def inverse_dft_model():
    # inp = Input(shape=(N_ADSORP,), name='generator_input', batch_size=generator_batchsize)
    inp = Input(shape=(N_ADSORP,), batch_size=generator_batchsize, name='generator_input')

    x1 = Dense(GRID_SIZE * GRID_SIZE * 64, name='fc1', activation='relu')(inp[:,:20])
    x2 = Dense(GRID_SIZE * GRID_SIZE * 64, name='fc2', activation='relu')(inp[:,20:])
    x1 = Reshape((GRID_SIZE, GRID_SIZE, 64))(x1)
    x2 = Reshape((GRID_SIZE, GRID_SIZE, 64))(x2)
    x1 = BatchNormalization()(x1)
    x2 = BatchNormalization()(x2)

    x1 = Conv2DTranspose(64, 20, strides=1, padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x2 = Conv2DTranspose(64, 20, strides=1, padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)
    x1 = Conv2DTranspose(64, 20, strides=1, padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x2 = Conv2DTranspose(64, 20, strides=1, padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)

    x1 = Conv2D(1, 3, strides=1, padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x2 = Conv2D(1, 3, strides=1, padding='same', activation='relu')(x2)
    x2 = BatchNormalization()(x2)

    x = K.concatenate((x1, x2), axis=-1)
    out = Conv2D(1, 3, strides=1, padding='same', activation=binary_sigmoid, name='generator_conv')(x)
    out = Reshape((GRID_SIZE, GRID_SIZE))(out)

    model = Model(inputs=inp, outputs=out, name='generator_model')

    return model


def make_dft_model():
    inp = Input(shape=(GRID_SIZE, GRID_SIZE), batch_size=generator_batchsize, name='dft_input')
    x = Lambda(lambda x: run_dft(x,
                                 batch_size=generator_batchsize,
                                 inner_loops=inner_loops)[0])(inp)
    model = Model(inputs=inp, outputs=x, name='dft_model')

    return model


generator_train_generator = make_generator_input(n_grids=generator_train_size,
                                                 use_generator=True,
                                                 batchsize=generator_batchsize)


def train(use_tpu=True):
    if use_tpu:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='yliu1021',
            zone='us-central1-f'
        )
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        tpu_strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)

        with tpu_strategy.scope():
            generator = inverse_dft_model()
            dft_model = make_dft_model()

            optimizer = SGD(lr, momentum=0.9, nesterov=True)
            inp = Input(shape=(N_ADSORP,), name='target_metric')
            generator_out = generator(inp)
            dft_out = dft_model(generator_out)

            training_model = Model(inputs=inp, outputs=dft_out)

            training_model.compile(optimizer,
                                   loss=loss,
                                   metrics=[area_between])
    else:
        generator = inverse_dft_model()
        dft_model = make_dft_model()

        optimizer = SGD(lr, momentum=0.9, nesterov=True)
        inp = Input(shape=(N_ADSORP,), name='target_metric')
        generator_out = generator(inp)
        dft_out = dft_model(generator_out)

        training_model = Model(inputs=inp, outputs=dft_out)

        training_model.compile(optimizer,
                               loss=loss,
                               metrics=[area_between])
    training_model.summary()

    filepath = os.path.join(base_dir, 'generator_{epoch:03d}.hdf5')
    save_freq=generator_train_size*5*generator_batchsize

    callbacks = [
        ReduceLROnPlateau(monitor='loss',
                          factor=0.1,
                          patience=10),
    ]
    if not use_tpu:
        callbacks.extend([
            TensorBoard(log_dir=log_loc,
                        write_graph=True,
                        write_images=True),
            ModelCheckpoint(filepath,
                            monitor='area_between',
                            save_best_only=True,
                            mode='min',
                            save_freq='epoch')
        ])

    training_dataset = tf.data.Dataset.from_generator(generator_train_generator, (tf.float32, tf.float32),
                                                      output_shapes=(tf.TensorShape([generator_batchsize, N_ADSORP]),
                                                                     tf.TensorShape([generator_batchsize, N_ADSORP])))
    '''
    train_set = [next(generator_train_generator())[0] for _ in range(generator_train_size)]
    train_set = np.concatenate(train_set).astype('float32')
    training_dataset = tf.data.Dataset.from_tensor_slices((train_set, train_set))
    training_dataset = training_dataset.batch(generator_batchsize).repeat()
    '''

    density_files = glob.glob(os.path.join('./data_generation', 'results', 'density_*.csv'))
    density_files.sort(reverse=False)
    density_files = density_files[:]
    true_densities = np.array([np.diff(np.genfromtxt(density_file, delimiter=',')) for density_file in density_files])
    trimmed_down = (len(true_densities) // generator_batchsize)*generator_batchsize
    true_densities = true_densities[:trimmed_down]
    
    print(f'{generator_train_size} curves per epoch')
    print(f'{generator_epochs} epochs')
    print(f'{generator_batchsize} batch size')
    print(f'{inner_loops} loop dft')
    training_model.fit(training_dataset,
                       steps_per_epoch=generator_train_size,
                       epochs=generator_epochs,
                       validation_data=(true_densities, true_densities),
                       validation_batch_size=generator_batchsize,
                       max_queue_size=10, shuffle=False,
                       callbacks=callbacks)

    generator.save(model_loc)


def visualize(see_grids, intermediate_layers):
    generator = inverse_dft_model()
    generator.load_weights(model_loc, by_name=True)

    vis_layers = list()
    for layer in generator.layers:
        if 'conv2d' in layer.name or 'batch_normalization' == layer.name:
            vis_layers.append(Model(generator.input, layer.output))
    
    relative_humidity = np.arange(41) * STEP_SIZE

    areas = list()
    errors = list()

    c = make_steps()[0][::]
    
    square_grids = np.zeros((20, 20, 20), dtype=np.float32)
    for i in range(1, 21):
        square_grids[i-1, :i, :i] = 1
    square_curves, _ = run_dft(square_grids, inner_loops=300)
    c = np.concatenate((square_curves, c))

    straight_line = np.zeros((1, 40), dtype=np.float32)
    straight_line[0, 0:40] = 1/40

    c = np.concatenate((straight_line, c))
    
    grids = generator.predict(c)
    densities, _ = run_dft(grids, inner_loops=300)

    intermediate_layer_outputs = list()
    for layer in vis_layers:
        intermediate = layer.predict(c)
        intermediate_layer_outputs.append(intermediate)
    
    for i, (diffs, grid, diffs_dft, sample1, sample2, sample3) in enumerate(zip(c, grids, densities, *intermediate_layer_outputs)):
        curve = np.cumsum(np.insert(diffs, 0, 0))
        curve_dft = np.cumsum(np.insert(diffs_dft, 0, 0))
        if see_grids:
            '''
            os.makedirs(f'figures/artificial/{i}', exist_ok=True)
            np.savetxt(f'figures/artificial/{i}/grid.csv', grid, delimiter=',')
            np.savetxt(f'figures/artificial/{i}/curve.csv', curve, delimiter=',')
            np.savetxt(f'figures/artificial/{i}/curve_dft.csv', curve_dft, delimiter=',')
            '''
            # show_grid(grid, curve, curve_dft)
            # plt.show()

            '''
            fig, ax = plt.subplots(8, 16, figsize=(16, 8))
            for i in range(128):
                ax[i//16, i % 16].matshow(sample1[:, :, i])
            plt.show()
            fig, ax = plt.subplots(8, 16, figsize=(16, 8))
            for i in range(128):
                ax[i//16, i % 16].matshow(sample2[:, :, i])
            plt.show()
            fig, ax = plt.subplots(8, 16, figsize=(16, 8))
            for i in range(128):
                ax[i//16, i % 16].matshow(sample3[:, :, i])
            plt.show()
            '''

    def vis_curves(curves):
        i = 0
        for c, _ in curves:
            print('computing...')
            curve_batch = np.array(c)
            grids = generator.predict(curve_batch)
            
            intermediate_layer_outputs = list()
            for layer in vis_layers:
                intermediate = layer.predict(curve_batch)
                print(intermediate.shape)
                intermediate_layer_outputs.append(intermediate)
            
            densities, _ = run_dft(grids, inner_loops=100)
            for diffs, grid, diffs_dft, sample1, sample2, sample3 in zip(c, grids, densities, *intermediate_layer_outputs):
                curve = np.cumsum(np.insert(diffs, 0, 0))
                curve_dft = np.cumsum(np.insert(diffs_dft, 0, 0))

                error = np.sum(np.abs(curve - curve_dft)) / len(curve)
                errors.append(error)
                
                area = np.sum(curve_dft) / len(curve_dft)
                areas.append(area)
                
                if see_grids:
                    '''
                    os.makedirs(f'figures/random_curves/{i}', exist_ok=True)
                    np.savetxt(f'figures/random_curves/{i}/grid.csv', grid, delimiter=',')
                    np.savetxt(f'figures/random_curves/{i}/curve.csv', curve, delimiter=',')
                    np.savetxt(f'figures/random_curves/{i}/curve_dft.csv', curve_dft, delimiter=',')
                    '''
                    i += 1
                    show_grid(grid, curve, curve_dft)
                    plt.show()

                    # fig, ax = plt.subplots(8, 16, figsize=(16, 8))
                    # for i in range(128):
                    #     ax[i//16, i % 16].matshow(sample1[:, :, i])
                    # plt.show()
                    # fig, ax = plt.subplots(8, 16, figsize=(16, 8))
                    # for i in range(128):
                    #     ax[i//16, i % 16].matshow(sample2[:, :, i])
                    # plt.show()
                    # fig, ax = plt.subplots(8, 16, figsize=(16, 8))
                    # for i in range(128):
                    #     ax[i//16, i % 16].matshow(sample3[:, :, i])
                    # plt.show()

    # curves = [next(generator_train_generator()) for _ in range(5)]
    # vis_curves(curves)

    # base_dir = './test_grids/step4'
    base_dir = './data_generation'
    density_files = glob.glob(os.path.join(base_dir, 'results', 'density_*.csv'))
    density_files.sort(reverse=False)
    density_files = density_files[:]
    true_densities = [np.diff(np.genfromtxt(density_file, delimiter=',')) for density_file in density_files]
    shuffle(true_densities)
    # true_densities = batch(true_densities, generator_batchsize)
    true_densities = batch(true_densities, 256)
    vis_curves(zip(true_densities, true_densities))

    print('Mean: ', np.array(errors).mean())
    print('Std: ', np.array(errors).std())
    plt.hist(errors, bins=15)
    plt.title('Error Distribution')
    plt.xlabel('Abs error')
    # plt.xlim(0, 1)
    plt.show()
    
    plt.scatter(areas, errors)
    print(np.array(areas).shape)
    print(np.array(areas).dtype)
    print(np.array(errors).shape)
    print(np.array(errors).dtype)
    os.makedirs(f'figures/generator_error/', exist_ok=True)
    np.savetxt('figures/generator_error/errors.csv', np.concatenate((np.array(areas)[:, None], np.array(errors)[:, None]), axis=1), delimiter=',')
    plt.title('Error w.r.t. area under curve')
    plt.xlabel('Area under DFT curve')
    plt.ylabel('Abs error')
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 1 and 'tpu' not in sys.argv:
        see_grids = len(sys.argv) >= 3
        intermediate_layers = len(sys.argv) >= 4
        visualize(see_grids, intermediate_layers)
    else:
        train('tpu' in sys.argv)
