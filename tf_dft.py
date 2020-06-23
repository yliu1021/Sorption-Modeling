import sys
import time
import math
import glob
import os

import numpy as np
import tensorflow as tf

from constants import *
import simul_dft

import matplotlib.pyplot as plt


def run_dft(grids, batch_size=None, inner_loops=5):
    """Runs the DFT simulation on a batch of grids

    Parameters
    ----------
    grids : This must be a tensor of shape [batch_size, GRID_SIZE, GRID_SIZE]
    """
    
    muu_lookup = list()
    for jj in range(N_ITER + 1):
        if jj <= N_ADSORP:
            RH = jj * STEP_SIZE
        else:
            RH = N_ADSORP*STEP_SIZE - (jj-N_ADSORP)*STEP_SIZE
        if RH == 0:
            muu = -90.0
        else:
            muu = MUSAT+KB*T*math.log(RH)
        muu_lookup.append(muu)

    _filter_wffy = tf.constant(
        [[[[0]], [[WFF * Y]], [[0]]],
         [[[WFF * Y]], [[0]], [[WFF * Y]]],
         [[[0]], [[WFF * Y]], [[0]]]],
        dtype=tf.float32
    )
    
    _filter_wff = tf.constant(
        [[[[0]], [[WFF * BETA]], [[0]]],
         [[[WFF * BETA]], [[0]], [[WFF * BETA]]],
         [[[0]], [[WFF * BETA]], [[0]]]],
        dtype=tf.float32
    )
    
    _filter_y = tf.constant(
        [[[[0]], [[Y]], [[0]]],
         [[[Y]], [[0]], [[Y]]],
         [[[0]], [[Y]], [[0]]]],
        dtype=tf.float32
    )
    
    _filter_1 = tf.constant(
        [[[[0]], [[1]], [[0]]],
         [[[1]], [[0]], [[1]]],
         [[[0]], [[1]], [[0]]]],
        dtype=tf.float32
    )
    
    # we tile the grid and then crop it so that the boundaries
    # from one side will also exist on the other side
    batch_size = grids.shape[0]

    r0 = tf.tile(grids, [1, 3, 3])
    r1 = tf.tile(grids, [1, 3, 3])
    rneg = 1 - r0

    r0 = r0[:, GRID_SIZE:2*GRID_SIZE, GRID_SIZE:2*GRID_SIZE]
    r1 = r1[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1]
    rneg = rneg[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1]

    r0 = tf.expand_dims(r0, -1)
    r1 = tf.expand_dims(r1, -1)
    rneg = tf.expand_dims(rneg, -1)

    total_pores = tf.maximum(tf.reduce_sum(grids, [1, 2]), 1)
    
    rs = list()
    
    densities = [tf.zeros(batch_size)]
    for jj in range(1, N_ADSORP):
        for i in range(inner_loops):
            vir1 = tf.nn.conv2d(r1, strides=[1,1,1,1], filters=_filter_1, padding='VALID')
            vir0 = tf.nn.conv2d(rneg, strides=[1,1,1,1], filters=_filter_y, padding='VALID')
            vi = WFF * (vir1 + vir0) + muu_lookup[jj];

            rounew = r0 * tf.nn.sigmoid(BETA * vi)

            r1 = tf.tile(rounew, [1, 3, 3, 1])
            r1 = r1[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1, :]
        rs.append(r1)

        density = tf.truediv(tf.reduce_sum(r1[:, 1:GRID_SIZE+1, 1:GRID_SIZE+1, :], axis=[1, 2, 3]), total_pores)
        densities.append(density)
    densities.append(tf.ones(batch_size))

    densities = tf.stack(densities, axis=1)
    return densities[:, 1:] - densities[:, :-1], rs


def run_dft_fast(grids, batch_size=None, inner_loops=5):
    """Runs the DFT simulation on a batch of grids

    Parameters
    ----------
    grids : This must be a tensor of shape [batch_size, GRID_SIZE, GRID_SIZE]
    """
    
    muu_lookup = list()
    for jj in range(N_ITER + 1):
        if jj <= N_ADSORP:
            RH = jj * STEP_SIZE
        else:
            RH = N_ADSORP*STEP_SIZE - (jj-N_ADSORP)*STEP_SIZE
        if RH == 0:
            muu = -90.0
        else:
            muu = MUSAT+KB*T*math.log(RH)
        muu_lookup.append(muu)

    _filter_wffy = tf.constant(
        [[[[0]], [[WFF * Y]], [[0]]],
         [[[WFF * Y]], [[0]], [[WFF * Y]]],
         [[[0]], [[WFF * Y]], [[0]]]],
        dtype=tf.float32
    )
    
    _filter_wff = tf.constant(
        [[[[0]], [[WFF * BETA]], [[0]]],
         [[[WFF * BETA]], [[0]], [[WFF * BETA]]],
         [[[0]], [[WFF * BETA]], [[0]]]],
        dtype=tf.float32
    )
    
    _filter_y = tf.constant(
        [[[[0]], [[Y]], [[0]]],
         [[[Y]], [[0]], [[Y]]],
         [[[0]], [[Y]], [[0]]]],
        dtype=tf.float32
    )
    
    _filter_1 = tf.constant(
        [[[[0]], [[1]], [[0]]],
         [[[1]], [[0]], [[1]]],
         [[[0]], [[1]], [[0]]]],
        dtype=tf.float32
    )
    
    # we tile the grid and then crop it so that the boundaries
    # from one side will also exist on the other side
    batch_size = grids.shape[0]

    r0 = tf.tile(grids, [1, 3, 3])
    r1 = tf.tile(grids, [1, 3, 3])
    rneg = 1 - r0

    r0 = r0[:, GRID_SIZE:2*GRID_SIZE, GRID_SIZE:2*GRID_SIZE]
    r1 = r1[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1]
    rneg = rneg[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1]

    r0 = tf.expand_dims(r0, -1)
    r1 = tf.expand_dims(r1, -1)
    rneg = tf.expand_dims(rneg, -1)

    total_pores = tf.maximum(tf.reduce_sum(grids, [1, 2]), 1)
    
    rs = list()
    
    densities = [tf.zeros(batch_size)]
    for jj in range(1, N_ADSORP):
        for i in range(inner_loops):
            vir1 = tf.nn.conv2d(r1, strides=[1,1,1,1], filters=_filter_1, padding='VALID')
            vir0 = tf.nn.conv2d(rneg, strides=[1,1,1,1], filters=_filter_y, padding='VALID')
            vi = WFF * (vir1 + vir0) + muu_lookup[jj];

            rounew = r0 * tf.nn.sigmoid(BETA * vi)

            r1 = tf.tile(rounew, [1, 3, 3, 1])
            r1 = r1[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1, :]
        rs.append(r1)

        density = tf.truediv(tf.reduce_sum(r1[:, 1:GRID_SIZE+1, 1:GRID_SIZE+1, :], axis=[1, 2, 3]), total_pores)
        densities.append(density)
    densities.append(tf.ones(batch_size))

    densities = tf.stack(densities, axis=1)
    return densities[:, 1:] - densities[:, :-1], rs


def run_dft_pad(grids, batch_size=None, inner_loops=5):
    """Runs the DFT simulation on a batch of grids

    Parameters
    ----------
    grids : This must be a tensor of shape [batch_size, GRID_SIZE, GRID_SIZE]
    """

    muu_lookup = list()
    for jj in range(N_ITER + 1):
        if jj <= N_ADSORP:
            RH = jj * STEP_SIZE
        else:
            RH = N_ADSORP*STEP_SIZE - (jj-N_ADSORP)*STEP_SIZE
        if RH == 0:
            muu = -90.0
        else:
            muu = MUSAT+KB*T*math.log(RH)
        muu_lookup.append(muu)

    _filter_wffy = tf.constant(
        [[[[0]], [[WFF * Y]], [[0]]],
         [[[WFF * Y]], [[0]], [[WFF * Y]]],
         [[[0]], [[WFF * Y]], [[0]]]],
        dtype=tf.float32
    )
    
    _filter_wff = tf.constant(
        [[[[0]], [[WFF * BETA]], [[0]]],
         [[[WFF * BETA]], [[0]], [[WFF * BETA]]],
         [[[0]], [[WFF * BETA]], [[0]]]],
        dtype=tf.float32
    )
    
    _filter_y = tf.constant(
        [[[[0]], [[Y]], [[0]]],
         [[[Y]], [[0]], [[Y]]],
         [[[0]], [[Y]], [[0]]]],
        dtype=tf.float32
    )
    
    _filter_1 = tf.constant(
        [[[[0]], [[1]], [[0]]],
         [[[1]], [[0]], [[1]]],
         [[[0]], [[1]], [[0]]]],
        dtype=tf.float32
    )
    
    # we tile the grid and then crop it so that the boundaries
    # from one side will also exist on the other side
    batch_size = grids.shape[0]

    r0 = tf.tile(grids, [1, 3, 3])
    r1 = tf.tile(grids, [1, 3, 3])
    rneg = 1 - r0

    r0 = r0[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1]
    r1 = r1[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1]
    rneg = rneg[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1]

    r0 = tf.reshape(r0, [batch_size, GRID_SIZE+2, GRID_SIZE+2, 1])
    r1 = tf.reshape(r1, [batch_size, GRID_SIZE+2, GRID_SIZE+2, 1])
    rneg = tf.reshape(rneg, [batch_size, GRID_SIZE+2, GRID_SIZE+2, 1])

    total_pores = tf.maximum(tf.reduce_sum(grids, [1, 2]), 1)

    densities = [tf.zeros(batch_size)]
    for jj in range(1, N_ADSORP):
        for i in range(inner_loops):
            vir1 = tf.nn.conv2d(r1, strides=[1,1,1,1], filters=_filter_1, padding='SAME')
            vir0 = tf.nn.conv2d(rneg, strides=[1,1,1,1], filters=_filter_y, padding='SAME')
            vi = WFF * (vir1 + vir0) + muu_lookup[jj];

            rounew = r0 * tf.nn.sigmoid(BETA * vi)

            # r1 = tf.tile(rounew, [1, 3, 3, 1])
            # r1 = r1[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1, :]

        density = tf.truediv(tf.reduce_sum(r1[:, 1:GRID_SIZE+1, 1:GRID_SIZE+1, :], axis=[1, 2, 3]), total_pores)
        densities.append(density)
    densities.append(tf.ones(batch_size))

    diffs = list()
    last = densities[0]
    for density in densities[1:]:
        diffs.append(density - last)
        last = density
    return tf.stack(diffs, axis=1)


def tile(grid, size):
    """
    Tile grid with size x size pores
    """
    height, width = grid.shape
    for i in range(0, height, 2*size):
        for j in range(0, width, 2*size):
            grid[i:i+size, j:j+size] = 1
            grid[i+size:i+2*size, j+size:j+2*size] = 1
    return int(np.sum(grid))


if __name__ == '__main__':
    inner_loops = 5
    try:
        inner_loops = int(sys.argv[1])
    except:
        pass

    while True:
        grids = np.zeros((GRID_SIZE - 1, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for i in range(GRID_SIZE - 1):
            count = tile(grids[i, -(i+1):, -(i+1):], 5)
            grids[i, :(count+1)//5, :5] = 1
        print(grids.shape)
        
        dft_curves, _ = run_dft(grids, inner_loops=inner_loops)
        for grid, curve in zip(grids, dft_curves):
            print(curve.numpy())
            
            fig = plt.figure(figsize=(10,4))
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            ax = plt.subplot(1, 2, 1)
            ax.clear()
            ax.set_title('Grid (Black = Solid, Whited = Pore)')
            ax.set_yticks(np.linspace(0, GRID_SIZE, 5))
            ax.set_xticks(np.linspace(0, GRID_SIZE, 5))
            ax.pcolor(1 - grid, cmap='Greys', vmin=0.0, vmax=1.0)
            ax.set_aspect('equal')

            actual_curve = np.insert(np.cumsum(curve), 0, 0)
            ax = plt.subplot(1, 2, 2)
            ax.clear()
            ax.set_title('Adsorption Curve')
            ax.set_xticks(np.linspace(0, 1, 10))
            ax.set_yticks(np.linspace(0, 1, 5))
            ax.set_ylim(0, 1)
            ax.plot(np.linspace(0, 1, N_ADSORP+1), actual_curve)
            ax.scatter(np.linspace(0, 1, N_ADSORP), curve)
            
            plt.show()

    # base_dir = '/Users/yuhanliu/Google Drive/Research/sorption_modeling/test_grids/step4'
    base_dir = './data_generation/'
    grid_files = glob.glob(os.path.join(base_dir, 'grids', 'grid_*.csv'))
    grid_files.sort(reverse=False)

    grid_files = grid_files[:]

    grids = [np.genfromtxt(grid_file, delimiter=',', dtype=np.float32) for grid_file in grid_files]
    print('Num grids: ', len(grids))
    start_time = time.time()
    densities, rs = run_dft(np.array(grids), inner_loops=inner_loops)
    end_time = time.time()
    print('Time: ', end_time - start_time)
    print('Grids per second: ', len(grids) / (end_time - start_time))

    '''
    for i in range(len(grids)):
        for j, step in enumerate(rs):
            fig = plt.figure(figsize=(10, 4))
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            ax = plt.subplot(1, 2, 1)
            ax.clear()
            ax.set_title('Grid (Black = Solid, White = Pore)')
            ax.set_yticks(np.linspace(0, 20, 5))
            ax.set_xticks(np.linspace(0, 20, 5))
            ax.pcolor(1 - grids[i], cmap='Greys', vmin=0.0, vmax=1.0)
            ax.set_aspect('equal')

            ax = plt.subplot(1, 2, 2)
            ax.clear()
            ax.set_title(f'Water Content (RH: {(j+1)/40:0.2})')
            ax.set_yticks(np.linspace(0, 22, 5)) 
            ax.set_xticks(np.linspace(0, 22, 5))
            ax.pcolor(1 - step[i, :, :, 0])
            plt.show()
    '''
    
    density_files = glob.glob(os.path.join(base_dir, 'results', 'density_*.csv'))
    density_files.sort(reverse=False)
    density_files = density_files[:]
    true_densities = [np.genfromtxt(density_file, delimiter=',') for density_file in density_files]

    areas = list()
    errors = list()
    for i, (d, t) in enumerate(zip(densities, true_densities)):
        x = np.linspace(0, 1, N_ADSORP + 1)
        area = np.sum(t) / len(t)
        error = np.sum(np.abs(np.cumsum(np.insert(d, 0, 0)) - t)) / len(t)
        areas.append(area)
        errors.append(error)

        plt.title('Cont. vs Discrete'.format(i+1))
        plt.plot(x, np.cumsum(np.insert(d, 0, 0)), label='Continuous')
        plt.plot(x, t, label='Discrete')
        plt.legend()
        plt.show()

    # plt.scatter(*zip(*points))
    print('Error: ', np.array(errors).mean())
    print('Error std dev: ', np.array(errors).std())

    plt.scatter(areas, errors)
    plt.title('Inner loops: {}'.format(inner_loops))
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel('Area under actual DFT curve')
    plt.ylabel('Abs area between TensorFlow-DFT curve and DFT curve')
    plt.show()

    plt.scatter(areas, errors)
    plt.title('Inner loops: {}'.format(inner_loops))
    plt.xlim(0, 1)
    plt.xlabel('Area under actual DFT curve')
    plt.ylabel('Abs area between TensorFlow-DFT curve and DFT curve')
    plt.show()
