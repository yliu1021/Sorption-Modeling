import time
import math
import glob
import os

import numpy as np
import tensorflow as tf

from constants import *
import simul_dft

import matplotlib.pyplot as plt

muu_lookup = list()
for jj in range(N_ITER + 1):
    if jj <= N_ADSORP:
        RH = jj * STEP_SIZE
    else:
        RH = N_ADSORP*STEP_SIZE - (jj-N_ADSORP)*STEP_SIZE
    if RH == 0:
        muu = -90
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

def run_dft(grids, batch_size=None):
    """Runs the DFT simulation on a batch of grids
    
    Parameters
    ----------
    grids : This must be a tensor of shape [batch_size, GRID_SIZE, GRID_SIZE]
    """
    
    # we tile the grid and then crop it so that the boundaries
    # from one side will also exist on the other side
    batch_size = len(grids) if batch_size is None else batch_size
    Ntotal_pores = tf.clip_by_value(tf.reduce_sum(grids, axis=[1, 2]), 1, 400)

    r0 = tf.tile(grids, [1, 3, 3])
    r1 = tf.tile(grids, [1, 3, 3])
    rneg = 1 - r0

    r0 = r0[:, GRID_SIZE:2*GRID_SIZE, GRID_SIZE:2*GRID_SIZE]
    r1 = r1[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1]
    rneg = rneg[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1]

    r0 = tf.reshape(r0, [batch_size, GRID_SIZE, GRID_SIZE, 1])
    r1 = tf.reshape(r1, [batch_size, GRID_SIZE+2, GRID_SIZE+2, 1])
    rneg = tf.reshape(rneg, [batch_size, GRID_SIZE+2, GRID_SIZE+2, 1])

    wffyr0_conv = tf.nn.conv2d(rneg, strides=[1, 1, 1, 1], filters=_filter_wffy, padding='VALID')

    densities = [tf.zeros(batch_size)]
    for jj in range(1, N_ADSORP):
        bias = (wffyr0_conv + muu_lookup[jj]) * BETA
        for i in range(160):
            vi = tf.nn.conv2d(r1, strides=[1, 1, 1, 1], filters=_filter_wff, padding='VALID',
                              name='vi_conv_%04d'%i)
            vi += bias
            
            rounew = r0 * tf.nn.sigmoid(vi)
            r1 = tf.tile(rounew, [1, 3, 3, 1])
            r1 = r1[:, GRID_SIZE-1:2*GRID_SIZE+1, GRID_SIZE-1:2*GRID_SIZE+1, :]

        density = tf.clip_by_value(tf.reduce_mean(r1, axis=[1, 2, 3]), 0, 1)
        densities.append(density)
    densities.append(tf.ones(batch_size))
    diffs = list()
    last = densities[0]
    for density in densities[1:]:
        diffs.append(density - last)
        last = density
    return tf.stack(diffs, axis=1)


if __name__ == '__main__':
    # grid_tf = tf.compat.v1.placeholder(tf.float32, shape=[462, GRID_SIZE, GRID_SIZE], name='input_grid')
    # density_tf = run_dft(grid_tf)

    base_dir = '/Users/yuhanliu/Google Drive/1st year/Research/sorption_modeling/test_grids/step4'
    grid_files = glob.glob(os.path.join(base_dir, 'grids', 'grid_*.csv'))
    grid_files.sort(reverse=True)
    grids = [np.genfromtxt(grid_file, delimiter=',', dtype=np.float32) for grid_file in grid_files]
    print(len(grids))
    start = time.time()
    densities = run_dft(grids)
    # densities = sess.run(density_tf, feed_dict={grid_tf: grids})
    end = time.time()
    print(densities)
    print(end - start)

    density_files = glob.glob(os.path.join(base_dir, 'results', 'density_*.csv'))
    density_files.sort(reverse=True)
    true_densities = [np.genfromtxt(density_file, delimiter=',') for density_file in density_files]

    for i, (d, t) in enumerate(zip(densities, true_densities)):
        x = np.linspace(0, 1, N_ADSORP + 1)
        plt.title('{}'.format(i+1))
        plt.plot(x, np.cumsum(np.insert(d, 0, 0)), label='tf')
        plt.plot(x, t, label='dft')
        plt.legend()
        plt.show()
