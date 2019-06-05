import numpy as np
import math
import random
import os
import sys

from keras.models import Sequential, load_model

from constants import *
import grids, train, stats
from simul_dft import run_dft

# initialization stage

network_func = {
    'nn': train.single_layer_model,
    'cnn': train.simple_cnn,
    # 'lin': train.lin_reg_model,  # doesn't fit format because not a neural net
}

def load_first_model(model_type, save=False):
    if VERBOSE:
        print('Loading initial blank model')
    model = network_func[model_type]()
    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=['accuracy'])
    if save:
        out_path = os.path.join(PREDICT_DIR, 'model_v00.h5')
        model.save(out_path)
    return model

def load_initial_train(setlist, version, fraction, save=False,
                       hys_dir=HYS_DIR, train_set_dir=TRAIN_SET_DIR, grid_dir=GRID_DIR):
    if VERBOSE:
        print('Loading initial training data')
    (grids, results) = train.load_data(setlist, hys_dir, version, fraction, grid_dir=grid_dir)
    n_grids = grids.shape[0]
    indexed_results = np.zeros((n_grids,2))
    indexed_results[:,0] = np.reshape(np.arange(n_grids), (n_grids,))
    indexed_results[:,1] = results
    if save:
        out_path = os.path.join(train_set_dir, 'train_grids00.csv')
        np.savetxt(out_path, grids, fmt='%i', delimiter=',')
        out_path = os.path.join(train_set_dir, 'train_results00.csv')
        np.savetxt(out_path, indexed_results, delimiter=',')
    return (grids, indexed_results)

# single step in process:

pore_functions = {
    'random': grids.generate_grid,
    'growing': grids.alt_generate_grid,
    'square': grids.rand_sq_generate_grid,
}

# TODO: Train model on a combination of the inputs

# train neural network on training set
def train_net(c_iter, model, gridset=None, resultset=None, save=False):
    if VERBOSE:
        print('Training model, iteration %i'%c_iter)
    if gridset is None:
        in_path = os.path.join(TRAIN_SET_DIR, 'train_grids%02d.csv'%c_iter)
        gridset = np.loadtxt(in_path, dtype='int', delimiter=',')
    if resultset is None:
        in_path = os.path.join(TRAIN_SET_DIR, 'train_results%02d.csv'%c_iter)
        resultset = np.loadtxt(in_path, dtype='float', delimiter=',')
    model.fit(gridset, resultset[:,1], epochs=100, batch_size=32)
    if save:
        out_path = os.path.join(PREDICT_DIR, 'model_v%02d.h5'%(c_iter+1))
        model.save(out_path)
    return model

# generate some random pores
def generate_pores(c_iter, n_grids, save=False):
    if VERBOSE:
        print('Generating pores, iteration %i'%c_iter)
    gridset = np.zeros((n_grids, GRID_SIZE * GRID_SIZE))
    for i in range(n_grids):
        if i % 1000 == 0:
            print('iter: ' + str(i))
        out_dir = os.path.join(TRAIN_SET_DIR, 'grids%02d'%c_iter)  # not used
        pore_choice = random.choice(PORE_CHOICES)
        (r, s, f, grid) = grids.transform_grid(i, GRID_SIZE, pore_functions[pore_choice](
            N_FILLED, THRESH, N_VAR, GRID_SIZE), False, out_dir, False)
        gridset[i,:] = grid.reshape((1, GRID_SIZE*GRID_SIZE))
    if save:
        out_path = os.path.join(RESULT_SET_DIR, 'gridset%02d.csv'%(c_iter+1))
        np.savetxt(out_path, gridset, fmt='%i', delimiter=',')
    return gridset

# run neural network on random pores
def eval_net(c_iter, model, gridset=None, save=False):
    if VERBOSE:
        print('Evaluating model, iteration %i'%c_iter)
    if gridset is None:
        in_path = os.path.join(RESULT_SET_DIR, 'gridset%02d.csv'%(c_iter+1))
        gridset = np.loadtxt(in_path, dtype='int', delimiter=',')
    n_grids = gridset.shape[0]
    results = np.zeros((n_grids, 2))
    results[:,0] = np.reshape(np.arange(n_grids), (n_grids,))
    results[:,1] = model.predict(gridset, verbose=1)[:,0]
    if save:
        out_path = os.path.join(RESULT_SET_DIR, 'resultset%02d.csv'%(c_iter+1))
        np.savetxt(out_path, results, delimiter=',')
    return results

hys_funcs = {
    'l': stats.linedist,
    'm': stats.loglinedist,
}

# pick out best results, augment training set
def run_simul(c_iter, version, gridset=None, resultset=None, save=False):
    if VERBOSE:
        print('Running simulation, iteration %i'%c_iter)
    if gridset is None:
        in_path = os.path.join(TRAIN_SET_DIR, 'train_grids%02d.csv'%(c_iter+1))
        gridset = np.loadtxt(in_path, dtype='int', delimiter=',')
    if resultset is None:
        in_path = os.path.join(RESULT_SET_DIR, 'optset%02d.csv'%(c_iter+1))
        resultset = np.loadtxt(in_path, delimiter=',')
    n_grids = gridset.shape[0]
    results = np.zeros((n_grids,2))
    results[:,0] = resultset[:,0]
    full_density = np.zeros((n_grids, N_ITER+1))
    for i in range(n_grids):
        if VERBOSE and i % 50 == 0:
            print('iter: ' + str(i))
        density = np.zeros((N_ITER+1,2))
        density[:,0] = np.reshape(np.arange(N_ITER+1), (N_ITER+1,))
        density[:,1] = run_dft(gridset[i,:])[:,0]
        full_density[i,:] = np.reshape(density[:,1], (1, N_ITER+1))
        results[i,1] = hys_funcs[version](gridset[i,:], density, stats.pore_size)
    if save:
        out_path = os.path.join(TRAIN_SET_DIR, 'train_results%02d.csv'%(c_iter+1))
        np.savetxt(out_path, results, delimiter=',')
        out_path = os.path.join(TRAIN_SET_DIR, 'train_density%02d.csv'%(c_iter+1))
        np.savetxt(out_path, full_density, delimiter=',')
    return results, full_density

# repeat n times
def run_n_iters(n_steps, n_gen, n_opt, model_type, setlist, version, fraction, save=False):
    model = load_first_model(model_type, save)
    (grids, results) = load_initial_train(setlist, version, fraction, save)
    train_grids = np.zeros((0, GRID_SIZE*GRID_SIZE))
    train_results = np.zeros((0, 2))
    for i in range(0, n_steps):
        train_grids = np.append(train_grids, grids, axis=0)
        train_results = np.append(train_results, results, axis=0)
        model = train_net(i, model, train_grids, train_results, save)
        grids = generate_pores(i, n_gen, save)
        results = eval_net(i, model, grids, save)
        indices = stats.find_min_indices(n_opt, grids, results)
        grids = grids[indices,:]
        results = results[indices,:]
        # print(indices)
        # print(grids)
        if save:
            out_path = os.path.join(TRAIN_SET_DIR, 'train_grids%02d.csv'%(i+1))
            np.savetxt(out_path, grids, fmt='%i', delimiter=',')
            out_path = os.path.join(RESULT_SET_DIR, 'optset%02d.csv'%(i+1))
            np.savetxt(out_path, results, delimiter=',')
        results, full_density = run_simul(i, version, grids, results, save)


if __name__ == '__main__':
    os.makedirs(PREDICT_DIR, exist_ok='True')
    os.makedirs(TRAIN_SET_DIR, exist_ok='True')
    os.makedirs(RESULT_SET_DIR, exist_ok='True')
    run_n_iters(N_STEPS, N_GENERATED, N_OPTIMAL, 'cnn', DATASETS, 'm', TRAIN_FRAC, True)




