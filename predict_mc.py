import numpy as np
import math
import random
import os
import sys

from keras.models import Sequential, load_model
from scipy.ndimage import measurements

from constants import *
import grids, stats, predict
from simul_dft import run_dft

def count_adjacent(grid, x, y):
    adj = 0
    if grid[x-1, y] == 1:
        adj += 1
    if grid[x+1, y] == 1:
        adj += 1
    if grid[x, y-1] == 1:
        adj += 1
    if grid[x, y+1] == 1:
        adj += 1
    return adj

# Fill one pore in the grid, potentially retransforming.
def fill_one_pore(grid, transform=False, n_failed_changes=0):
    grid = np.array(grid)
    for i in range(MAX_ATTEMPTS):
        x = random.randint(THRESH, GRID_SIZE - THRESH - 1)
        y = random.randint(THRESH, GRID_SIZE - THRESH - 1)
        if grid[x, y] == 0 and count_adjacent(grid, x, y) > 0:
            grid[x, y] = 1
            break
    else:
        # print('Failed to make change: %i'%n_failed_changes)
        return grid, False
    return grid, True

# Empty one pore in the grid, potentially retransforming.
def empty_one_pore(grid, transform=False, n_failed_changes=0):
    grid = np.array(grid)
    for i in range(MAX_ATTEMPTS):
        x = random.randint(THRESH, GRID_SIZE - THRESH - 1)
        y = random.randint(THRESH, GRID_SIZE - THRESH - 1)
        if grid[x, y] == 1: # and count_adjacent(grid, x, y) < 4:
            grid[x, y] = 0
            labels, n_clusters = measurements.label(grid)
            if n_clusters == 1: # don't ruin connected components
                break
            grid[x, y] = 1
    else:
        # print('Failed to make change: %i'%n_failed_changes)
        return grid, False
    return grid, True

def prob_explore(diff, sigma):
    return math.exp(-diff/sigma)

def model_cost(grid, model=None):
    if model == 'dft' or model is None:
        density = np.zeros((N_ITER+1,2))
        density[:,0] = np.reshape(np.arange(N_ITER+1), (N_ITER+1,))
        if grid.shape != (1, GRID_SIZE ** 2):
            grid = np.reshape(grid, (GRID_SIZE ** 2,))
        density[:,1] = run_dft(grid)[:,0]
        return stats.loglinedist(grid, density, stats.pore_size)
    else:
        if grid.shape == (1, GRID_SIZE ** 2):
            return model.predict(grid)[:,0]
        return model.predict(np.reshape(grid, (1, GRID_SIZE ** 2)))[:,0]

# Return the probability of acceptance.
def prob_accept(model, old_grid, new_grid, sigma):
    old_res = model_cost(old_grid, model)
    new_res = model_cost(new_grid, model)
    diff = new_res - old_res
    res = np.array([old_res, new_res, diff, min(prob_explore(diff, sigma), 1.0), sigma, 0])
    res = res.flatten().astype('float')
    if diff < 0:
        return 1.01, res
    return prob_explore(diff, sigma), res

# Run the MC simulation based on a grid.
def run_mc(grid, n_steps, model=None, save=True, update_period=1, reset_rate=N_MC_ITER*2): 
    gridset = np.zeros((n_steps + 1, GRID_SIZE ** 2))
    statset = np.zeros((n_steps + 1, N_RES_PARAMS))
    n_failed_changes = 0
    avg_accept_rate = MC_ACCEPT
    iter_count = 0
    sigma = INIT_SIGMA
    for i in range(n_steps):
        if random.random() < MC_GROW_PROB:
            pore_method = fill_one_pore
        else:
            pore_method = empty_one_pore
        n_grid, changed = pore_method(grid, False, n_failed_changes)
        if not changed:
            n_failed_changes += 1
        accept_prob, res = prob_accept(model, grid, n_grid, sigma)
        res[N_RES_PARAMS - 1] = avg_accept_rate
        print(res)
        gridset[i,:] = np.reshape(grid, (1,GRID_SIZE ** 2))
        statset[i,:] = np.reshape(res, (1,N_RES_PARAMS))
        if random.random() < accept_prob:
            grid = n_grid
            accepted = 1
        else:
            accepted = 0
        if accept_prob <= 1:
            avg_accept_rate = (avg_accept_rate*iter_count + accepted) / (iter_count+1)
            iter_count += 1
            if i % update_period == 0:
                sigma += -1 * INIT_ALPHA * (avg_accept_rate - MC_ACCEPT)
        if i % reset_rate == 0:
            avg_accept_rate = MC_ACCEPT
            iter_count = 0
    if save:
        path = os.path.join(MC_GRID_DIR, 'gridset.csv')
        np.savetxt(path, gridset, delimiter=',')
        path = os.path.join(MC_GRID_DIR, 'results.csv')
        np.savetxt(path, statset, delimiter=',')
    return grid

def load_min_grid():
    os.makedirs(MC_GRID_DIR, exist_ok=True)
    grids, results = predict.load_initial_train(DATASETS, 'm', TRAIN_FRAC, 
                                                False, MC_HYS_DIR, None, MC_SRC_GRID_DIR)
    min_indices = stats.find_min_indices(1, grids, results)
    print(min_indices)
    print(results[min_indices])
    grid = np.reshape(grids[min_indices[0]], (GRID_SIZE, GRID_SIZE))
    print(grid)
    return grid

# Some hardcoded stuff testing the thing on a model. 
def test_model(update_period=1, reset_rate=N_MC_ITER):
    # model = load_model(os.path.join(MCM_BASE_DIR, 'cnn_123-70.h5'))
    model = 'dft'
    grid = load_min_grid()
    # print(model_cost(grid))
    run_mc(grid, N_MC_ITER, model=model, update_period=update_period, reset_rate=reset_rate)
    # print(model_cost(grid))


if __name__=='__main__':
    test_model(100, 100)


