import numpy as np
import math
import random
import os
import sys

from keras.models import Sequential, load_model

from constants import *
import grids, predict, predict_mc

def random_init_positions(n_bees, grid_size):
    # can be modified to something else if we want
    gridset = predict.generate_pores(0, n_bees)
    return gridset

def random_init_velocities(n_bees, grid_size):
    v_min, v_max = sigmoid_bounds(SWARM_PMIN, SWARM_PMAX)
    velocity = (v_max / 3) * np.random.randn(n_bees, grid_size ** 2)
    return velocity

def model_cost(n_bees, gridset, model, reject_invalid=False):
    # 100 is an arbitrary large number
    resultset = 100 * np.ones((n_bees,))
    n_bees = gridset.shape[0]
    for i in range(n_bees):
        if not reject_invalid or grids.validate_grid(gridset[i]):
            resultset[i] = predict_mc.model_cost(gridset[i], model)
    return resultset

def update_velocity(best_p, best_g, position, velocity):
    p_contrib = P_WEIGHT * random.random() * (best_p - position)
    g_contrib = G_WEIGHT * random.random() * (best_g - position)
    return velocity + p_contrib + g_contrib

def update_position(position, velocity):
    sigmoid = np.reciprocal(1 + np.exp(-velocity))
    rand = np.random.rand(*position.shape)
    return (rand < velocity).astype('int')

def swarm_search(n_bees, n_iter, grid_size, model, init_pos=None, init_vel=None, 
                 save=True, out_dir=SWARM_DIR, pos_dir=SWARM_POS_DIR,
                 vel_dir=SWARM_VEL_DIR, best_dir=SWARM_BEST_DIR):
    position, velocity = init_pos, init_vel
    if init_pos is None:
        position = random_init_positions(n_bees, grid_size)
    if init_vel is None:
        velocity = random_init_velocities(n_bees, grid_size)
    personal_bests = position
    global_best = position[0]
    # 100 is an arbitrary large number
    min_costs = 100 * np.ones((n_bees,))
    best_cost = 100
    cost_set = np.zeros((n_iter, n_bees))
    min_cost_set = np.zeros((n_iter, n_bees))
    best_pos_set = np.zeros((n_iter, grid_size ** 2))
    best_cost_set = np.zeros((n_iter,))
    for i in range(n_iter):
        print('Iteration ' + str(i))
        if save:
            np.savetxt(pos_dir + '/pos%04d.csv'%i, position, fmt='%i', delimiter=',')
            np.savetxt(vel_dir + '/vel%04d.csv'%i, velocity, delimiter=',')
            np.savetxt(best_dir + '/personal%04d.csv'%i, personal_bests, fmt='%i', delimiter=',')
        costs = model_cost(n_bees, position, model)
        for b in range(n_bees):
            if costs[b] < min_costs[b]:
                min_costs[b] = costs[b]
                personal_bests[b] = position[b]
            if costs[b] < best_cost:
                best_cost = costs[b]
                global_best = position[b]
        cost_set[i] = np.reshape(costs, (1, n_bees))
        min_cost_set[i] = np.reshape(min_costs, (1, n_bees))
        best_pos_set[i] = np.reshape(global_best, (1, grid_size ** 2))
        best_cost_set[i] = best_cost
        velocity = update_velocity(personal_bests, global_best, position, velocity)
        position = update_position(position, velocity)
    if save:
        np.savetxt(out_dir + '/costs.csv', cost_set, delimiter=',')
        np.savetxt(out_dir + '/min_costs.csv', min_cost_set, delimiter=',')
        np.savetxt(out_dir + '/bests.csv', best_pos_set, fmt='%i', delimiter=',')
        np.savetxt(out_dir + '/best_costs.csv', best_cost_set, delimiter=',')

def sigmoid_bounds(p_min, p_max):
    v_min = -math.log(1/p_min - 1)
    v_max = -math.log(1/p_max - 1)
    return v_min, v_max


if __name__ == '__main__':
    os.makedirs(SWARM_DIR, exist_ok=True)
    os.makedirs(SWARM_POS_DIR, exist_ok=True)
    os.makedirs(SWARM_VEL_DIR, exist_ok=True)
    os.makedirs(SWARM_BEST_DIR, exist_ok=True)
    model = load_model(os.path.join(MCM_BASE_DIR, 'cnn_123-70.h5'))
    # model = 'dft'
    swarm_search(SWARM_COUNT, N_SWARM_ITER, GRID_SIZE, model)


