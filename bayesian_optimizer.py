import os
import glob
import json # for pretty printing dict
import shutil

import targeted_generate

import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize

from constants import *

'''
The target metric that we minimize is the worst abs difference between the
target curve we generate and the actual curve of the generated grid averaged
across all grids/curves in the last step of the training process.
'''

# Min max are inclusive
training_hyperparameters = {
    'explore_rate': {
        'type': float,
        'min' : 0.0,
        'max' : 1.0
    }
}

proxy_enforcer_hyperparameters = {
}

generator_hyperparameters = {
}

all_hyperparameters = dict()
all_hyperparameters.update(training_hyperparameters)
all_hyperparameters.update(proxy_enforcer_hyperparameters)
all_hyperparameters.update(generator_hyperparameters)
all_hyperparameter_keys = list(all_hyperparameters.keys())
print(all_hyperparameter_keys)
parameter_bounds = list()
for key in all_hyperparameter_keys:
    parameter_opts = all_hyperparameters[key]
    min_value = parameter_opts['min']
    max_value = parameter_opts['max']
    parameter_bounds.append((min_value, max_value))


def get_base_dir(step):
    return 'generative_model_optimization_{}'.format(step)


def train_network(step, hyperparameters):
    hyperparameters['base_dir'] = get_base_dir(step)
    hyperparameters['train_steps'] = 6
    accuracy = targeted_generate.start_training(**hyperparameters)
    return accuracy


def minimize(parameters):
    hyperparameters = dict(zip(all_hyperparameter_keys, parameters))
    for key, value in hyperparameters.items():
        parameter_opts = all_hyperparameters[key]
        t = parameter_opts['type']
        if t is int:
            hyperparameters[key] = int(round(value))
        elif t is float:
            hyperparameters[key] = float(value)
    step = len(glob.glob('generative_model_optimization_*'))
    print('Starting evaluating with: ')
    print(json.dumps(hyperparameters, sort_keys=True, indent=4))
    result = train_network(step, hyperparameters)
    step += 1
    print('Finished evaluating with: ')
    print(json.dumps(hyperparameters, sort_keys=True, indent=4))
    print('Got result: {}'.format(result))
    return result


res = gp_minimize(minimize, parameter_bounds, n_calls=7,
                  n_random_starts=3,
                  verbose=True)
print(all_hyperparameter_keys)
print(res.x)
print(res.fun)
print(res.x_iters)
print(res.func_vals)
