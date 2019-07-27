import os
import glob
import json # for pretty printing dict

import generate

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
    'learning_rate_damper': {
        'type': float,
        'min' : 0.1,
        'max' : 0.5
    }
}

proxy_enforcer_hyperparameters = {
    'first_filter_size': {
        'type': int,
        'min' : 3,
        'max' : 9
    },
    'last_conv_depth': {
        'type': int,
        'min' : 128,
        'max' : 300
    },
    'dense_layer_size': {
        'type': int,
        'min' : 1024,
        'max' : 2500
    }
}

generator_hyperparameters = {
    'first_conv_depth': {
        'type': int,
        'min' : 64,
        'max' : 128
    },
    'pre_deconv1_depth': {
        'type': int,
        'min' : 80,
        'max' : 120
    },
    'post_deconv2_depth': {
        'type': int,
        'min' : 32,
        'max' : 96
    },
    'last_filter_size': {
        'type': int,
        'min' : 3,
        'max' : 9
    }
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


def evaluate_step(step):
    base_dir = get_base_dir(step)
    steps_dir = glob.glob(os.path.join(base_dir, 'step[0-9]*'))
    steps_dir.sort(key=lambda x: int(x.split('/')[-1][4:]))
    last_step_dir = steps_dir[-1]
    
    artificial_metrics_dir = os.path.join(last_step_dir, 'artificial_metrics')
    density_dir = os.path.join(last_step_dir, 'results')
    
    artificial_metrics_files = glob.glob(os.path.join(artificial_metrics_dir, 'artificial_metric_*.csv'))
    density_files = glob.glob(os.path.join(density_dir, 'density_*.csv'))
    artificial_metrics_files.sort()
    density_files.sort()
    
    densities = [np.genfromtxt(density_file, delimiter=',', skip_header=1,
                               max_rows=N_ADSORP) for density_file in density_files]
    artificial_metrics = [np.genfromtxt(metrics_file) for metrics_file in artificial_metrics_files]
    densities = np.insert(np.array(densities)[:, 1:, 1], N_ADSORP-1, 1, axis=1)
    artificial_metrics = np.cumsum(np.array(artificial_metrics), axis=1)
    densities = np.insert(densities, 0, 0, axis=1)
    artificial_metrics = np.insert(artificial_metrics, 0, 0, axis=1)

    diffs = np.max(np.abs(densities - artificial_metrics), axis=1)
    return diffs.mean()


def train_network(step, hyperparameters):
    generate.base_dir = get_base_dir(step)
    generate.start_training(start=1, end=11, **hyperparameters)
    return evaluate_step(step)


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


res = gp_minimize(minimize, parameter_bounds, n_calls=40,
                  verbose=True)
print(all_hyperparameter_keys)
print(res.x)
print(res.fun)
print(res.x_iters)
print(res.func_vals)
