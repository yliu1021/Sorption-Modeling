import os
import glob
import json # for pretty printing dict
import shutil

import targeted_generate

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize

from constants import *
import models
import data

'''
The target metric that we minimize is the worst abs difference between the
target curve we generate and the actual curve of the generated grid averaged
across all grids/curves in the last step of the training process.
'''

# Min max are inclusive
training_hyperparameters = {
}

predictor_hyperparameters = {
    'first_filter_size': {
        'type': int,
        'min' : 3,
        'max' : 5
    },
    'last_conv_depth': {
        'type': int,
        'min' : 100,
        'max' : 400
    },
    'dense_layer_size': {
        'type': int,
        'min' : 500,
        'max' : 4000
    },
    'num_convs': {
        'type': int,
        'min' : 1,
        'max' : 5
    },
    'boundary_expand': {
        'type': int,
        'min' : 4,
        'max' : 20
    }
}

generator_hyperparameters = {
}

all_hyperparameters = dict()
all_hyperparameters.update(training_hyperparameters)
all_hyperparameters.update(predictor_hyperparameters)
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
    # hyperparameters['base_dir'] = get_base_dir(step)
    # hyperparameters['train_steps'] = 6
    # accuracy = targeted_generate.start_training(**hyperparameters)
    K.clear_session()

    predictor_model, _ = models.make_predictor_model(**hyperparameters)
    train_grids, train_curves = data.get_all_data(matching='none')
    # Define our loss function and compile our model
    loss_func = hyperparameters.get('loss_func', 'kullback_leibler_divergence')
    models.unfreeze(predictor_model)
    learning_rate = 10**-3
    optimizer = Adam(learning_rate, clipnorm=1.0)
    predictor_model.compile(optimizer, loss=loss_func, metrics=['mae', models.worst_abs_loss])
    # Fit our model to the dataset
    predictor_batch_size = hyperparameters.get('predictor_batch_size', 64)
    predictor_epochs = 15
    h = predictor_model.fit(x=train_grids, y=train_curves, batch_size=predictor_batch_size,
                            epochs=predictor_epochs, validation_split=0.1)
    mae = h.history['val_mae'][-1] + h.history['val_mae'][-2] + h.history['val_mae'][-3]
    mae /= 3
    return mae


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


res = gp_minimize(minimize, parameter_bounds, n_calls=50,
                  n_random_starts=10,
                  verbose=True)
print(all_hyperparameter_keys)
print(res.x)
print(res.fun)
print(res.x_iters)
print(res.func_vals)
