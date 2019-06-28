import numpy as np
import math
import os
import sys

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, Dropout
from sklearn import linear_model

from constants import *

# Loads data from all datasets specified in setlist, taking only a fraction
# of them from either the front or the back.  
def load_data(setlist, hys_dir, version, fraction=1, rev=False, grid_dir=GRID_DIR):
    partial = [int(dataset[1] * fraction) for dataset in setlist]
    for i in range(1, len(partial)):
        partial[i] += partial[i - 1]
    #print(partial)
    x_train = np.zeros((partial[-1], GRID_SIZE * GRID_SIZE))
    y_train = np.zeros((partial[-1]))
    data_i = 0
    for dataset in setlist:
        path = os.path.join(hys_dir, 'hys' + dataset[0] + version + '.csv')
        results = np.loadtxt(path, dtype='float', delimiter=',')
        red_size = int(dataset[1] * fraction)
        for i in range(0, red_size):
            # Awful index manipulation
            index = i + partial[data_i] - red_size
            n = i if not rev else dataset[1] - i - 1
            path = os.path.join(grid_dir + dataset[0], 'grid_%04d.csv'%n)
            grid = np.loadtxt(path, dtype='int', delimiter=',')
            x_train[index,:] = grid.reshape((1, GRID_SIZE*GRID_SIZE))
            y_train[index] = results[n,1]
        data_i += 1
    return x_train, y_train

# Single-layer neural network with n_hidden neurons in the hidden layer.
def single_layer_model(n_hidden=64):
    model = Sequential()
    model.add(Dense(units=n_hidden, activation='relu', name='fc1',
                    input_shape=(GRID_SIZE*GRID_SIZE,)))
    model.add(Dense(units=1))
    return model

# Simple convolutional neural network.
def simple_cnn(one_layer=False):
    model = Sequential()
    model.add(Reshape((GRID_SIZE, GRID_SIZE, 1), input_shape = (GRID_SIZE*GRID_SIZE,)))
    model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu', name='conv1'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    if not one_layer:
        model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', name='conv2'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', name='conv3'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', name='fc4'))
    model.add(Dense(units=1))
    return model

# Trains a model on the input data. 
def train(model, x_train, y_train, x_test):
    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, batch_size=32)
    return model

# Trains a linear regression model on the input data.
def lin_reg(x_train, y_train, x_test):
    model = linear_model.LinearRegression()
    model = model.fit(x_train, y_train)
    print('regression score: ' + str(model.score(x_train, y_train)))
    y_res = model.predict(x_test)
    return y_res

# Basic linear regression model. 
def lin_reg_model():
    return linear_model.LinearRegression()

# Formats results into an array to be saved to file. 
def record(y_res, x_test, y_test):
    n_grids = x_test.shape[0]
    results = np.zeros((n_grids, 3))
    results[:,0] = np.arange(n_grids)
    results[:,1] = y_test
    results[:,2] = np.reshape(y_res, (n_grids))
    return results

# For each model, trains on setlist1, then evaluates on setlist2.
def train_one_on_other(setlist1, setlist2, train_dir, hys_dir, version):
    (x_train, y_train) = load_data(setlist1, hys_dir, version)
    (x_test, y_test) = load_data(setlist2, hys_dir, version)
    name1 = ''.join([dataset[0] for dataset in setlist1])
    name2 = ''.join([dataset[0] for dataset in setlist2])
    train_and_eval(x_train, y_train, x_test, y_test, name1, name2, train_dir)

# For each model, trains on a fraction of a setlist and tests on the other.
def train_on_fraction(setlist, fraction, train_dir, hys_dir, version):
    (x_train, y_train) = load_data(setlist, hys_dir, version, fraction, False)
    (x_test, y_test) = load_data(setlist, hys_dir, version, 1-fraction, True)
    base_str = ''.join([dataset[0] for dataset in setlist]) + '-'
    name1 = base_str + str(int(fraction * 100))
    name2 = base_str + str(int((1-fraction) * 100))
    train_and_eval(x_train, y_train, x_test, y_test, name1, name2, train_dir)

# Trains and evaluates each model onto the given datasets
def train_and_eval(x_train, y_train, x_test, y_test, name1, name2, train_dir):
    # # Neural networks
    # for i in range(1, 7):
    #     results = record(train(single_layer_model(4**i), x_train, y_train, x_test), x_test, y_test)
    #     path = os.path.join(train_dir, 'res_nn%i_'%(4**i) + name1 + '_' + name2 + '.csv')
    #     np.savetxt(path, results, delimiter=',')
    # # Convolutional Neural Network
    model = train(simple_cnn(), x_train, y_train, x_test)
    results = record(model.predict(x_test, batch_size=128), x_test, y_test)
    path = os.path.join(train_dir, 'res_cnn_' + name1 + '_' + name2 + '.csv')
    np.savetxt(path, results, delimiter=',')
    results = record(model.predict(x_train, batch_size=128), x_train, y_train)
    path = os.path.join(train_dir, 'res_cnn_' + name1 + '_' + name1 + '.csv')
    np.savetxt(path, results, delimiter=',')
    model.save(os.path.join(MODEL_BASE_DIR, 'cnn_' + name1 + '.h5'))
    # # Linear regression
    # results = record(lin_reg(x_train, y_train, x_test), x_test, y_test)
    # path = os.path.join(train_dir, 'res_lin_' + name1 + '_' + name2 + '.csv')
    # np.savetxt(path, results, delimiter=',')

def generate_train_set(hys_dir=HYS_DIR, model_base_dir=MODEL_BASE_DIR, 
                       model_result_dir=MODEL_RESULT_DIR):
    os.makedirs(model_base_dir, exist_ok=True)
    for version in VERSIONS:
        train_dir = model_result_dir + '_' + version
        os.makedirs(train_dir, exist_ok=True)
        train_on_fraction(DATASETS, TRAIN_FRAC, train_dir, hys_dir, version)
        # train_one_on_other(DATASETS, DATASETS, train_dir, hys_dir, version)
        # train_one_on_other([DATASETS[0], DATASETS[2]], [DATASETS[1]], 
        #                    train_dir, hys_dir, version)
        # train_one_on_other([DATASETS[1], DATASETS[2]], [DATASETS[0]], 
        #                    train_dir, hys_dir, version)

if __name__ == '__main__':
    os.makedirs(MC_MODEL_DIR, exist_ok=True)
    generate_train_set(MC_HYS_DIR, MCM_BASE_DIR, MCM_RESULT_DIR)



