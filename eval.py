import numpy as np
import math
import os
import sys
import matplotlib.pyplot as plt

from constants import *

names = ['123-70_123-30', '123-70_123-70'] #'123_123', '13_2', '23_1']

def compile(hys_dir, hys_name, train_dir, train_name, plot=True):
    if hys_name is not None:
        path = os.path.join(hys_dir, hys_name + '.csv')
        hys = np.loadtxt(path, dtype='float', delimiter=',')
    path = os.path.join(train_dir, train_name + '.csv')
    diff = np.loadtxt(path, dtype='float', delimiter=',')
    rmse = math.sqrt(np.sum(np.square(diff[:,1] - diff[:,2]))/diff.shape[0])
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(diff[:,1], diff[:,2])
        ax.set(xlabel='Difference', ylabel='Predicted', title='RMSE: ' + str(rmse))
        fig.savefig(train_dir + '/' + train_name + '.png')
        print(rmse)
        plt.close(fig)
    return rmse

def compile_all():
    for version in VERSIONS:
        train_dir = MODEL_RESULT_DIR + '_' + version
        for dataset in DATASETS:
            hys_name = 'hys' + dataset[0] + version
            for model in MODELS:
                for name in names:
                    train_name = 'res_' + model + '_' + name
                    compile(HYS_DIR, hys_name, train_dir, train_name)

def plot_nn_rmse(start_size, end_size, plot=True):
    results = np.zeros((end_size - start_size + 1, 2))
    for version in VERSIONS:
        train_dir = MODEL_RESULT_DIR + '_' + version
        for i in range(start_size, end_size + 1):
            for name in names:
                train_name = 'res_nn' + str(4**i) + '_' + name
                results[i-start_size,0] = i
                results[i-start_size,1] = compile(None, None, train_dir, train_name, False)
        if plot: 
            fig, ax = plt.subplots()
            ax.scatter(results[:,0], results[:,1])
            ax.set(xlabel='# Neurons (log scale)', ylabel='RMSE', 
                   title='Neural Network Performance')
            fig.savefig(train_dir + '/nn_comp.png')
            plt.close(fig)
        np.savetxt(train_dir + '/nn_comp.csv', results)


if __name__ == '__main__':
    compile_all()
    # plot_nn_rmse(1, 6)



