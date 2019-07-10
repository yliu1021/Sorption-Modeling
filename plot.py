import numpy as np
from scipy.ndimage import measurements
import math
import os
import sys
import matplotlib.pyplot as plt

from constants import *
import stats

def plot_few_verbose(n_grids, grid_dir):
    im_names = []
    for i in range(n_grids):
        im_names.append('raw_%04d'%i)
        im_names.append('shift_%04d'%i)
        im_names.append('rot_%04d'%i)
        im_names.append('grid_%04d'%i)
    for name in im_names:
        # Load the image
        im_path = os.path.join(grid_dir, name + '.csv')
        #print(im_path)
        grid = np.genfromtxt(im_path, delimiter=',')
        plot_grid(grid, os.path.join(grid_dir, name + '.png'))

def plot_grid(grid, path):
    # Create the plot
    fig, ax = plt.subplots()
    ax.pcolor(grid, cmap='Greys')
    ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))
    # Save the plot
    fig.savefig(path)
    plt.close(fig)

def plot_grids(n_grids, grid_dir, plot_dir, n_digits=4):
    os.makedirs(plot_dir, exist_ok=True)
    for i in range(n_grids):
        path_str = '/grid_%0' + str(n_digits) + 'd.csv'
        grid = np.genfromtxt(grid_dir + path_str%i, delimiter=',')
        path_str = '/testgrid_%0' + str(n_digits) + 'd.png'
        plot_grid(grid, plot_dir + path_str%i)

def plot_grids_from_sets(n_sets, grid_dir, plot_dir):
    for i in range(1, n_sets+1):
        os.makedirs(grid_dir + '%02d'%i, exist_ok=True)
        gridset = np.genfromtxt(grid_dir + '/train_grids%02d.csv'%i, delimiter=',')
        print(gridset.shape)
        for j in range(gridset.shape[0]):
            grid = np.reshape(gridset[j,:], (GRID_SIZE, GRID_SIZE))
            plot_grid(grid, grid_dir + '%02d'%i + '/testgrid_%04d.png'%j)

def plot_curve(den, path):
    fig, ax = plt.subplots()
    ax.plot(den[0:N_ADSORP,0] * STEP_SIZE*100, den[0:N_ADSORP,1])
    ax.plot((2*N_ADSORP - den[N_ADSORP:N_ITER,0]) * STEP_SIZE*100, den[N_ADSORP:N_ITER,1])
    ax.set(xlabel='Relative Humidity (%)', ylabel='Density',
         title='Sorption Curve')
    plt.legend(('Adsorption', 'Desorption',), loc='lower right')
    fig.savefig(path)
    plt.close(fig)

def plot_curves(n_grids, result_dir, curve_dir):
    os.makedirs(curve_dir, exist_ok=True)
    for i in range(n_grids):
        den = np.loadtxt(result_dir + '/density_%04d.csv'%i, dtype='float', delimiter=',', skiprows=1)
        plot_curve(den, curve_dir + '/sorp_%04d.png'%i)

def plot_curves_from_sets(n_sets, result_dir, plot_dir):
    for i in range(1, n_sets+1):
        os.makedirs(plot_dir + '%02d'%i, exist_ok=True)
        resultset = np.genfromtxt(result_dir + '/train_density%02d.csv'%i, delimiter=',')
        for j in range(resultset.shape[0]):
            # print(resultset.shape)
            den = np.zeros((N_ITER+1,2))
            den[:,0] = np.reshape(np.arange(N_ITER+1), (N_ITER+1,))
            den[:,1] = np.reshape(resultset[j,:], (N_ITER+1,))
            plot_curve(den, plot_dir + '%02d'%i + '/sorp_%04d.png'%j)

def plot_scatter(hys_dir, hys_name, feature):
    hys = np.loadtxt(hys_dir + '/' + hys_name + '.csv', dtype='float', delimiter=',')
    fig, ax = plt.subplots()
    ax.scatter(hys[:,0], hys[:,1])
    ax.set(xlabel=feature[3], ylabel=feature[4], title=feature[4] + ' v. ' + feature[3])
    fig.savefig(hys_dir + '/' + hys_name + '.png')
    plt.close(fig)

def plot_mult_scatter(hys_dir, hys_names, feature):
    hys = np.loadtxt(hys_dir + '/' + hys_names[0] + '.csv', dtype='float', delimiter=',')
    for i in range(1, len(hys_names)):
        nhys = np.loadtxt(hys_dir + '/' + hys_names[i] + '.csv', dtype='float', delimiter=',')
        hys = np.append(hys, nhys, axis=0)
    fig, ax = plt.subplots()
    ax.scatter(hys[:,0], hys[:,1])
    ax.set(xlabel=feature[3], ylabel=feature[4], title=feature[4] + ' v. ' + feature[3])
    fig.savefig(hys_dir + '/aggregate.png')
    plt.close(fig)

# TDOO: plot a comparison of the ones that are generated as "optimal"
def plot_compare(data_dir, train_dir, result_dir):
    for i in range(1, N_STEPS+1):
        pred = np.loadtxt(result_dir + '/optset%02d.csv'%i, dtype='float', delimiter=',')
        real = np.loadtxt(train_dir + '/train_results%02d.csv'%i, dtype='float', delimiter=',')
        fig, ax = plt.subplots()
        ax.scatter(real[:,1], pred[:,1])
        # plt.xlim((1, 3.5))
        # plt.ylim((1, 3.5))
        ax.set(xlabel='DFT results', ylabel='CNN results', title='Predicted difference comparison')
        fig.savefig(data_dir + '/comp%02d.png'%i)
        plt.close(fig)

def plot_summary():
    for dataset in DATASETS:
        grid_dir = GRID_DIR + dataset[0]
        plot_dir = PLOT_DIR + dataset[0]
        result_dir = RESULT_DIR + dataset[0]
        curve_dir = CURVE_DIR + dataset[0]
        hys_dir = HYS_DIR
        n_grids = dataset[1]
        plot_grids(n_grids, grid_dir, plot_dir)
        plot_curves(n_grids, result_dir, curve_dir)
        for feature in stats.FEATURES:
            hys_name = 'hys' + dataset[0] + feature[0]
            plot_scatter(hys_dir, hys_name, feature)

def plot_diagnosis():
    plot_grids_from_sets(N_STEPS, TRAIN_SET_DIR, RESULT_SET_DIR)
    plot_curves_from_sets(N_STEPS, TRAIN_SET_DIR, RESULT_SET_DIR)
    for i in range(1, N_STEPS+1):
        plot_scatter(TRAIN_SET_DIR, 'data_m%02d'%i, stats.FEATURES[1])
        plot_scatter(RESULT_SET_DIR, 'data_m%02d'%i, stats.FEATURES[1])
    os.makedirs(DATA_SET_DIR, exist_ok=True)
    plot_compare(DATA_SET_DIR, TRAIN_SET_DIR, RESULT_SET_DIR)

if __name__ == '__main__':
    plot_summary()
    # plot_diagnosis()
    # plot_mult_scatter(HYS_DIR, ['hys1m', 'hys2m'], stats.FEATURES[1])
    # plan = 'c1'
    # os.makedirs('predict_mc/resgrids_' + plan, exist_ok=True)
    # plot_grids(1000, 'predict_mc/iteration_' + plan, 'predict_mc/resgrids_' + plan, 6)


