import numpy as np
import math
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)
import os
from multiprocessing import Pool
import tqdm

from constants import *

import time

# folder = os.makedirs(RESULT_DIR, exist_ok=True)

# Define the effective potential function
def veff(r, muu, NL):
    vieff = np.zeros((N_SQUARES + 1))
    for i in range(1, N_SQUARES + 1):
        for jj in range(1,5):
            aa = NL[i,jj]
            vieff[i] = vieff[i]+WFF*(r[aa,4]+Y*(1-r[aa,3]))
        vieff[i] = vieff[i]+muu
    return vieff

# Define the density function
def rou(veff0, r):
    x = np.zeros((N_SQUARES + 1))
    for i in range(1, N_SQUARES + 1):
        x[i] = r[i,3]/(1+math.exp(-BETA*veff0[i]))
    return x

# Run the DFT simulation for grid n, reading and writing to file
def run_dft_iter(n, grid_dir=GRID_DIR, result_dir=RESULT_DIR):
    grid = np.loadtxt(grid_dir + '/grid_%04d.csv'%n, dtype='int', delimiter=',')
    grid = grid.reshape(N_SQUARES)
    density = run_dft_fast(grid)
    density = pd.DataFrame(density)
    density.to_csv(result_dir + '/density_%04d.csv'%n)

# The multiprocessing module can only pickle objects at the global level
pool_arg = {}
def run_dft_pool(i):
    run_dft_iter(i, pool_arg['grid_dir'], pool_arg['result_dir'])
            
def run_dft_iters(grid_base_dir=GRID_DIR, result_base_dir=RESULT_DIR):
    for dataset in DATASETS:
        grid_dir = grid_base_dir + dataset[0]
        result_dir = result_base_dir + dataset[0]
        os.makedirs(result_dir, exist_ok=True)
        n_grids = dataset[1]
        
        # we set the appropriate objects for multiprocessing to pickle
        pool_arg['grid_dir'] = grid_dir
        pool_arg['result_dir'] = result_dir
        p = Pool()
        # then use tqdm to display a progress bar for running the dft simulation
        # on all the grids
        list(tqdm.tqdm(p.imap(run_dft_pool, range(n_grids)), total=n_grids))

# Run the DFT simulation for a single flattened grid
def run_dft(grid):
    # Intialize the pore
    r = np.zeros((N_SQUARES + 1, 5))
    Lx = 0
    for i in range(1, N_SQUARES + 1):
        r[i,1] = Lx
        Lx += 1
        if i % GRID_SIZE == 0:
            Lx = 0
    Ly = 0
    for i in range(1, N_SQUARES + 1):
        r[i,2] = Ly
        if i % GRID_SIZE == 0:
            Ly = Ly+1
    
    #print(grid.shape)
    r[1:N_SQUARES + 1, 3] = grid
    Ntotal_pores = sum(r[:,3])
    #print(Ntotal_pores)       

    # Record the neighbor list
    rc = 1.01
    rc_square = rc**2
    NN = np.zeros((N_SQUARES + 1, 1), dtype=np.int)
    NL = np.zeros((N_SQUARES + 1, N_SQUARES), dtype=np.int)
    r12 = np.zeros((2,3))
    for i in range(1,N_SQUARES + 1-1):
        for jj in range(i+1,N_SQUARES + 1):
            r12[1,1] = r[jj,1]-r[i,1]
            r12[1,2] = r[jj,2]-r[i,2]
            r12[1,1] = r12[1,1]-round(r12[1,1]/GRID_SIZE)*GRID_SIZE
            r12[1,2] = r12[1,2]-round(r12[1,2]/GRID_SIZE)*GRID_SIZE
            d12_square = r12[1,1]*r12[1,1]+r12[1,2]*r12[1,2]
            if d12_square <= rc_square:
                NN[i] = NN[i]+1
                NN1 = int(NN[i])
                NL[i,NN1] = jj
                NN[jj] = NN[jj]+1
                NN2 = int(NN[jj])
                NL[jj,NN2]= i
    NN_max = NN[:,0].max()           
    NL = NL[:,0:NN_max+1]

    # Let the pores be filled fully    
    for i in range(1, N_SQUARES + 1):   
        if r[i,3] == 1:
            r[i,4] = 1
    density = np.zeros((N_ITER + 1, 1))
    #print('start')
    
    # Calculate the density through iteration
    for jj in range(0, N_ITER + 1):
        #print(jj)
        if jj <= N_ADSORP:
            RH = jj * STEP_SIZE                     # increase relative humidity
        else:
            RH = N_ADSORP*STEP_SIZE - (jj-N_ADSORP)*STEP_SIZE  # decrease relative humidity 
        if RH == 0:
            muu = -90                               # a large negative potential as suggested in the paper
        else:                                      
            muu = MUSAT+KB*T*math.log(RH)
        for i in range(1,100000000):
            vi = veff(r,muu,NL)
            rounew = rou(vi,r)
            drou = rounew - r[:,4]
            power_drou = np.sum(drou**2)/(N_SQUARES)           # convergence criteria
            if power_drou < 1e-10:
                r[:,4] = rounew[:]
                break
            else:
                r[:,4] = rounew                     # convergence criteria
            if i == 100000000-1:
                print('error')                      # cannot converge
        print(i)
        density[jj] = sum(r[:,4])/(Ntotal_pores)    # normalized by the number of empty pores
    return density

def run_dft_fast(grid):
    # Intialize the pore
    r = np.zeros((N_SQUARES + 1, 4), dtype='float64')
    range_20 = np.arange(20, dtype='float64')
    r[:, 0] = np.insert(np.tile(range_20, GRID_SIZE), 0, 0)
    r[:, 1] = np.insert(np.repeat(range_20, GRID_SIZE), 0, 0)

    r[1:N_SQUARES + 1, 2] = grid
    Ntotal_pores = np.sum(r[:,2])
    if Ntotal_pores == 0:
        return np.zeros((N_ITER + 1, 1))

    # Record the neighbor list
    rc = 1.01
    rc_square = rc**2
    NN = np.zeros((N_SQUARES + 1), dtype=np.int)
    NL = np.zeros((N_SQUARES + 1, N_SQUARES), dtype=np.int)
    for i in range(1,N_SQUARES + 1-1):
        r1 = r[:, 0] - r[i, 0]
        r2 = r[:, 1] - r[i, 1]
        r1 = r1 - np.round(r1 / GRID_SIZE) * GRID_SIZE
        r2 = r2 - np.round(r2 / GRID_SIZE) * GRID_SIZE
        d12_squares = r1 * r1 + r2 * r2
        small_enough = d12_squares <= 1.0201 # rc_square
        for jj in range(i+1, N_SQUARES + 1):
            if small_enough[jj]:
                NN[i] += 1
                NN[jj] += 1
                NL[i,NN[i]] = jj
                NL[jj,NN[jj]]= i        
    NL = NL[:,0:NN.max()+1]

    # Let the pores be filled fully
    r[1:N_SQUARES+1, 3] = r[1:N_SQUARES+1, 2]
    
    # Calculate the density through iteration
    density = np.zeros((N_ITER + 1, 1))
    for jj in range(0, N_ITER + 1):
        #print(jj)
        if jj <= N_ADSORP:
            RH = jj * STEP_SIZE                     # increase relative humidity
        else:
            RH = N_ADSORP*STEP_SIZE - (jj-N_ADSORP)*STEP_SIZE  # decrease relative humidity 
        if RH == 0:
            muu = -90                               # a large negative potential as suggested in the paper
        else:                                      
            muu = MUSAT+KB*T*math.log(RH)
        for _ in range(1,100000000):
            r_acc = WFF*(r[:, 3] + Y*(1 - r[:, 2]))
            vi = np.zeros((N_SQUARES + 1))
            for i2 in range(1, N_SQUARES + 1):
                vi[i2] += r_acc[NL[i2, 1]] + r_acc[NL[i2, 2]] + r_acc[NL[i2, 3]] + r_acc[NL[i2, 4]]
            vi[1:N_SQUARES+1] += muu

            rounew = np.zeros((N_SQUARES + 1))
            rounew = r[:, 2] / (1 + np.exp(-BETA*vi))
            rounew[0] = 0
            drou = rounew - r[:,3]
            power_drou = np.sum(drou**2)/(N_SQUARES)           # convergence criteria
            if power_drou < 1e-10:
                r[:,3] = rounew
                break
            else:
                r[:,3] = rounew                     # convergence criteria
        else:
            print('error')                      # cannot converge
        density[jj] = sum(r[:,3])/(Ntotal_pores)    # normalized by the number of empty pores

    return density

if __name__ == '__main__':
    # for n in range(1000):
    #     run_dft_iter(n)
    # sys.exit(0)
    setup = """\
from __main__ import run_dft, run_dft_fast
import numpy as np
from constants import N_SQUARES
grid = np.genfromtxt('generative_model/step1/grids/grid_0031.csv', delimiter=',')
grid = grid.reshape(N_SQUARES)
"""

    from random import randint
    for i in range(5):
        r = randint(0, 299)
        grid = np.genfromtxt('generative_model_1/step1/grids/grid_%04d.csv'%r, delimiter=',')
        grid = grid.reshape(N_SQUARES)
        den1 = run_dft(grid)
        den2 = run_dft_fast(grid)
        largest_diff = np.max(np.abs(den1 - den2))
        if largest_diff < 10e-15:
            print("Got same densities (with diff {})".format(largest_diff))
        else:
            print("Got different densities")
            print("Grid: {}".format(r))
            print(np.concatenate((den1, den2), axis=1))
            print(largest_diff)
            print(np.sum(np.abs(den1 - den2)))
            exit(0)

    import timeit
    res1 = timeit.timeit('run_dft(grid)', number=5, setup=setup)
    print('Original version:\t{:.6f} seconds'.format(res1))
    res2 = timeit.timeit('run_dft_fast(grid)', number=5, setup=setup)
    print('Fast version:\t\t{:.6f} seconds'.format(res2))
    print('{:.1%} improvement'.format((res1-res2)/res1))
    exit(0)

    run_dft_iters(MC_SRC_GRID_DIR, MC_SRC_RESULT_DIR)



