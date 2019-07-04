import numpy as np
import math
import pandas as pd
import sys
import os
from multiprocessing import Pool
import tqdm

from constants import *

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
    density = run_dft(grid)
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
    #print('NL')

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
        density[jj] = sum(r[:,4])/(max(Ntotal_pores, 1))    # normalized by the number of empty pores
    return density

if __name__ == '__main__':
    # for n in range(1000):
    #     run_dft_iter(n)
    # sys.exit(0)
    run_dft_iters(MC_SRC_GRID_DIR, MC_SRC_RESULT_DIR)



