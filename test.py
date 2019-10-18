import os
import glob
import random
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, product
import imageio
import tqdm
import pdb

from constants import *

from simul_dft import *

# base_dir = 'cpp/evol_iter_grids/empty_steepest/'
base_dir = 'swarm_grids/swarm_1pore'
# base_dir = 'generative_model_3/sample_grids_better/'
STOP_ITER = 12000

def press(event):
    if event.key != 'q':
        exit(0)

def show_grids():
    PARTITIONS = 6
    P_SIZE = 1.0/(PARTITIONS-1)

    check_perms = [x for x in product(list(range(PARTITIONS)), repeat=PARTITIONS) if all(i <= j for i, j in zip(x, x[1:]))]

    print("madeperms")

    for curve in check_perms:
        target_density = np.zeros((N_ADSORP+1,1))

        for index, _ in enumerate(target_density):
            partition_num = 0
            while index > ((partition_num+1) * (N_ADSORP+1)/(PARTITIONS-1)):
                partition_num += 1
            target_density[index] = (curve[partition_num+1]-curve[partition_num]) * ((index*STEP_SIZE)-(partition_num)/(PARTITIONS-1)) + curve[partition_num]*P_SIZE

        target_density[0] = 0
        target_density[N_ADSORP] = 1

        target_density = target_density * 100

        fig = plt.figure(1, figsize=(6, 8))
        fig.canvas.mpl_connect('key_press_event', press)

        ax = plt.subplot(212)
        plt.xlabel('Relative humidity (%)')
        plt.ylabel('Water content (%)')
        ax.plot(np.arange(N_ADSORP+1)*STEP_SIZE*100, target_density)
        plt.plot([1],[1])
        ax.set_aspect(1)

    print(len(check_perms))
    plt.show()


show_grids()