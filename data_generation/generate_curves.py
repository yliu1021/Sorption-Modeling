import sys
sys.path.append('..')

import math
import matplotlib.pyplot as plt
import numpy as np
import os

from constants import *

curves = []
rescaled = []

def generate_curves(div=2):
    def backtrack(l, x, y):
        if math.isclose(x, 1, abs_tol=0.0001) and math.isclose(y, 1, abs_tol=0.0001):
            l.append(y)
            curves.append(l)
            return
        if x < 1:
            backtrack(l+[y], x+(1/div), y)
        if y < 1:
            backtrack(l, x, y+(1/div))
    backtrack([0], (1/div), 0)


def rescale_curves():
    for c in curves:
        l = [0]
        for i in range(1, N_ADSORP):
            intervals = 1/(len(c)-1)
            x = i*(1/N_ADSORP)
            ind = int(x/intervals)
            y = c[ind] + ((c[ind+1]-c[ind])/intervals)*(x-(ind*intervals))
            l.append(y)
        l.append(1)
        rescaled.append(l)


def save_curves():
    os.makedirs('results/', exist_ok=True)
    for i, c in enumerate(rescaled):
        np.savetxt('results/density_%04d.csv'%i, c, newline=',')


if __name__ == '__main__':
    for i in range(2, 4):
        generate_curves(div=i)

    rescale_curves()
    rescaled = np.round(np.array(rescaled), decimals=10)
    rescaled = np.unique(rescaled, axis=0)
    print(rescaled.shape)
    save_curves()

    # plt.ion()
    # plt.show()
    # for c in rescaled:
    #     plt.plot(np.linspace(0, 1, len(c)), c)
    #     plt.draw()
    #     plt.pause(0.001)
    # plt.ioff()
    # plt.show()
