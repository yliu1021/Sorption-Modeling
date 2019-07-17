import numpy as np
import matplotlib.pyplot as plt

n = 40
def gen_diffs(mean, var):
    mean = np.random.uniform(-np.log(2+1/40), np.log(2+1/40))
    diffs = np.exp(np.random.normal(mean, var, n))
    return diffs / np.sum(diffs)

def gen_func(mean, var):
    f = np.insert(np.cumsum(gen_diffs(mean, var)), 0, 0)
    return f

for _ in range(10):
    nf = 20
    for i in range(nf):
        y = gen_func(0, i/nf*6)
        x = np.linspace(0, 1, n+1)
        plt.plot(x, y)
        plt.show()
