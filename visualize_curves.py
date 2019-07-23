import numpy as np
import matplotlib.pyplot as plt

n = 40
x = np.linspace(0, 1, n+1)

def gen_diffs(mean, var, _n=n, up_to=1):
    diffs = np.clip(np.exp(np.random.normal(mean, var, _n)), -10, 10)
    return diffs / np.sum(diffs) * up_to

def gen_func():
    f = np.insert(np.cumsum(gen_diffs(0, 2)), 0, 0)
    return f

def gen_func():
    anchor = np.random.uniform(0, 1)
    x = np.clip(np.random.normal(0.5, 0.4), 0.1, 0.9)
    ind = int(n*x)
    f_1 = np.insert(np.cumsum(gen_diffs(0, 3, ind, anchor)), 0, 0)
    f_2 = np.insert(np.cumsum(gen_diffs(0, 3, n - ind - 2, 1-anchor)), 0, 0) + anchor
    f = np.concatenate((f_1, np.array([anchor]), f_2))
    return f, x

ys = list()
xs = list()
for i in range(100):
    f, ind = gen_func()
    print(f[0], f[-1])
    plt.plot(x, f)
    plt.show()
