import numpy as np
import matplotlib.pyplot as plt

n = 40
x = np.linspace(0, 1, n+1)

def gen_diffs(mean, var, _n=n, up_to=1):
    diffs = np.clip(np.exp(np.random.normal(mean, var, _n)), -10, 10)
    diffs = np.exp(np.random.normal(mean, var, _n))
    return diffs / np.sum(diffs) * up_to

def gen_func():
    anchor = np.random.uniform(0, 1)
    x = np.random.uniform(0.05, 0.95)
    ind = int(n*x)
    f_1 = np.insert(np.cumsum(gen_diffs(0, 4, ind, anchor)), 0, 0)
    f_2 = np.insert(np.cumsum(gen_diffs(0, 4, n - ind - 2, 1-anchor)), 0, 0) + anchor
    f = np.concatenate((f_1, np.array([anchor]), f_2))
    f[-1] = 1.0
    return f

ys = list()
xs = list()
for i in range(100):
    f = gen_func()
    print(f[0], f[-1])
    plt.plot(x, f)
    plt.show()
