import numpy as np
import matplotlib.pyplot as plt

n = 40
x = np.linspace(0, 1, n+1)

def gen_diffs(mean, var):
    diffs = np.clip(np.exp(np.random.normal(0, var, n)), -10, 10)
    return diffs / np.sum(diffs)

def gen_func(mean, var):
    f = np.insert(np.cumsum(gen_diffs(mean, var)), 0, 0)
    return f

loss = 0.1
for _ in range(20):
    correct_diffs = gen_diffs(0, 2)
    correct_f = np.insert(np.cumsum(correct_diffs), 0, 0)
    plt.plot(x, correct_f)
    plt.show()

exit(0)

nf = 20
for i in range(nf):
    y = gen_func(0, i/nf*6)
    plt.plot(x, y)
    plt.show()
