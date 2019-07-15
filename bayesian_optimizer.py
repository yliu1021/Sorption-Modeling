import os

import numpy as np
# import keras
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

'''
We try to optimize the model by minimizing the difference between the intended
metric and the actual metric that was generated among all the metrics

We weigh the loss of lower metrics higher than the loss of higher metrics

Minimize:
loss = Sum_{i} (target_metric_{i} - actual_metric{i})^2 / (actual_metric{i} + 0.1) ^ 0.5
'''


