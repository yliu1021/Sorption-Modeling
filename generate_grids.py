import argparse
import os

from keras.models import load_model
from keras import backend as K
import numpy as np

def round_through(x):
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    x = (0.5 * x) + 0.5 # Gradient is steeper than regular sigmoid activation
    return K.clip(x, 0, 1)


def binary_sigmoid(x):
    return round_through(_hard_sigmoid(x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="location of the generator.hdf5 file")
    parser.add_argument("--save_loc", help="the directory to place all the new grids")
    parser.add_argument("--num_grids", help="the number of grids to generate", type=int, default=100)
    parser.add_argument("--variance", help="the variance of the latent code parameters used to generate the grids",
                        type=float, default=0.25)
    args = parser.parse_args()
    
    input_density = input().split(' ')
    density = np.array([input_density] * args.num_grids, dtype='float')
    diffs = np.diff(density, append=1.0)

    model = load_model(args.model_path, custom_objects={'binary_sigmoid': binary_sigmoid})
    uniform_latent_code = np.clip(np.random.normal(loc=0.5, scale=args.variance,
                                                   size=(args.num_grids, 5)), 0, 1)

    grids = model.predict([diffs, uniform_latent_code])
    grids = grids.astype('int')
    
    for i, grid in enumerate(grids):
        path = os.path.join(args.save_loc, 'grid_%04d.csv'%i)
        np.savetxt(path, grid, fmt='%i', delimiter=',')