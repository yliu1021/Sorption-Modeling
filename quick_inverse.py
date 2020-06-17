import sys
from random import shuffle, choice

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import glob

from tf_dft import run_dft
from tf_dft_inverse import make_generator_input
from constants import *


def gen_curve_weights(target_vec, sample_vecs, slack):
    """
    Returns a set of weights such that sample_vecs sum up to curve_diffs
    """
    n = len(sample_vecs)
    A = np.array(sample_vecs).T

    def cost(weights):
        return np.linalg.norm(A @ weights - target_vec)

    def grad(weights):
        return 2*A.T @ (A @ weights - target_vec)

    def hess(weights):
        return 2*A.T @ A

    def contour(weights):
        # we want np.ones_like(weights).T @ weights = 1
        # gradient of that is np.ones_like(weights)
        return np.ones_like(weights)

    def perp(vec, wrt):
        # returns the perpendicular component of vec with respect to wrt
        par = wrt * (np.dot(vec, wrt) / np.dot(wrt, wrt))
        perp = vec - par
        assert np.dot(perp, wrt) < 1e-4, f'{np.dot(perp, wrt)}'
        return perp

    weights = np.random.uniform(0.5, 1, n)
    weights /= np.sum(weights)

    # method = 'SLSQP'
    res = scipy.optimize.minimize(cost, weights, method='trust-constr', jac=grad, hess=hess,
                                  bounds=scipy.optimize.Bounds(np.zeros_like(weights), np.ones_like(weights), keep_feasible=True),
                                  constraints=scipy.optimize.LinearConstraint(np.ones_like(weights), 1, 1, keep_feasible=True),
                                  options={
                                      'maxiter': 100,
                                      'disp': False
                                  })
    weights = res.x
    
    assert np.abs(np.sum(weights) - 1) < 1e-4, f"Weights don't sum to 1 {weights}"
    assert np.min(weights) >= 0, f"Weights below 0 {weights}"
    assert np.max(weights) <= 1, f"Weights above 1 {weights}"
    
    return weights
    

def round(weights, lat_size):
    scaled_weights = weights * lat_size
    count_weights = np.around(scaled_weights)
    
    while True:
        normalized_weights = count_weights / lat_size
        
        round_errs = weights - normalized_weights
        round_err_args = np.argsort(round_errs)

        tol = 1e-5
        if np.sum(normalized_weights) - 1 > tol:
            # we need to decrease normalized weights
            for round_ind in round_err_args:
                if count_weights[round_ind] > 0:
                    count_weights[round_ind] -= 1
                    break
            else:
                raise ValueError('Unable to find closest ind to round')
        elif np.sum(normalized_weights) - 1 < -tol:
            # we need to increase normalized weights
            for round_ind in reversed(round_err_args):
                if count_weights[round_ind] < lat_size:
                    count_weights[round_ind] += 1
                    break
            else:
                raise ValueError('Unable to find closest ind to round')
        else:
            break

    assert np.abs(np.sum(normalized_weights) - 1) <= tol, f"normalized weights doesn't add to 1"
    assert np.min(normalized_weights) >= 0, f"normalized weights below 0"
    assert np.max(normalized_weights) <= 1, f"normalized weights above 1"
    assert np.abs(np.sum(count_weights) - lat_size) <= tol, "count weights doesn't add to lat size"
    
    return count_weights, normalized_weights


def fill(grid, counts, sample_grids, lat_size=25):
    """
    Fills grid with counts[i] copies of sample_grids[i] for each i
    """
    assert len(counts) == len(sample_grids), "counts and sample_grids size don't match"

    lat_size_width = int(np.sqrt(lat_size))
    assert lat_size_width * lat_size_width == lat_size, "lat_size must be a perfect square (as of now)"

    width, height = grid.shape
    assert width == height, "only square grids suppored (as of now)"

    assert width > lat_size_width and width % lat_size_width == 0, "grid size must be a positive multiple of lat size width"
    tile_size = width // lat_size_width

    assert sum(counts) == lat_size, "counts don't add up to lat size"
    
    grid[:, :] = 0 # set everything as a solid first

    count_ind = 0
    for i in range(1, width, tile_size):
        for j in range(1, height, tile_size):
            while count_ind < len(counts) and counts[count_ind] == 0:
                count_ind += 1
            if count_ind == len(counts):
                break
            grid[i:i+tile_size-1, j:j+tile_size-1] = sample_grids[count_ind][:tile_size-1, :tile_size-1]
            counts[count_ind] -= 1


def main(inner_loops):
    base_dir = './data_generation'
    density_files = glob.glob(os.path.join(base_dir, 'results', 'density_*.csv'))
    density_files.sort(reverse=False)
    density_files = density_files[:]
    real_densities_diffs = [np.diff(np.genfromtxt(density_file, delimiter=',')) for density_file in density_files]
    shuffle(real_densities_diffs)

    tile_size = GRID_SIZE // 20
    lat_size = tile_size * tile_size
    
    sample_grids = list()
    for i in range(18):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        grid[:i+1, :i+1] = 1
        sample_grids.append(grid)
    for i in range(1, 18):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        grid[:18, :18] = 1
        grid[1:i+1, 1:i+1] = 0
        sample_grids.append(grid)
    for i in range(1, 18):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        grid[:i, :18] = 1
        grid[:18, :i] = 1
        sample_grids.append(grid)
    sample_grids = np.array(sample_grids)

    sample_grids = np.roll(sample_grids, 2, axis=1)
    sample_grids = np.roll(sample_grids, 2, axis=2)
    sample_curves_diffs, _ = run_dft(sample_grids, inner_loops=inner_loops)
    sample_curves_diffs = sample_curves_diffs.numpy()
    sample_curves = np.cumsum(sample_curves_diffs, axis=1)

    for grid, curve_diffs in zip(sample_grids, sample_curves_diffs):
        fig = plt.figure(figsize=(10,4))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        ax = plt.subplot(1, 2, 1)
        ax.clear()
        ax.set_title('Grid (Black = Solid, Whited = Pore)')
        ax.set_yticks(np.linspace(0, grid.shape[0], 5))
        ax.set_xticks(np.linspace(0, grid.shape[1], 5))
        ax.pcolor(1 - grid, cmap='Greys', vmin=0.0, vmax=1.0)
        ax.set_aspect('equal')
        
        actual_curve = np.insert(np.cumsum(curve_diffs), 0, 0)
        ax = plt.subplot(1, 2, 2)
        ax.clear()
        ax.set_title('Adsorption Curve')
        ax.set_xticks(np.linspace(0, 1, 10))
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_ylim(0, 1)
        ax.plot(np.linspace(0, 1, N_ADSORP+1), actual_curve, color='red')
        ax.scatter(np.linspace(0, 1, N_ADSORP), curve_diffs)
        
        plt.show()
    
    weighting_errs = list()
    norm_weighting_errs = list()
    generator_errs = list()
    for i, target_curve_diffs in enumerate(real_densities_diffs):
        target_curve = np.insert(np.cumsum(target_curve_diffs), 0, 0)
        
        weights = gen_curve_weights(target_curve[1:], sample_curves, 80)

        weighted_target_curve = weights.T @ sample_curves
        weighted_target_curve = np.insert(weighted_target_curve, 0, 0)

        err = np.abs(np.sum(target_curve - weighted_target_curve)) / len(target_curve)
        weighting_errs.append(err)

        count_weights, norm_weights = round(weights, lat_size=lat_size)
        count_weights = count_weights.astype(np.int32)

        norm_weighted_target_curve = norm_weights.T @ sample_curves
        norm_weighted_target_curve = np.insert(norm_weighted_target_curve, 0, 0)

        err = np.abs(np.sum(target_curve - norm_weighted_target_curve)) / len(target_curve)
        norm_weighting_errs.append(err)
        
        generated_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        fill(generated_grid, count_weights, sample_grids, lat_size=lat_size)

        generated_curve_diffs, _ = run_dft(np.array([generated_grid]), inner_loops=inner_loops)
        generated_curve = np.insert(np.cumsum(generated_curve_diffs), 0, 0)

        err = np.abs(np.sum(target_curve - generated_curve)) / len(target_curve)
        generator_errs.append(err)
        
        e = np.array(generator_errs)
        print(f'\r {i}/{len(real_densities_diffs)} | Error: {e.mean():.5f} | Max: {e.max():.5f} | Min: {e.min():.5f} | Std: {e.std():.5f}', end='')

        continue
    
        fig = plt.figure(figsize=(10,4))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        ax = plt.subplot(1, 2, 1)
        ax.clear()
        ax.set_title('Grid (Black = Solid, Whited = Pore)')
        ax.set_yticks(np.linspace(0, generated_grid.shape[0], 5))
        ax.set_xticks(np.linspace(0, generated_grid.shape[1], 5))
        ax.pcolor(1 - generated_grid, cmap='Greys', vmin=0.0, vmax=1.0)
        ax.set_aspect('equal')

        ax = plt.subplot(1, 2, 2)
        ax.clear()
        ax.set_title('Adsorption Curve')
        ax.set_xticks(np.linspace(0, 1, 10))
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_ylim(0, 1)
        ax.plot(np.linspace(0, 1, N_ADSORP+1), target_curve, label='target curve')
        ax.plot(np.linspace(0, 1, N_ADSORP+1), weighted_target_curve, label='weighted curve')
        ax.plot(np.linspace(0, 1, N_ADSORP+1), norm_weighted_target_curve, label='norm weighted curve')
        ax.plot(np.linspace(0, 1, N_ADSORP+1), generated_curve, label='generated curve')
        plt.legend()
        plt.show()
        
    print()
    errs = np.array(errs)
    print(errs.mean())
    print(errs.std())
    plt.hist(errs, bins=20)
    plt.show()
        
    exit(0)
    


if __name__ == '__main__':
    main(inner_loops=int(sys.argv[1]))
