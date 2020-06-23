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
    for i in range(0, width, tile_size):
        for j in range(0, height, tile_size):
            while count_ind < len(counts) and counts[count_ind] == 0:
                count_ind += 1
            if count_ind == len(counts):
                break
            grid[i:i+tile_size, j:j+tile_size] = sample_grids[count_ind][:tile_size, :tile_size]
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
    effective_weights = list()
    
    # 19 * 19 = 361 which is approximately 360
    # aim for half cells are pores ~180 pores
    # half 
    for pore_size in range(1, 19//2):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for i in range(0, 19, 2*pore_size):
            for j in range(0, 19, 2*pore_size):
                grid[i:i+pore_size, j:j+pore_size] = 1
                grid[i+pore_size:i+2*pore_size, j+pore_size:j+2*pore_size] = 1
        sample_grids.append(grid)
    for width in range(1, 19//2):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for i in range(0, 19, 2*width):
            grid[i:i+width] = 1
        sample_grids.append(grid)

    # 19 * 19 = 361 now we go for the full size pores
    for i in range(15, 20):
        grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        sample_grids.append(grid)
    
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    mask[:19, :19] = 1
    sample_grids = np.array([g*mask for g in sample_grids])
    effective_weights = np.array([np.sum(g) for g in sample_grids])
    
    sample_curves_diffs, _ = run_dft(sample_grids, inner_loops=inner_loops)
    sample_curves_diffs = sample_curves_diffs.numpy()
    sample_curves = np.cumsum(sample_curves_diffs, axis=1)

    if 's' in sys.argv[1:]:
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

    generated_grids = list()
    
    weighting_errs = list()
    norm_weighting_errs = list()
    generator_errs = list()
    target_curve_areas = list()
    for i, target_curve_diffs in enumerate(real_densities_diffs):
        target_curve = np.insert(np.cumsum(target_curve_diffs), 0, 0)
        target_curve_area = np.sum(target_curve) / len(target_curve)
        target_curve_areas.append(target_curve_area)
        
        weights = gen_curve_weights(target_curve[1:], sample_curves, 80)
        weights /= effective_weights
        weights /= np.sum(weights)
        weighted_target_curve = weights.T @ sample_curves
        weighted_target_curve = np.insert(weighted_target_curve, 0, 0)
        err = np.sum(np.abs(target_curve - weighted_target_curve)) / len(target_curve)
        weighting_errs.append(err)

        count_weights, norm_weights = round(weights, lat_size=lat_size)
        count_weights = count_weights.astype(np.int32)
        norm_weighted_target_curve = norm_weights.T @ sample_curves
        norm_weighted_target_curve = np.insert(norm_weighted_target_curve, 0, 0)
        err = np.sum(np.abs(target_curve - norm_weighted_target_curve)) / len(target_curve)
        norm_weighting_errs.append(err)
        
        generated_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        fill(generated_grid, count_weights, sample_grids, lat_size=lat_size)
        generated_grids.append(generated_grid)

        print(f'\r {i}/{len(real_densities_diffs)}', end='')
        '''
        generated_curve_diffs, _ = run_dft(np.array([generated_grid]), inner_loops=inner_loops)
        generated_curve = np.insert(np.cumsum(generated_curve_diffs), 0, 0)
        err = np.sum(np.abs(target_curve - generated_curve)) / len(target_curve)
        generator_errs.append(err)

        e = np.array(generator_errs)
        print(f'\r {i}/{len(real_densities_diffs)} | Error: {e.mean():.5f} | Max: {e.max():.5f} | Min: {e.min():.5f} | Std: {e.std():.5f}', end='')
        '''

        if 'g' not in sys.argv[1:]:
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
        # ax.plot(np.linspace(0, 1, N_ADSORP+1), weighted_target_curve, label='weighted curve')
        # ax.plot(np.linspace(0, 1, N_ADSORP+1), norm_weighted_target_curve, label='norm weighted curve')
        ax.plot(np.linspace(0, 1, N_ADSORP+1), generated_curve, label='generated curve')
        plt.legend()
        plt.show()
    print()

    ##############################################
    dft_batch_size = 128
    generated_grids = np.array(generated_grids)
    generated_curve_diffs = list()
    for i in range(0, len(generated_grids), dft_batch_size):
        print(f'\r {i}/{len(generated_grids)}', end='')
        gen_diffs, _ = run_dft(generated_grids[i:i+dft_batch_size], inner_loops=100)
        generated_curve_diffs.extend(gen_diffs)
    print()
    generated_curve_diffs = np.array(generated_curve_diffs)
    generated_curves = np.cumsum(generated_curve_diffs, axis=1)
    for i, (target_curve_diffs, generated_curve) in enumerate(zip(real_densities_diffs, generated_curves)):
        target_curve = np.insert(np.cumsum(target_curve_diffs), 0, 0)
        generated_curve = np.insert(generated_curve, 0, 0)

        err = np.sum(np.abs(target_curve - generated_curve)) / len(target_curve)
        generator_errs.append(err)

        e = np.array(generator_errs)
        print(f'\r {i}/{len(real_densities_diffs)} | Error: {e.mean():.5f} | Max: {e.max():.5f} | Min: {e.min():.5f} | Std: {e.std():.5f}', end='')
    print()
    ##############################################
    
    target_curve_areas = np.array(target_curve_areas)
    for errs, name in zip([weighting_errs, norm_weighting_errs, generator_errs], ['weighting', 'normed', 'gen']):
        errs = np.array(errs)
        print(f'{name}: {errs.mean():.5f} | {errs.std():.5f}')

        os.makedirs(f'figures/quick_inverse_error/', exist_ok=True)
        area_err = np.concatenate((np.array(target_curve_areas)[:, None], np.array(errs)[:, None]), axis=1)
        np.savetxt(f'figures/quick_inverse_error/errors_{name}.csv', area_err, delimiter=',')
        
        plt.hist(errs, bins=20)
        plt.title(name)
        plt.show()

        plt.scatter(target_curve_areas, errs)
        plt.title(f'{name} w.r.t area')
        plt.show()
        
    exit(0)
    


if __name__ == '__main__':
    main(inner_loops=int(sys.argv[1]))
