import os
import glob
import shutil
import random
import argparse
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from tensorflow.keras.models import load_model

import models
import data
from constants import *


base_dir = 'generative_model_default'
# base_dir = 'generative_model_3_cpu'
# base_dir = 'generative_model_seed_grids'
base_dir = 'generative_model_new'
step_index = -1
index = 0
run = True
def press(event):
    global index
    global step_index
    global run
    if event.key == 'd':
        index += 1
    elif event.key == 'a':
        index -= 1
    elif event.key == 'w':
        step_index += 1
    elif event.key == 's':
        step_index -= 1
    if event.key == 'q':
        run = False


def show_grids(v):
    global index
    global step_index
    global run

    all_files = list()
    step_dirs = glob.glob(os.path.join(base_dir, 'step*'))
    
    def extract_step_num(d):
        d = d.split('/')[-1]
        d = d[4:]
        if d.startswith('_'):
            d = d[1:]
        try:
            return int(d)
        except:
            return 0

    step_dirs.sort(key=extract_step_num)
    for i, step_dir in enumerate(step_dirs):
        if step_dir.endswith('step_{}'.format(v)) or step_dir.endswith('step{}'.format(v)):
            step_index = i
        grid_files = glob.glob(os.path.join(step_dir, 'grids/grid_*.csv'.format(v)))
        density_files = glob.glob(os.path.join(step_dir, 'results/density_*.csv'.format(v)))
        target_density_files = glob.glob(os.path.join(step_dir, 'target_densities/artificial_curve_*.csv'))
        if len(target_density_files) == 0:
            target_density_files = glob.glob(os.path.join(step_dir, 'artificial_metrics/artificial_metric_*.csv'))
        grid_files.sort()
        density_files.sort()
        target_density_files.sort()
        if len(target_density_files) == 0:
            all_step_files = list(zip(grid_files, density_files))
        else:
            all_step_files = list(zip(grid_files, density_files, target_density_files))
        if len(all_step_files) != 0:
            all_files.append(all_step_files)

    predictor_model = None
    print('Attempting to load predictor model...')
    try:
        predictor_model = load_model(os.path.join(base_dir, 'model_saves/predictor_step_{}.hdf5'.format(v)),
                                     custom_objects={'binary_sigmoid': models.binary_sigmoid})
        print('Loaded successfully')
    except:
        print("Couldn't load predictor_model")

    fig = plt.figure(figsize=(10, 4))
    fig.canvas.mpl_connect('key_press_event', press)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    while run:
        s = step_index % len(all_files)
        i = index % len(all_files[s])
        files = all_files[s][i]
        target_density_file = None
        if len(files) == 2:
            grid_file, density_file = files
        else:
            grid_file, density_file, target_density_file = files

        df = pd.read_csv(density_file, index_col=0)
        relative_humidity = np.arange(41) * STEP_SIZE
        
        grid = np.genfromtxt(grid_file, delimiter=',')
        density = np.append(df['0'][:N_ADSORP], 1)
        target_density = None
        if target_density_file:
            target_density = np.insert(np.genfromtxt(target_density_file), 0, 0)
            target_density = np.cumsum(target_density)
        predicted_density = None
        if predictor_model:
            predicted_density = np.insert(predictor_model.predict(np.array([grid]))[0], 0, 0)
            predicted_density = np.cumsum(predicted_density)

        fig.suptitle('Step {}, Grid {}'.format(s, i))

        ax = plt.subplot(1, 2, 1)
        ax.clear()
        ax.set_title('Grid (Black = Solid, White = Pore)')
        ax.set_yticks(np.linspace(0, 20, 5))
        ax.set_xticks(np.linspace(0, 20, 5))
        ax.pcolor(1 - grid, cmap='Greys', vmin=0.0, vmax=1.0)
        ax.set_aspect('equal')

        ax = plt.subplot(1, 2, 2)
        ax.clear()
        ax.set_title('Adsorption Curve')
        ax.plot(relative_humidity, density, label='DFT')
        if predicted_density is not None:
            ax.plot(relative_humidity, predicted_density, label='Predictor')
        if target_density is not None:
            ax.plot(relative_humidity, target_density, label='Target')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Relative Humidity')
        ax.set_ylabel('Proportion of Pores filled')
        ax.set_aspect('equal')
        ax.legend()
        
        plt.show()
        plt.waitforbuttonpress(timeout=-1)


def show_validation():
    model_step = input('Model step: ')
    model_dir = os.path.join(base_dir, 'model_saves')
    predictor_file = os.path.join(model_dir, 'predictor_step_{}.hdf5'.format(model_step))
    generator_file = os.path.join(model_dir, 'generator_step_{}.hdf5'.format(model_step))

    try:
        print('Loading predictor model')
        predictor = load_model(predictor_file, custom_objects={'worst_abs_loss': models.worst_abs_loss})
        predictor.summary()
    except Exception as e:
        print('Unable to load predictor model: {}'.format(e))
        return

    try:
        print('Loading generator model')
        generator = load_model(generator_file, custom_objects={'binary_sigmoid': models.binary_sigmoid})
        generator.summary()
    except Exception as e:
        print('Unable to load generator model: {}'.format(e))
        return

    validation_dir = os.path.join(base_dir, 'validation')
    try:
        os.makedirs(validation_dir, exist_ok=False)
    except FileExistsError:
        shutil.rmtree(validation_dir)
        os.makedirs(validation_dir, exist_ok=False)
    
    grids_dir = os.path.join(validation_dir, 'grids')
    densities_dir = os.path.join(validation_dir, 'results')
    target_densities_dir = os.path.join(validation_dir, 'target_densities')

    os.makedirs(grids_dir, exist_ok=True)
    # First create grids that have increasing pore size
    # 0x0, 1x1, 2x2, 3x3, ..., 20x20
    print('Creating baseline grids (0x0, 1x1, ..., 20x20)')
    for i in range(GRID_SIZE+1):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        grid[:i, :i] = 1
        path = os.path.join(grids_dir, 'grid_%04d.csv'%i)
        np.savetxt(path, grid, fmt='%i', delimiter=',')
    # Then evaluate those grids
    os.system('./fast_dft {}'.format(validation_dir))
    
    grids = list()
    target_densities = list()
    metadata = list()
    # Then create grids using generator
    # ---------------------------------
    # First use the curves from the 0x0, 1x1, 2x2, ..., 20x20 grids
    # For each nxn grid curve, we use latent codes that vary in one dimension
    # from 0 to 1 for each dimension 0 to `boost_dim`
    print('Creating from generator')
    density_diffs = list()
    densities_dir = os.path.join(validation_dir, 'results')
    density_files = glob.glob(os.path.join(densities_dir, 'density_*.csv'))
    density_files.sort()
    latent_codes = list()
    boost_dim = 5
    for i, density_file in enumerate(density_files):
        density = np.genfromtxt(density_file, delimiter=',', skip_header=1, max_rows=N_ADSORP)[:, 1]
        density[0] = 0
        density = np.append(density, 1.0)
        diffs = np.diff(density)
        for boost_dim_ind in range(boost_dim):
            for step in np.linspace(0.0, 1.0, num=5):
                latent_code = [0.5] * boost_dim
                latent_code[boost_dim_ind] = step
                density_diffs.append(diffs)
                latent_codes.append(latent_code)
                metadata.append({'size': i, 'lc': latent_code, 
                                 'boost_dim_ind': boost_dim_ind, 'step': float(step)})
                target_densities.append(density)
    density_diffs = np.array(density_diffs)
    latent_codes = np.array(latent_codes)
    print('Generating {} grids...'.format(len(density_diffs)), end='', flush=True)
    start = time.time()
    new_grids = generator.predict([density_diffs, latent_codes])
    durr = time.time() - start
    print(' Done.\tTook {:.3f} seconds ({:.2f}ms per grid)'.format(durr, durr/len(new_grids)*1000))
    grids.extend(new_grids)
    
    # Now just create some random grids
    num_new_grids = 500
    artificial_curves, latent_codes = data.make_generator_input(num_new_grids, boost_dim, allow_squeeze=True, as_generator=False)
    artificial_curves = list(zip(artificial_curves, latent_codes))
    artificial_curves.sort(key=lambda x: np.sum(np.cumsum(x[0])), reverse=True)
    artificial_curves, latent_codes = list(zip(*artificial_curves))
    artificial_curves = np.array(artificial_curves)
    latent_codes = np.array(latent_codes)
    print('Generating {} grids...'.format(len(artificial_curves)), end='', flush=True)
    start = time.time()
    new_grids = generator.predict([artificial_curves, latent_codes])
    durr = time.time() - start
    print(' Done. Took {:.3f} seconds ({:.2f}ms per grid)'.format(durr, durr/len(new_grids)*1000))
    grids.extend(new_grids)
    metadata += [{'num': i, 'area': np.sum(np.cumsum(ac)), 'lc': lc}
                    for i, (ac, lc) in enumerate(zip(artificial_curves, latent_codes))]

    # Save the new grids
    shutil.rmtree(grids_dir)
    shutil.rmtree(densities_dir)
    os.makedirs(grids_dir, exist_ok=True)
    for i, grid in enumerate(grids):
        path = os.path.join(grids_dir, 'grid_%04d.csv'%i)
        np.savetxt(path, grid, fmt='%i', delimiter=',')
    # Evaluate densities
    print('Evaluating generated grids')
    start = time.time()
    os.system('./fast_dft {}'.format(validation_dir))
    durr = time.time() - start
    print('Done. Took {:.3f} seconds ({:.2f}ms per grid)'.format(durr, durr/len(grids)*1000))

    density_files = glob.glob(os.path.join(densities_dir, 'density_*.csv'))
    density_files.sort()
    densities = list()
    for density_file in density_files:
        density = np.genfromtxt(density_file, delimiter=',', skip_header=1, max_rows=N_ADSORP)[:, 1]
        density = np.append(density, 1)
        density[0] = 0
        densities.append(density)
    
    for artificial_curve in artificial_curves:
        density = np.insert(np.cumsum(artificial_curve), 0, 0)
        target_densities.append(density)

    print('Predicting curves')
    predicted_diffs = predictor.predict(np.array(grids))
    predicted_densities = list()
    for predicted_diff in predicted_diffs:
        predicted_density = np.insert(np.cumsum(predicted_diff), 0, 0)
        predicted_densities.append(predicted_density)

    all_info = list(zip(metadata, grids, densities, target_densities, predicted_densities))
    
    global index
    
    fig = plt.figure(figsize=(10, 4))
    fig.canvas.mpl_connect('key_press_event', press)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    def setup_plt(md, grid, density, t_density, p_density):
        if 'size' in md and 'lc' in md:
            s = md['size']
            l = md['lc']
            fig.suptitle('Grid {}x{}, lc {}'.format(s, s, l))
        elif 'num' in md and 'area' in md and 'lc' in md:
            i = md['num']
            a = md['area']
            l = md['lc']
            fig.suptitle('Grid {}/{}, lc {}, area {:.3f}'.format(i, num_new_grids,
                                                                 np.array2string(l, precision=2),
                                                                 a))

        ax = plt.subplot(1, 2, 1)
        ax.clear()
        ax.set_title('Grid (Black = Solid, White = Pore)')
        ax.set_yticks(np.linspace(0, 20, 5))
        ax.set_xticks(np.linspace(0, 20, 5))
        ax.pcolor(1 - grid, cmap='Greys', vmin=0.0, vmax=1.0)
        ax.set_aspect('equal')

        ax = plt.subplot(1, 2, 2)
        ax.clear()
        ax.set_title('Adsorption Curve')
        relative_humidity = np.arange(41) * STEP_SIZE
        ax.plot(relative_humidity, density, label='DFT')
        ax.plot(relative_humidity, p_density, label='Predictor')
        ax.plot(relative_humidity, t_density, label='Target')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Relative Humidity')
        ax.set_ylabel('Proportion of Pores filled')
        ax.set_aspect('equal')
        ax.legend()

    while run:
        i = index % len(all_info)
        
        setup_plt(*all_info[i])
        
        plt.show()
        plt.waitforbuttonpress(timeout=-1)
    
    # Get errors
    print('Calculating errors')
    generator_errs = list()
    predictor_errs = list()
    cross_errs = list()
    for md, grid, density, t_density, p_density in all_info:
        generator_err = np.sum(np.abs(density - t_density)) / (N_ADSORP + 1)
        predictor_err = np.sum(np.abs(density - p_density)) / (N_ADSORP + 1)
        cross_err = np.sum(np.abs(t_density - p_density)) / (N_ADSORP + 1)
        generator_errs.append(generator_err)
        predictor_errs.append(predictor_err)
        cross_errs.append(cross_err)
    
    baseline_generator_errs = dict()
    baseline_predictor_errs = dict()
    baseline_cross_errs = dict()

    areas = list()
    random_generator_errs = list()
    random_predictor_errs = list()
    random_cross_errs = list()

    for md, g_err, p_err, c_err in zip(metadata, generator_errs, predictor_errs, cross_errs):
        if 'boost_dim_ind' in md and 'step' in md:
            b = md['boost_dim_ind']
            s = md['step']
            k = (b, s)
            if k in baseline_generator_errs:
                baseline_generator_errs[k].append(g_err)
            else:
                baseline_generator_errs[k] = [g_err]
            if k in baseline_predictor_errs:
                baseline_predictor_errs[k].append(p_err)
            else:
                baseline_predictor_errs[k] = [p_err]
            if k in baseline_cross_errs:
                baseline_cross_errs[k].append(c_err)
            else:
                baseline_cross_errs[k] = [c_err]
        elif 'num' in md and 'area' in md and 'lc' in md:
            i = md['num']
            a = md['area']
            l = md['lc']
            areas.append(a)
            random_generator_errs.append(g_err)
            random_predictor_errs.append(p_err)
            random_cross_errs.append(c_err)

    def plot_baseline(baseline_err, title):
        k = list(baseline_err.keys())
        sizes = len(baseline_err[k[0]])
        labels = ['Grid {}x{}'.format(s, s) for s in range(sizes)]
        x = np.arange(sizes)
        offsets = np.linspace(-0.4, 0.4, num=len(k))
        w = offsets[1] - offsets[0]
        for i, (s, l) in enumerate(zip(k, labels)):
            offset = offsets[i]
            errs = baseline_err[s]
            plt.bar(x + offset, errs, w, label=k)
        plt.ylim(0, 1)
        plt.xticks(x, labels, rotation=90)
        plt.title(title)
        plt.show()
        plt.waitforbuttonpress(timeout=-1)
    
    plot_baseline(baseline_generator_errs, 'Generator Errors')
    plot_baseline(baseline_predictor_errs, 'Predictor Errors')
    plot_baseline(baseline_cross_errs, 'Cross Errors')

    plt.plot(areas, random_generator_errs, label='Generator Errors')
    plt.plot(areas, random_predictor_errs, label='Predictor Errors')
    plt.plot(areas, random_cross_errs, label='Cross Errors')
    plt.legend()
    plt.title('Errors')
    plt.xlabel('Area under adsorption curve')
    plt.ylabel('Error (unsigned area between curves)')
    plt.ylim(0, 1)
    plt.show()
    plt.waitforbuttonpress(timeout=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("v", nargs="?", help="Show the grids/results of step v",
                        type=int, default=-1)
    args = parser.parse_args()
    v = args.v
    if v >= 0:
        show_grids(v)
    else:
        show_validation()
