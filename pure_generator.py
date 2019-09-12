import os
import shutil
import glob
import sys
from multiprocessing import Pool
from random import randint, shuffle, sample
import json
import argparse
import math

import numpy as np
np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 1000
import pandas as pd
# os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm

import data
import models
from constants import *


def castF(x):
    return K.cast(x, K.floatx())

def castB(x):
    return K.cast(x, bool)

def iou_loss_core(true,pred):
    intersection = true * pred
    notTrue = 1 - true
    union = true + (notTrue * pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())


def neg_iou_loss_core(true, pred):
    return -(iou_loss_core(true, pred))


def iou_metric(true, pred):

    tresholds = [0.5 + (i*.05)  for i in range(10)]

    #flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, 0.5))

    #total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    #has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))    
    pred1 = castF(K.greater(predSum, 1))

    #to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    #separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    #getting iou and threshold comparisons
    iou = iou_loss_core(testTrue,testPred) 
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    #mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    #to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives) 

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])


def make_dirs(*dirs, exist_ok=True):
    for d in dirs:
        os.makedirs(d, exist_ok=exist_ok)


def combine_generator_auxiliary(generator_model, gen_lc_decoder):
    grid_inp = Input(shape=(GRID_SIZE, GRID_SIZE), name='train_generator_lc_decoder_input')
    target_curve_inp = Input(shape=(N_ADSORP), name='target_curve_input')
    decoder_out = gen_lc_decoder(grid_inp)
    generator_out = generator_model([target_curve_inp, decoder_out])

    return Model(inputs=[target_curve_inp, grid_inp], outputs=generator_out, name='train_model')
    
        
def train_step(step, generator_model, gen_lc_decoder, train_model, **kwargs):
    # Set up our directories
    base_dir = kwargs.get('base_dir', 'generative_model_pure')
    step_dir = os.path.join(base_dir, 'step_{}'.format(step))
    grids_dir = os.path.join(step_dir, 'grids')
    densities_dir = os.path.join(step_dir, 'results')
    target_densities_dir = os.path.join(step_dir, 'target_densities')
    model_save_dir = os.path.join(base_dir, 'model_saves')
    model_logs = os.path.join(step_dir, 'model_logs')
    make_dirs(step_dir,
              grids_dir, densities_dir, target_densities_dir,
              model_save_dir, model_logs)
    model_save_file = os.path.join(model_save_dir, 'model_step_{}.hdf5'.format(step))
    boost_dim = kwargs.get('boost_dim', 5)
    
    # Train our generator/decoder
    augment_factor = kwargs.get('augment_factor', 50)
    train_grids, train_curves = data.get_all_data(matching=base_dir, augment_factor=augment_factor)
    
    epochs = kwargs.get('epochs', 200)
    batch_size = kwargs.get('batch_size', 64)
    
    loss_func = kwargs.get('loss_func', neg_iou_loss_core)
    learning_rate = kwargs.get('learning_rate', 10**-1)
    optimizer = SGD(learning_rate, clipnorm=1.0)

    lr_patience = max(min(int(round(epochs * 0.25)), 30), 3)
    es_patience = max(int(round(epochs * 0.8)), 5)
    train_model.compile(optimizer, loss=loss_func, metrics=[iou_metric, 'accuracy', 'mae'])
    train_model.fit(x=[train_curves, train_grids], y=train_grids, batch_size=batch_size,
                    epochs=epochs, validation_split=0.1,
                    callbacks=[ReduceLROnPlateau(patience=lr_patience, factor=0.1),
                               EarlyStopping(patience=es_patience, restore_best_weights=True),
                               TensorBoard(log_dir=model_logs, histogram_freq=max(epochs//100,1),
                                           write_graph=False, write_images=False)])

    generator_model.save(model_save_file)

    # Generate new data
    num_new_grids = kwargs.get('num_new_grids', 100)
    data_upscale_factor = kwargs.get('data_upscale_factor', 1.5)
    artificial_curves, latent_codes = data.make_generator_input(int(num_new_grids*data_upscale_factor), boost_dim, as_generator=False)
    generated_grids = generator_model.predict([artificial_curves, latent_codes])
    print(generated_grids[:2])
    generated_grids = np.around(generated_grids)
    saved_grids = generated_grids.astype('int')
    for i, grid in enumerate(saved_grids):
        path = os.path.join(grids_dir, 'grid_%04d.csv'%i)
        np.savetxt(path, grid, fmt='%i', delimiter=',')

    print('Evaluating candidate grids')
    os.system('./fast_dft {}'.format(step_dir))

    for i, artificial_curve in enumerate(artificial_curves):
        path = os.path.join(target_densities_dir, 'artificial_curve_%04d.csv'%i)
        np.savetxt(path, artificial_curve, fmt='%f', delimiter=',')


def start_training(**kwargs):
    base_dir = kwargs.get('base_dir', 'generative_model_pure')
    make_dirs(base_dir)

    generator_model = models.make_generator_model(generator_activation='sigmoid', **kwargs)
    gen_lc_decoder = models.make_generator_lc_decoder(**kwargs)
    train_model = combine_generator_auxiliary(generator_model, gen_lc_decoder)

    train_steps = kwargs.get('train_steps', 10)
    for step in range(train_steps):
        train_step(step, generator_model, gen_lc_decoder, train_model, **kwargs)


if __name__ == '__main__':
    start_training()
