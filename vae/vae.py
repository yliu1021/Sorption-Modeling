import sys
sys.path.append('..')

import numpy as np
import os

from constants import *
import data
import vae_models

save_dir = 'cvae'

def start_training(**kwargs):
    epochs = kwargs.get('epochs', 30)
    batch_size = kwargs.get('batch_size', 128)

    # encoder, decoder, vae = vae_models.make_vae_classical()
    # encoder, decoder, vae = vae_models.make_vae_deconv()
    encoder, decoder, vae = vae_models.make_cvae()

    num_test = 1
    # x_train, y_train = data.get_all_data(matching='vae_cnn_data', augment_factor=20)
    x_train, y_train = data.get_all_data(matching='../generative_model_3', augment_factor=500)
    x_test = x_train[:num_test]
    y_test = y_train[:num_test]
    x_train = x_train[num_test:]
    y_train = y_train[num_test:]
    x_train = np.reshape(x_train, [-1, GRID_SIZE, GRID_SIZE, 1])
    x_test = np.reshape(x_test, [-1, GRID_SIZE, GRID_SIZE, 1])

    vae.compile(optimizer='rmsprop')
    vae.fit([x_train, y_train], epochs=epochs, batch_size=batch_size)

    os.makedirs(save_dir, exist_ok=True)
    encoder.save(os.path.join(save_dir, "encoder.tf"))
    decoder.save(os.path.join(save_dir, "decoder.tf"))
    vae.save(os.path.join(save_dir, "vae.tf"))

if __name__ == '__main__':
    start_training(epochs=3, batch_size=2048)