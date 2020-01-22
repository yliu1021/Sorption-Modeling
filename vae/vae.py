import math
import numpy as np
import os
import sys
sys.path.append('..')

import data_generator
from constants import *
import data
import vae_models
from vae_options import *

from tensorflow.keras.optimizers import Adam

def start_training(**kwargs):
    epochs = kwargs.get('epochs', 30)
    batch_size = kwargs.get('batch_size', 128)

    # encoder, decoder, vae = vae_models.make_vae_classical()
    # encoder, decoder, vae = vae_models.make_vae_deconv()
    encoder, decoder, vae = vae_models.make_cvae()
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    vae.compile(optimizer=opt)

    gdg = data_generator.GridDataGenerator(directory='../generative_model_3/step_0/', shift_range=19, rotate=True, flip=True, validation_split=0.1, test_split=0.1, batch_size=batch_size)

    # vae.fit([x_train, y_train], epochs=epochs, batch_size=batch_size)
    history = vae.fit_generator(gdg.flow(), steps_per_epoch=math.ceil(gdg.numTrainGrids // batch_size), epochs=epochs)

    print(history.history)
    
    os.makedirs(model_name, exist_ok=True)
    encoder.save(os.path.join(model_name, "encoder.tf"))
    decoder.save(os.path.join(model_name, "decoder.tf"))
    vae.save(os.path.join(model_name, "vae.tf"))

if __name__ == '__main__':
    start_training(epochs=3, batch_size=32)
