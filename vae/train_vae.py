import math
import os
import sys
from tqdm import tqdm
sys.path.append('..')

import data_generator
from constants import *
import data
import vae_models
from vae_options import *

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

def start_training(model_name, **kwargs):
    epochs = kwargs.get('epochs', 30)
    batch_size = kwargs.get('batch_size', 128)

    encoder, decoder, vae = vae_models.make_cvae()

    checkpoint_path = os.path.join(model_name, "checkpoints/cp.ckpt")
    # checkpoint_path = os.path.join(model_name, "checkpoints/cp-{epoch:04d}.ckpt")

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  verbose=1
                                  )

    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    vae.compile(optimizer=opt)

    # gdg = data_generator.GridDataGenerator(directory='../data_generation/', shift_range=19, rotate=True, flip=True, validation_split=0.1, test_split=0.1, batch_size=batch_size)
    # gdg = data_generator.GridDataGenerator(directory='../generative_model_3/step_0/', shift_range=19, rotate=True, flip=True, validation_split=0.1, test_split=0.1, batch_size=batch_size)
    # gdg = data_generator.GridDataGenerator(directory='../generative_model_3/step_0/', rotate=True, flip=True, batch_size=batch_size)
    gdg = data_generator.GridDataGenerator(directory='../generative_model_3/step_0/', batch_size=batch_size, validation_split=0.1)

    # vae.fit([x_train, y_train], epochs=epochs, batch_size=batch_size)
    history = vae.fit(gdg.flow(),
                      steps_per_epoch=math.ceil(gdg.numTrainGrids/batch_size),
                      validation_data=gdg.flow_validation(),
                      validation_steps=math.ceil(len(gdg.validationSet)/batch_size),
                      epochs=epochs,
                      callbacks=[cp_callback])

    print(history.history)
    
    os.makedirs(model_name, exist_ok=True)
    vae.save(os.path.join(model_name, "vae.tf"))

if __name__ == '__main__':
    start_training(model_name="vae_conditional", epochs=3, batch_size=32)