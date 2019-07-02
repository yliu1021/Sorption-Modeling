import os
import argparse
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np

import keras
from keras.layers import *
from keras.models import Model
from keras.models import load_model
from keras.callbacks import *

from constants import *


boost_dim = 0

def create_target_model():
    base_dir = MODEL_BASE_DIR
    model_loc = os.path.join(base_dir, "cnn_0-70.h5")

    model = load_model(model_loc)
    model.trainable = False
    return model


def create_generative_model():
    inputs = Input(shape=(1+boost_dim,))
    
    x = Dense(N_SQUARES)(inputs)
    x = Dense(N_SQUARES, activation='relu')(x)
    x = Reshape((GRID_SIZE, GRID_SIZE, 1))(x)
    x = Conv2D(16, 3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(16, 3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, strides=1, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(N_SQUARES, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=x)


def create_discriminative_model():
    inputs = Input(shape=(N_SQUARES,))
    
    x = Dense(N_SQUARES)(inputs)
    x = Reshape((GRID_SIZE, GRID_SIZE, 1))(x)
    x = Conv2D(16, 3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(16, 3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, strides=1, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=x)


def train_generative_model():
    os.makedirs("generative_model", exist_ok=True)
    os.makedirs("generative_model/checkpoints", exist_ok=True)
    
    x_train = np.random.uniform(0.0, 5.0, size=(10000, 1+boost_dim))
    y_train = x_train[:, 0]

    gen_model = create_generative_model()
    target_model = create_target_model()

    train_in = Input(shape=(1+boost_dim,))

    pred_tensor = target_model(gen_model(train_in))
    base_model = Model(inputs=train_in, outputs=pred_tensor)
    base_model.compile(optimizer='adam',
                       loss='mean_squared_error')
    base_model.summary()
    checkpoint_location = "generative_model/checkpoints/base_model.{epoch:02d}-{val_loss:.2E}.hdf5"
    base_model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.3,
                   callbacks=[EarlyStopping(patience=5),
                              TensorBoard(log_dir='./generative_model/logs/', histogram_freq=1, batch_size=16, write_grads=True,
                                          write_images=True, update_freq='epoch'),
                              ReduceLROnPlateau(patience=3, factor=0.2, min_lr=10**-7),
                              ModelCheckpoint(checkpoint_location, save_best_only=True)])

    gen_model.save("generative_model/generative_model.hdf5")
    target_model.save("generative_model/test_model.hdf5")


def predict(n_grids=100):
    os.makedirs("generative_model/grids0", exist_ok=True)
    gen_model = load_model("generative_model/generative_model.hdf5")
    print("generating random losses")
    x_rand = np.random.uniform(0.0, 10.0, size=(n_grids, 1+boost_dim))
    x_rand[:, 0] = np.linspace(0.0, 5.0, num=n_grids)
    print("predicting...")
    grids = gen_model.predict(x_rand)
    grids = np.around(grids).astype('int')
    grids = np.reshape(grids, (n_grids, GRID_SIZE, GRID_SIZE))
    print("saving...")
    for i in range(n_grids):
        print('\rfinished {} / {}'.format(i+1, n_grids), end='')
        path = os.path.join("generative_model/grids0", "grid_%04d.csv" % i)
        np.savetxt(path, grids[i, :, :], fmt="%i", delimiter=',')
    print('\nDone')


if __name__ == "__main__":
    train_generative_model()
    predict(n_grids=1000)
