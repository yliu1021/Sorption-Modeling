from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model

import sys
sys.path.append('..')

from constants import *

# Keras helper functions (losses/layers/activations)
def round_through(x):
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    x = (0.5 * x) + 0.5 # Gradient is steeper than regular sigmoid activation
    return K.clip(x, 0, 1)


def binary_sigmoid(x):
    return round_through(_hard_sigmoid(x))

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# network parameters
original_dim = GRID_SIZE * GRID_SIZE
input_shape = (GRID_SIZE, GRID_SIZE, 1)

# batch_size = 128
# kernel_size = 3
# filters = 16
# latent_dim = 2
# epochs = 15 

def make_vae_classical(**kwargs):
    latent_dim = kwargs.get('latent_dim', 2)
    intermediate_dim = kwargs.get('intermediate_dim', 512)

    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    x = Reshape((original_dim,))(x)
    x = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = latent_inputs
    x = Dense(intermediate_dim, activation='relu')(x)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= GRID_SIZE * GRID_SIZE
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    return encoder, decoder, vae


def make_vae_deconv(**kwargs):
    latent_dim = kwargs.get('latent_dim', 2)
    kernel_size = kwargs.get('filters', 3)
    filters = kwargs.get('filters', 16)

    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for _ in range(2):
        filters *= 2
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = latent_inputs
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for _ in range(2):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        filters //= 2

    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= GRID_SIZE * GRID_SIZE
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    return encoder, decoder, vae


def make_cvae(**kwargs):
    latent_dim = kwargs.get('latent_dim', 5)
    kernel_size = kwargs.get('filters', 3)
    filters = kwargs.get('filters', 512)
    boundary_expand = kwargs.get('boundary_expand', 4)

    encoder_grid_input = Input(shape=input_shape, name='encoder_grid_input')
    x = encoder_grid_input
    x = Reshape((GRID_SIZE, GRID_SIZE, 1))(x)
    # x = Lambda(lambda x: K.tile(x, [1, 3, 3]))(x)
    # x = Reshape((GRID_SIZE * 3, GRID_SIZE * 3, 1))(x)
    # x = Cropping2D(cropping=(GRID_SIZE-boundary_expand, GRID_SIZE-boundary_expand))(x)

    for _ in range(1):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   padding='same')(x)
        x = BatchNormalization()(x)
    
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)
    x = BatchNormalization()(x)

    for _ in range(1):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   padding='same')(x)
        x = BatchNormalization()(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)

    encoder_curve_inp = Input(shape=(N_ADSORP,), name='encoder_curve_input')
    curve_layer = Dense(40, activation='relu')(encoder_curve_inp)

    x = concatenate([x, curve_layer], axis=1)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model([encoder_grid_input, encoder_curve_inp], [z_mean, z_log_var, z], name='encoder')

    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.mean(K.sum(kl_loss, axis=-1))
    kl_loss *= -0.5
    encoder.add_loss(kl_loss)

    ####################################################################################
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    decoder_curve_inp = Input(shape=(N_ADSORP,), name='decoder_curve_input')
    
    x1 = Dense(256, activation='relu')(latent_inputs)
    x1 = BatchNormalization()(x1)
        
    x2 = Dense(256, activation='relu')(decoder_curve_inp)
    x2 = BatchNormalization()(x2)
    
    decoder_inputs = concatenate([x1, x2], axis=1)
    x = Dense(512, activation='relu')(decoder_inputs)
    x = BatchNormalization()(x)
    
    x = Dense(5*5*512, activation='relu')(x)
    x = BatchNormalization()(x)

    x = Reshape((5, 5, 512))(x)

    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(filters=1,
                        kernel_size=kernel_size,
                        activation='sigmoid',
                        padding='same',
                        name='decoder_output')(x)
    outputs = x
    
    # outputs = Cropping2D(cropping=(boundary_expand, boundary_expand))(x)

    # instantiate decoder model
    decoder = Model([latent_inputs, decoder_curve_inp], outputs, name='decoder')

    ####################################################################################
    
    # instantiate VAE model
    vae_curve_inp = Input(shape=(N_ADSORP,), name='vae_curve_input')
    vae_grid_inp = Input(shape=input_shape, name='vae_grid_input')
    outputs = decoder([encoder([vae_grid_inp, vae_curve_inp])[2], vae_curve_inp])
    vae = Model([vae_grid_inp, vae_curve_inp], outputs, name='vae')

    vae_grid_inp = AveragePooling2D(pool_size=4, strides=3, padding='valid')(vae_grid_inp)
    outputs = AveragePooling2D(pool_size=4, strides=3, padding='valid')(outputs)

    reconstruction_loss = binary_crossentropy(K.flatten(vae_grid_inp), K.flatten(outputs))
    vae.add_loss(reconstruction_loss)

    # vae = Model([grid_inp, grid_inp], outputs, name='vae')

    return encoder, decoder, vae
