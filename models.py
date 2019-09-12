import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K

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


def worst_abs_loss(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred))


def freeze(model):
    for layer in model.layers:
        layer.trainable = False


def unfreeze(model):
    for layer in model.layers:
        layer.trainable = True


# Predictor model
def make_predictor_model(**kwargs):
    first_filter_size = kwargs.get('first_filter_size', 3)
    last_conv_depth = kwargs.get('last_conv_depth', 256)
    dense_layer_size = kwargs.get('dense_layer_size', 2048)
    boost_dim = kwargs.get('boost_dim', 5)
    num_convs = kwargs.get('num_convs', 1)
    boundary_expand = kwargs.get('boundary_expand', 4)

    inp = Input(shape=(GRID_SIZE, GRID_SIZE), name='proxy_enforcer_input')
    x = Lambda(lambda x: K.tile(x, [1, 3, 3]))(inp)
    x = Reshape((GRID_SIZE * 3, GRID_SIZE * 3, 1))(x)
    x = Cropping2D(cropping=(GRID_SIZE-boundary_expand, GRID_SIZE-boundary_expand))(x)

    x = Conv2D(16, first_filter_size, padding='valid', name='preconv')(x)
    for i in range(1, num_convs+1):
        x = Conv2D(32, 3, padding='valid', name='preconv{}'.format(i))(x)
    x = Conv2D(64, 3, padding='valid', name='preconv_final')(x)
    x = Dropout(0.1)(x)
    
    x = Conv2D(128, 3, padding='valid', strides=2, name='conv_stride_1')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(last_conv_depth, 3, padding='valid', strides=2, name='conv_stride_2')(x)   
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.1)(x)

    x_fc1 = Dense(dense_layer_size, name='hidden_fc_1', activation='relu')(x)
    x_fc1 = Dropout(0.5)(x_fc1)
    hidden = Dense(dense_layer_size, name='hidden_fc_final', activation='relu')(x_fc1)
    hidden = Dropout(0.5)(hidden)

    latent_code = Dense(boost_dim, name='latent_codes')(x_fc1)
    out = Dense(N_ADSORP, name='out', activation='softmax')(hidden)

    lc_model = Model(inputs=[inp], outputs=[latent_code], name='latent_code_model')    
    model = Model(inputs=[inp], outputs=[out], name='predictor_model')

    return model, lc_model


# Generator model
def make_generator_model(**kwargs):
    first_conv_depth = kwargs.get('first_conv_depth', 128)
    pre_deconv1_depth = kwargs.get('pre_deconv1_depth', 128)
    post_deconv2_depth = kwargs.get('post_deconv2_depth', 32)
    last_filter_size = kwargs.get('last_filter_size', 6)
    boost_dim = kwargs.get('boost_dim', 5)
    generator_activation = kwargs.get('generator_activation', binary_sigmoid)
    
    lc_merge_depth = 512
    
    latent_code_inp = Input(shape=(boost_dim,))
    latent_code_processed = Dense(512, use_bias=True, activation='relu')(latent_code_inp)

    inp = Input(shape=(N_ADSORP,))
    x = Concatenate()([inp, latent_code_inp])

    Q_GRID_SIZE = GRID_SIZE // 4

    x = Dense(2048, name='fc1')(x)

    x = Dense(2048, name='fc2')(x)
    x = LeakyReLU()(x)

    x = Dense(Q_GRID_SIZE * Q_GRID_SIZE * first_conv_depth, name='fc3')(x)
    x = LeakyReLU()(x)

    x = Reshape((Q_GRID_SIZE, Q_GRID_SIZE, first_conv_depth))(x)

    x = Conv2DTranspose(first_conv_depth, 5, strides=1, padding='same', name='pre_deconv1')(x)
    x = LeakyReLU()(x)
    
    x = Conv2DTranspose(pre_deconv1_depth, 3, strides=1, padding='same', name='pre_deconv2')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(lc_merge_depth, 3, strides=2, padding='same', name='deconv_expand1')(x)
    x = LeakyReLU()(x)

    latent_code_merge = Dense(lc_merge_depth, use_bias=True)(latent_code_processed)
    x = Multiply()([x, latent_code_merge])

    x = Conv2DTranspose(64, 3, strides=2, padding='same', name='deconv_expand2')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(post_deconv2_depth, 3, strides=1, padding='same', name='post_deconv')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(32, 3, strides=1, padding='same', name='smooth_conv')(x)
    x = LeakyReLU()(x)
    
    out = Conv2D(1, last_filter_size, strides=1, padding='same',
                 activation=generator_activation, name='generator_conv')(x)
    out = Reshape((GRID_SIZE, GRID_SIZE))(out)

    model = Model(inputs=[inp, latent_code_inp], outputs=[out],
                  name='generator_model')
    
    return model


# Generator auxiliary latent code decoder
def make_generator_lc_decoder(**kwargs):
    boost_dim = kwargs.get('boost_dim', 5)
    decoder_depth = kwargs.get('decoder_depth', 512)

    inp = Input(shape=(GRID_SIZE, GRID_SIZE), name='generator_lc_decoder_input')
    x = Reshape((GRID_SIZE, GRID_SIZE, 1))(inp)
    x = Conv2D(8, 3, strides=1, padding='valid', name='preconv')(x)
    x = Flatten()(x)

    x = Dense(decoder_depth, activation='relu', name='decoder_intermediate_layer')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu', name='decoder_final_layer0')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu', name='decoder_final_layer1')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu', name='decoder_final_layer2')(x)
    x = Dropout(0.5)(x)

    x = Dense(boost_dim, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=x, name='generator_lc_decoder')
    return model
