from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, Lambda, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import pickle

import vae_models

def load_vae(model_name):
    encoder, decoder, vae = vae_models.make_cvae()

    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    vae.compile(optimizer=opt)

    vae.load_weights(model_name+'/checkpoints/ck.ckpt')
    # encoder.set_weights(vae.get_layer('encoder').get_weights())
    # decoder.set_weights(vae.get_layer('decoder').get_weights())

    # vae._make_train_function()
    # with open(model_name+'/optimizer.pkl', 'rb') as f:
    #     weight_values = pickle.load(f)
    # import pdb; pdb.set_trace()
    # vae.optimizer.set_weights(weight_values)

    return encoder, decoder, vae


if __name__ == '__main__':
    load_vae('vae_conditional')
