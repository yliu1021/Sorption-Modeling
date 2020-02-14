from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, Lambda, Cropping2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import vae_models

def load_vae(model_name):
    encoder, decoder, vae = vae_models.make_cvae()

    vae.load_weights(model_name+'/checkpoints/cp.ckpt')
    encoder.set_weights(vae.get_layer('encoder').get_weights())
    decoder.set_weights(vae.get_layer('decoder').get_weights())

    return encoder, decoder, vae


if __name__ == '__main__':
    load_vae('vae_conditional')
