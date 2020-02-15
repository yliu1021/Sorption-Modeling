import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


_filter_1 = tf.constant(
    [[[[0]], [[1]], [[0]]],
     [[[1]], [[0]], [[1]]],
     [[[0]], [[1]], [[0]]]],
    dtype=tf.float32
)


def run_dft(grids):    
    batch_size = grids.shape[0]
    r1 = tf.tile(grids, [1, 3, 3])
    r1 = r1[:, 20-1:2*20+1, 20-1:2*20+1]
    r1 = tf.reshape(r1, [batch_size, 20+2, 20+2, 1])
    vir1 = tf.nn.conv2d(r1, strides=[1,1,1,1], filters=_filter_1, padding='VALID')
    return vir1


def make_dft_model():
    inp = Input(shape=(20, 20), batch_size=64, name='dft_input')
    x = Lambda(lambda x: run_dft(x))(inp)
    model = Model(inputs=inp, outputs=x, name='dft_model')
    return model



cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='yliu1021')
tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
tpu_strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
with tpu_strategy.scope():
    dft_model = make_dft_model()

