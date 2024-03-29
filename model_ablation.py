import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from clr import *
from attention_blur import *
import json
# Model's architecture
arch = {
  "name": "Arch 1",
  "attention_module": 1,
  "pooling_method": "blur",
  "activation_function": "mish"
}
def avg_pool(inp, pool=2, stride=2):
    x = tf.keras.layers.AveragePooling2D(pool_size=pool, strides=stride)(inp)
    return x


def conv(inp, kernel, filt, dilation=2, stride=1, pad='same'):
    x = layers.Conv2D(filters=filt, kernel_size=kernel, strides=stride, padding=pad,
                      kernel_regularizer=keras.regularizers.l2(0.01))(inp)
    return x


def aug_block(inp, fout, dk, dv, nh, kernel=11):
    x = augmented_conv2d(inp, filters=fout, kernel_size=(kernel, kernel), depth_k=dk, depth_v=dv, num_heads=nh,
                         relative_encodings=True)
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.Activation(arch['activation_function'])(x)
    return x


def ARB(inp, fout, dk, dv, nh, kernel, aug=True):
    x = conv(inp, kernel=1, filt=fout * 4, pad='same')
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.Activation(arch['activation_function'])(x)
    x = layers.DepthwiseConv2D(kernel_size=kernel, strides=1, padding='same')(x)
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.Activation(arch['activation_function'])(x)
    if aug == True:
        a = aug_block(x, fout * 4, dk, dv, nh, kernel)
        # x = layers.Add()([a, x])
        x = tf.concat([a, x], axis=-1)
    x = conv(x, kernel=1, filt=fout, pad='same')
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.Activation(arch['activation_function'])(x)
    return x


def transition(inp, filters):
    x = conv(inp, kernel=1, filt=filters, pad='same')
    if arch['pooling_method'] == 'Blur':
        x = BlurPool2D()(x)
    elif arch['pooling_method'] == 'Average':
        x = layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    elif arch['pooling_method'] == 'Max':
        x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    else:
        assert 0, print('Unrecogized pooling method: ', arch['pooling_method'])
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    return x

def conv_block(x, growth_rate, kernel_size = 3):


    # 1x1 Convolution (Bottleneck layer)
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.Activation(arch['activation_function'])(x)
    x = layers.Conv2D(growth_rate * 4, 1, strides=1, padding='same')(x)

    # 3x3 Convolution
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.Activation(arch['activation_function'])(x)
    x = layers.Conv2D(growth_rate, kernel_size, strides=1, padding='same')(x)
    return x

def dense(x, kernel, num, nh=4, filters=10, aug=True):
    x_list = [x]
    if arch['attention_module'] == 0:
        aug = False
    for i in range(num):
        x = ARB(x, filters, 0.1, 0.1, nh, kernel, aug)
        x_list.append(x)
        x = tf.concat(x_list, axis=-1)
    return x


def create_model(arch_num):
    global arch
    arch = json.load(open('configs/Arch.json'))[arch_num]

    Input = layers.Input(shape=(224, 224, 3), dtype=tf.float32)  # 224
    x = dense(Input, 5, num=8, aug=False)
    ###############################
    y = transition(x, 64)  # 112
    x = dense(y, 5, num=8, aug=False)
    ###############################
    y = transition(x, 64)  # 56
    x = dense(y, nh=1, kernel=3, num=6)
    ###############################
    y = transition(x, 64)  # 28
    x = dense(y, nh=4, kernel=3, num=8)
    ###############################
    y = transition(x, 64)  # 14
    x = dense(y, nh=4, kernel=3, num=10)
    ###############################
    y = transition(x, 64)  # 7
    x = dense(y, nh=4, kernel=3, num=12)
    ###############################
    y = transition(x, 128)  # 4
    x = dense(y, nh=4, kernel=3, num=14)
    ###############################
    x = transition(x, 128)  # 2
    x = dense(x, nh=4, kernel=2, num=32)
    x = aug_block(x, 100, 0.1, 0.1, 10, 2)
    ###############################
    x = avg_pool(x)  # 1
    x = conv(x, 1, 42, 1, 1)
    x = tf.keras.layers.Lambda(lambda x: tf.keras.activations.relu(x, max_value=1.))(x)
    x = tf.keras.layers.Reshape((21, 2))(x)
    # x = tf.cast(x, tf.float32)
    model = tf.keras.Model(Input, x)
    model.compile(optimizer=keras.optimizers.SGD(), loss=rmse, metrics=['accuracy'])
    return model
def rmse(x, y):
    # print(x.shape, y.shape)
    x = tf.math.sqrt(tf.keras.losses.MSE(x, y))
    # print(x.shape, y.shape)
    # exit()
    return x
'''policy = tf.keras.mixed_precision.experimental.Policy('float32')
tf.keras.mixed_precision.experimental.set_policy(policy)'''

if __name__ == "__main__":
    # for i in range(1,13):
    #     model = create_model(i)
    #     print('Create_model arch:',i)
    #     model.summary()
    model = create_model(1)
    model.summary()