import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
'''from clr import *
from attention_blur import *'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Model's architecture

def ConvBlock(inp, filt, kernel_size = 7):
    x = layers.Conv2D(filters=filt, kernel_size=kernel_size,
                      strides=1, padding='same',)(inp)
    return x

def stage(inp, outp):
    x = layers.Activation('relu')(ConvBlock(inp, 128))
    x = layers.Activation('relu')(ConvBlock(x, 128))
    x = layers.Activation('relu')(ConvBlock(x, 128))
    x = layers.Activation('relu')(ConvBlock(x, 128))
    x = layers.Activation('relu')(ConvBlock(x, 128))
    x = layers.Activation('relu')(ConvBlock(x, 128, kernel_size = 1))
    x = ConvBlock(inp, outp, kernel_size = 1)
    return x

def stage1(inp, outp):
    x = layers.Activation('relu')(ConvBlock(inp, 512, kernel_size = 1))
    x = ConvBlock(x, outp, kernel_size = 1)
    return x

def vgg19(inp):
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
        inp)
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)

    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='conv5_3')(x)

    return x


def create_model_cpm():
    Input = layers.Input(shape=(224, 224, 3), dtype=tf.float32)  # 224
    outc = 21
    features = vgg19(Input)
    ############################### 1
    x1 = stage1(features)
    ############################### 2
    x2 = stage(tf.concat([features, x1], 1), outc)
    ############################### 3
    x3 = stage(tf.concat([features, x2], 1), outc)
    ############################### 4
    x4 = stage(tf.concat([features, x3], 1), outc)
    ############################### 5
    x5 = stage(tf.concat([features, x4], 1), outc)
    ############################### 6
    x6 = stage(tf.concat([features, x5], 1), outc)
    ###############################
    x = tf.stack([x1, x2, x3, x4, x5, x6], axis=1)
    # x = tf.cast(x, tf.float32)
    model = tf.keras.Model(Input, x)
    model.compile(optimizer=keras.optimizers.SGD(), loss=rmse, metrics=['accuracy'])
    return model

def rmse(x, y):
    x = tf.math.sqrt(tf.keras.losses.MSE(x, y))
    return x
'''policy = tf.keras.mixed_precision.experimental.Policy('float32')
tf.keras.mixed_precision.experimental.set_policy(policy)'''

