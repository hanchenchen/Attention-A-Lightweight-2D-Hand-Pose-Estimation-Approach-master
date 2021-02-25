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
    x1 = stage1(features, outc)
    ############################### 2
    x2 = stage(tf.concat([features, x1], -1), outc)
    ############################### 3
    x3 = stage(tf.concat([features, x2], -1), outc)
    ############################### 4
    x4 = stage(tf.concat([features, x3], -1), outc)
    ############################### 5
    x5 = stage(tf.concat([features, x4], -1), outc)
    ############################### 6
    x6 = stage(tf.concat([features, x5], -1), outc)
    ###############################
    x = tf.stack([x1, x2, x3, x4, x5, x6], axis=-1)
    # x = tf.cast(x, tf.float32)
    model = tf.keras.Model(Input, x)
    model.compile(optimizer=keras.optimizers.SGD(), loss=rmse, metrics=['accuracy'])#, run_eagerly=True)
    model.summary()
    return model
def rmse(x, y):
    # print(x.shape, y.shape)
    x = tf.math.sqrt(tf.keras.losses.MSE(x, y))
    # print(x.shape, y.shape)
    return x
'''
tf.config.run_functions_eagerly(True)
def rmse(gt, y):
    print('!!!!!')
    heatmap = []

    for i in range(y.shape[0]):
        heatmap.append(computeHeatmaps(gt[i], [224//8, 224//8]))
    heatmap = tf.stack(heatmap, axis = 0)
    heatmap = tf.stack([heatmap] * 6, axis=-1)
    #  heatmap = tf.stack([heatmap], )
    x = tf.math.sqrt(tf.keras.losses.MSE(heatmap, y))
    print(heatmap.shape, y.shape, x.shape)
    exit()
    return x

import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions

def getOneGaussianHeatmap(inputs):
    grid = tf.cast(inputs[0], tf.float32)
    mean = tf.cast(inputs[1], tf.float32)
    std = tf.cast(inputs[2], tf.float32)
    # assert std.shape == (1,)
    assert len(grid.shape) == 2
    assert grid.shape[-1] == 2

    mvn = tfd.MultivariateNormalDiag(
        loc=mean,
        scale_identity_multiplier=std)
    prob = mvn.prob(grid) * 2 * np.pi * std * std

    return prob

def computeHeatmaps(kps2D, patchSize, std=5.):

    gets the gaussian heat map for the keypoints
    :param kps2d:Nx2 tensor
    :param patchSize: hxw
    :param std: standard dev. for the gaussain
    :return:Nxhxw heatmap
    X, Y = tf.meshgrid(tf.range(patchSize[1]), tf.range(patchSize[0]))
    grid = tf.stack([X, Y], axis=2)
    grid = tf.reshape(grid, [-1, 2])
    grid_tile = tf.tile(tf.expand_dims(grid, 0), [kps2D.shape[0], 1, 1])
    heatmaps = tf.map_fn(getOneGaussianHeatmap, (grid_tile, kps2D[:, :2], tf.zeros(kps2D.shape[0], 1) + std), dtype=tf.float32)
    heatmaps = tf.reshape(heatmaps, [kps2D.shape[0], X.shape[0], X.shape[1]])
    heatmaps = tf.stack(heatmaps.numpy().tolist(), axis = -1)
    return heatmaps
''''''
from PIL import Image, ImageDraw
label = tf.fill([21,2], 1)
heatmap = computeHeatmaps(label, [224, 224])
print(heatmap)
pil_img = Image.fromarray(heatmap[1].numpy()*255)
pil_img.show()''''''
'''