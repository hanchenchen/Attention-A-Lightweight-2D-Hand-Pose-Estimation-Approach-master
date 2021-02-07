import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from clr import *
from attention_blur import *
import os
import argparse
from frei_hand import load_training_dataset, load_dataset
import json

# Model's architecture

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
    x = layers.Activation('Mish')(x)
    return x


def ARB(inp, fout, dk, dv, nh, kernel, aug=True):
    x = conv(inp, kernel=1, filt=fout * 4, pad='same')
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.Activation('Mish')(x)
    x = layers.DepthwiseConv2D(kernel_size=kernel, strides=1, padding='same')(x)
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.Activation('Mish')(x)
    if aug == True:
        a = aug_block(x, fout * 4, dk, dv, nh, kernel)
        x = layers.Add()([a, x])
    x = conv(x, kernel=1, filt=fout, pad='same')
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    x = layers.Activation('Mish')(x)
    return x


def transition(inp, filters):
    x = conv(inp, kernel=1, filt=filters, pad='same')
    x = BlurPool2D()(x)
    x = layers.BatchNormalization(axis=-1, fused=True)(x)
    return x


def dense(x, kernel, num, nh=4, filters=10, aug=True):
    x_list = [x]
    for i in range(num):
        x = ARB(x, filters, 0.1, 0.1, nh, kernel, aug)
        x_list.append(x)
        x = tf.concat(x_list, axis=-1)
    return x


def create_model():
    Input = layers.Input(shape=(224, 224, 3), dtype='bfloat16')  # 224
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
    x = tf.cast(x, tf.float32)
    model = tf.keras.Model(Input, x)
    model.compile(optimizer=keras.optimizers.SGD(), loss=rmse, metrics=['accuracy'])
    return model

def rmse(x, y):
    x = tf.math.sqrt(tf.keras.losses.MSE(x, y))
    return x
parser = argparse.ArgumentParser(description='use the specified dataset to train the model.')
parser.add_argument('dataset_name', type=str, default='FreiHAND_pub_v2',
                    help='choose one dataset.')
args = parser.parse_args()
choosed = ['FreiHAND_pub_v2', 'Panoptic']


configs = json.load(open('configs/' + args.dataset_name + '.json'))


########## Dataset ##############
print('Generating datasets...')
traning_dataset = load_training_dataset('training')
validation_dataset = load_training_dataset('validation')
tf.print(traning_dataset, validation_dataset)
print('Generate the model')
########### HYPERPARAMETERS ###########
BATCH_SIZE = configs['batch_size']
step_factor = configs['step_factor']
EPOCHS = configs['epochs']
steps_per_epoch = int(configs['size']*0.8 // BATCH_SIZE)
val_steps = int(configs['size']*0.1 // BATCH_SIZE)
step_size = steps_per_epoch * step_factor
print(steps_per_epoch, val_steps)
# Your model's name
print('The model will be saved in directory:', args.dataset_name)

########### print Learning Rate ###########
lr_print = showLR()

############ Cyclical Learning Rate ###############
clr_triangular = CyclicLR(base_lr=configs['learning_rate'], max_lr=0.1, step_size=step_size, mode='triangular2')

######## Tensorboard ############
# Your log directory
import datetime
logdir = args.dataset_name + "/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)

######## Model Save #############
# Your saving directory
filepath = args.dataset_name + '/weights.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1,
                                             save_weights_only=True)

##########Callback###############
# clbk = [tensorboard_callback, checkpoint, lr_print, clr_triangular]
clbk = [checkpoint, lr_print, clr_triangular]

###############Fit#############
policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
tf.keras.mixed_precision.experimental.set_policy(policy)
model = create_model()
model.load_weights(filepath)  # continue to train
# model.summary() # architecture
history = model.fit(traning_dataset, validation_data=validation_dataset, initial_epoch=0, steps_per_epoch=steps_per_epoch, validation_steps=val_steps, epochs=EPOCHS, verbose=1, callbacks=clbk)
history.history # print history of training
# model.summary()
model.save_weights(filepath)
