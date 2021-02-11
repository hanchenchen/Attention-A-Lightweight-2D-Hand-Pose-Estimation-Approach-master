import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from clr import *
from attention_blur import *
import os
import argparse
from load_tfrecord import load_training_dataset, load_dataset
import json
from model import create_model



parser = argparse.ArgumentParser(description='use the specified dataset to train the model.')
parser.add_argument('dataset_name', type=str, default='FreiHAND_pub_v2',
                    help='choose one dataset.')
args = parser.parse_args()
choosed = ['FreiHAND_pub_v2', 'Panoptic']

configs = json.load(open('configs/' + args.dataset_name + '.json'))
os.environ['CUDA_VISIBLE_DEVICES'] = configs['GPU']
########## Dataset ##############
print('Generating datasets...')
training_dataset = load_training_dataset( args.dataset_name, 'training')
validation_dataset = load_training_dataset(args.dataset_name, 'validation')

print('Generate the model')
########### HYPERPARAMETERS ###########
BATCH_SIZE = configs['batch_size']
step_factor = configs['step_factor']
EPOCHS = configs['epochs']
steps_per_epoch = min(int(configs['size']*0.8 // BATCH_SIZE),2000)
val_steps = min(int(configs['size']*0.1 // BATCH_SIZE),200)
step_size = steps_per_epoch * step_factor
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
'''policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
tf.keras.mixed_precision.experimental.set_policy(policy)'''
model = create_model()
if configs['continue']:
    model.load_weights(filepath)  # continue to train
# model.summary() # architecture
history = model.fit(training_dataset, validation_data=validation_dataset, initial_epoch=0, steps_per_epoch=steps_per_epoch, validation_steps=val_steps, epochs=EPOCHS, verbose=1, callbacks=clbk)
history.history # print history of training
# model.summary()
model.save_weights(filepath)
