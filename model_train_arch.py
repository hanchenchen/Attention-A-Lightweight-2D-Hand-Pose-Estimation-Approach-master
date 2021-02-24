import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from clr import *
from attention_blur import *
import os
import argparse
from load_tfrecord import load_training_dataset, load_dataset
import json
from model_ablation import create_model
from model_cpm import create_model_cpm

parser = argparse.ArgumentParser(description='Choose dataset, architesture and GPU')
parser.add_argument('dataset_name', type=str, default='FreiHAND_pub_v2',
                    help='choose one dataset(FreiHAND_pub_v2/Panoptic/HO3D_v2/SHP).')
parser.add_argument('--arch', type=str, default='1',
                    help='ablation studies. ')
parser.add_argument('--GPU', type=str, default=0,
                    help='GPU. ')
args = parser.parse_args()

configs = json.load(open('configs/' + args.dataset_name + '.json'))
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

########## Dataset ##############
print('Generating datasets...')
training_dataset = load_training_dataset( args.dataset_name, 'training')
validation_dataset = load_training_dataset(args.dataset_name, 'validation')

print('Generate the model_arch:', args.arch)
########### HYPERPARAMETERS ###########
BATCH_SIZE = configs['batch_size']
step_factor = configs['step_factor']
EPOCHS = configs['epochs']
steps_per_epoch = int(configs['size']*0.8 // BATCH_SIZE)
val_steps = int(configs['size']*0.1 // BATCH_SIZE)
step_size = steps_per_epoch * step_factor
# Your model's name
dire = 'cpm' if args.arch == 'cpm' else 'arch' + args.arch
print('The model will be saved in directory:', dire)

########### print Learning Rate ###########
lr_print = showLR()

############ Cyclical Learning Rate ###############
clr_triangular = CyclicLR(base_lr=configs['learning_rate'], max_lr=0.1, step_size=step_size, mode='triangular2')

######## Tensorboard ############
# Your log directory
import datetime
dir_path = args.dataset_name + '/' + dire
logdir = dir_path + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)

######## Model Save #############
# Your saving directory
filepath = dir_path + '/weights.hdf5'
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1,
                                             save_weights_only=True)

##########Callback###############
clbk = [tensorboard_callback, checkpoint, lr_print, clr_triangular]
# clbk = [checkpoint, lr_print, clr_triangular]

###############Fit#############
'''policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
tf.keras.mixed_precision.experimental.set_policy(policy)'''
if dire == 'cpm':
    model = create_model_cpm()
else:
    model = create_model(int(args.arch))
if os.path.exists(filepath):
    print('continue to train...')
    model.load_weights(filepath)  # continue to train
# model.summary() # architecture
history = model.fit(training_dataset, validation_data=validation_dataset, initial_epoch=0, steps_per_epoch=steps_per_epoch, validation_steps=val_steps, epochs=EPOCHS, verbose=1, callbacks=clbk)
history.history # print history of training
# model.summary()
model.save_weights(filepath)
