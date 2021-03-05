import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from clr import *
from attention_blur import *
import os
import argparse
from load_tfrecord import load_training_dataset, load_xyz_dataset, load_cpm_dataset, raw_images
import json
from model_ablation import create_model
from model_cpm import create_model_cpm
from pck import get_pck_with_sigma,get_pck_with_pixel
import numpy as np
from utils import *

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
# tf.config.run_functions_eagerly(True)
########## Dataset ##############
print('Generating datasets...')
dire = 'cpm' if args.arch == 'cpm' else 'arch' + args.arch
if dire == 'cpm':
    training_dataset = load_cpm_dataset(args.dataset_name, 'training')
    validation_dataset = load_cpm_dataset(args.dataset_name, 'validation')
else:
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

print('The model will be saved in directory:', dire)

########### print Learning Rate ###########
lr_print = showLR()


######## Tensorboard ############
# Your log directory
import datetime
dir_path = args.dataset_name + '/' + dire
logdir = dir_path + "/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False)

######## Model Save #############
# Your saving directory
filepath = dir_path + '/weights.02-0.07653.hdf5' # !! last weights
checkpoint = keras.callbacks.ModelCheckpoint(dir_path + '/weights.{epoch:02d}-{val_loss:.5f}.hdf5', monitor='val_loss', save_best_only=False, verbose=1,
                                             save_weights_only=True)

best_pck = 0
##########Callback###############
from tqdm import tqdm
class get_pck(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        predictions = {}
        ground_truth = {}
        num = min(6000, int(configs['size']*0.1)) # if args.arch == 'cpm' else -1 # cpu memory
        names, images, labels = load_xyz_dataset(args.dataset_name, num,  'validation')

        results = self.model.predict(images, steps = num, verbose = 1)  # .take(10)) # the number of samples (batch, 28, 28, 21, 6)
        names = [''.join(str(j) for j in i)[2:-1] for i in list(names.as_numpy_iterator())]
        if args.arch == 'cpm':
            results = (get2DKpsFromHeatmap(results[:, :, :, :, -1]) * 8.).tolist()
        else:
            results = (results * 224).tolist()
        labels = [(i[0] * 224).tolist() for i in list(labels.as_numpy_iterator())]
        # print(names[0],results[0],labels[0])
        for i in tqdm(range(len(results))):
            predictions[names[i]] = {'prd_label': results[i], 'resol': 224}
            ground_truth[names[i]] = labels[i]
        pck_results = get_pck_with_sigma(predictions, ground_truth, [0.05, 0.1, 0.15, 0.2])
        print("End epoch ", epoch, " of training, ", pck_results)
        global best_pck
        if pck_results['sigma_pck']['0.2'] >= best_pck:
            best_pck = pck_results['sigma_pck']['0.2']
            model.save_weights(dir_path + '/pck.'+str(epoch)+'-'+str(best_pck)+'.hdf5')
            print('update best_pck_weights, pck0.2 = ', str(best_pck))
        json_logs = []
        if os.path.exists(dir_path+ '/logs.json'):
            json_logs = json.load(open(dir_path+ '/logs.json'))
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        json_logs.append({'epoch': epoch, 'lr': lr, 'logs': logs, 'pck_results': pck_results})
        # print(json_logs)
        json.dump(json_logs, open(dir_path+ '/logs.json', 'w'))
        '''test_image = raw_images(args.dataset_name, 'validation')
        for i in range(1):  # len(results)):
            name = names[i]
            pil_img = test_image[i]
            print('test:', name)
            show_hand(pil_img.copy(), ground_truth[name], dire + '/qualitative_results/gt_' + name)
            show_hand(pil_img, predictions[name]['prd_label'],
                      dire + '/logs/images/epoch_'+str(epoch) +'_'+ name)'''
############ load model ###############
lr = configs['learning_rate']
initial_epoch = 0
if dire == 'cpm':
    model = create_model_cpm()
else:
    model = create_model(int(args.arch))
if os.path.exists(filepath):
    print('continue to train...')
    print(filepath)
    model.load_weights(filepath)  # continue to train
    if os.path.exists(dir_path + '/logs.json'):
        json_logs = json.load(open(dir_path+ '/logs.json'))
        lr = json_logs[-1]['lr']
        initial_epoch = len(json_logs)
        for i in range(initial_epoch):
            best_pck = max(best_pck, json_logs[i]['pck_results']['sigma_pck']['0.2'])
    print('initial_epoch:', initial_epoch)
    print('best_pck:', best_pck)


############ Cyclical Learning Rate ###############
clr_triangular = CyclicLR(base_lr=lr, max_lr=0.1, step_size=step_size, mode='triangular2')

clbk = [tensorboard_callback, checkpoint, lr_print, clr_triangular, get_pck()]
# clbk = [checkpoint, lr_print, clr_triangular]

###############Fit#############
'''policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
tf.keras.mixed_precision.experimental.set_policy(policy)'''

# model.summary() # architecture
history = model.fit(training_dataset, validation_data=validation_dataset, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=val_steps, epochs=EPOCHS, verbose=1, callbacks=clbk)
history.history # print history of training
# model.summary()
# model.save_weights(filepath)
