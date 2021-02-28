import argparse
import json
import os
from load_tfrecord import raw_images, load_dataset, load_training_dataset, load_xyz_dataset
import tensorflow as tf
from model_ablation import create_model
from model_cpm import create_model_cpm
import time
from pck import get_pck_with_sigma,get_pck_with_pixel
import numpy as np
from utils import *
start = time.time()
parser = argparse.ArgumentParser(description='Choose dataset, architesture and GPU')
parser.add_argument('dataset_name', type=str, default='FreiHAND_pub_v2',
                    help='choose one dataset(FreiHAND_pub_v2/Panoptic/HO3D_v2/SHP).')
parser.add_argument('--arch', type=str, default='1',
                    help='ablation studies. ')
parser.add_argument('--GPU', type=str, default='3',
                    help='GPU. ')
args = parser.parse_args()

configs = json.load(open('configs/' + args.dataset_name + '.json'))
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
dire = args.dataset_name + '/' + ('cpm' if args.arch == 'cpm' else 'arch' + args.arch)
dire = dire + '/best_loss'
if not os.path.exists(dire):
    os.makedirs(dire)
filepath = dire + '.hdf5'
print('Evaluating ...', filepath)
if args.arch == 'cpm':
    model = create_model_cpm()
else:
    model = create_model(int(args.arch))
model.load_weights(filepath)
predictions = {}
ground_truth = {}
names, images, labels = load_xyz_dataset(args.dataset_name,-1, 'testing')

results = model.predict(images.take(6000), batch_size = 1, steps = 6000, verbose = 1) # .take(10)) # the number of samples (batch, 28, 28, 21, 6)

if args.arch == 'cpm':
    results = (get2DKpsFromHeatmap(results[:, :, :, :, -1])*8.).tolist()
    print(type(results))
else:
    results = (results*224).tolist()
    print(type(results))
names = [''.join(str(j) for j in i)[2:-1] for i in list(names.as_numpy_iterator())]
labels = [(i[0]*224).tolist() for i in list(labels.as_numpy_iterator())]
print('results:', len(results), len(names), len(labels))
print(names[0],results[0],labels[0])
from tqdm import tqdm
for i in tqdm(range(len(results))):
    predictions[names[i]] = {'prd_label': results[i], 'resol': 224}
    ground_truth[names[i]] = labels[i]
    # print(predictions[names[i]])
if not os.path.exists(dire + '/quantitative_results'):
    os.makedirs(dire + '/quantitative_results')
if not os.path.exists(dire + '/qualitative_results'):
    os.makedirs(dire + '/qualitative_results')
json.dump(predictions, open(dire + '/quantitative_results/predictions.json', 'w'))
json.dump(ground_truth, open(dire + '/quantitative_results/ground_truth.json', 'w'))
pck_results_pixel = get_pck_with_pixel(predictions, ground_truth, save_path = dire + '/quantitative_results/results_pixel.png')
print('pck_results_pixel["AUC"]:', pck_results_pixel['AUC'])
json.dump(pck_results_pixel, open(dire + '/quantitative_results/pck_results_pixel.json', 'w'))
pck_results_sigma = get_pck_with_sigma(predictions, ground_truth, save_path = dire + '/quantitative_results/results_sigma.png')
print('pck_results_sigma["AUC"]:',pck_results_sigma["AUC"])
json.dump(pck_results_sigma, open(dire + '/quantitative_results/pck_results_sigma.json', 'w'))
test_image = raw_images(args.dataset_name, 'testing')
for i in range(10): # len(results)):
    name = names[i]
    pil_img = test_image[i]
    print('test:', name)
    show_hand(pil_img.copy(), ground_truth[name], dire + '/qualitative_results/gt_' + name)
    show_hand(pil_img, predictions[name]['prd_label'],
              dire + '/qualitative_results/pred_' + name)
end = time.time()
print('predicted done in',end - start, 'sec.')