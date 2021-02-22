import argparse
import json
import os
from model import create_model
from load_tfrecord import raw_images, load_dataset, load_training_dataset, load_xyz_dataset
import tensorflow as tf
from model import create_model
import time
from pck import get_pck_with_sigma,get_pck_with_pixel
import numpy as np
from utils import *
start = time.time()
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
parser = argparse.ArgumentParser(description='use the specified dataset to train the model.')
parser.add_argument('dataset_name', type=str, default='FreiHAND_pub_v2',
                    help='choose one dataset.')
args = parser.parse_args()
choosed = ['FreiHAND_pub_v2', 'Panoptic']
configs = json.load(open('configs/' + args.dataset_name + '.json'))
filepath = args.dataset_name + '/weights.hdf5'
model = create_model()
model.load_weights(filepath)
predictions = {}
ground_truth = {}
names, images, labels = load_xyz_dataset(args.dataset_name, 'testing')

images_dir = '/'.join(configs['images_path'].split('/')[:-1])
results = model.predict(images.take(10)) # the number of samples
names = [''.join(str(j) for j in i)[2:-1] for i in list(names.as_numpy_iterator())]
results = (results*224).tolist()

labels = [(i[0]*224).tolist() for i in list(labels.as_numpy_iterator())]
print('results:', len(results))
# print(names[0],results[0],labels[0])

for i in range(len(results)):
    predictions[names[i]] = {'prd_label': results[i], 'resol': 224}
    ground_truth[names[i]] = labels[i]

json.dump(predictions, open(args.dataset_name + '/quantitative_results/predictions.json', 'w'))
json.dump(ground_truth, open(args.dataset_name + '/quantitative_results/ground_truth.json', 'w'))
pck_results_pixel = get_pck_with_pixel(predictions, ground_truth, args.dataset_name + '/quantitative_results/results_pixel.png')
print('pck_results_pixel["AUC"]:', pck_results_pixel['AUC'])
json.dump(pck_results_pixel, open(args.dataset_name + '/quantitative_results/pck_results_pixel.json', 'w'))
pck_results_sigma = get_pck_with_sigma(predictions, ground_truth, args.dataset_name + '/quantitative_results/results_sigma.png')
print('pck_results_sigma["AUC"]:',pck_results_sigma["AUC"])
json.dump(pck_results_sigma, open(args.dataset_name + '/quantitative_results/pck_results_sigma.json', 'w'))
end = time.time()
print('predicted done in',end - start, 'sec.')
test_image = raw_images(args.dataset_name)
for i in range(len(results)):
    name = names[i]
    pil_img = test_image[i]
    print('test:', name)
    show_hand(pil_img.copy(), ground_truth[name], args.dataset_name + '/qualitative_results/gt_' + name)
    show_hand(pil_img, predictions[name]['prd_label'],
              args.dataset_name + '/qualitative_results/pred_' + name)
