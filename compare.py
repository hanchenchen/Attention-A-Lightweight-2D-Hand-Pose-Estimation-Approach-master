import argparse
import json
import tensorflow as tf
import time
from pck import get_pck_with_sigma,get_pck_with_pixel
import numpy as np
from utils import *
import os
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import auc
parser = argparse.ArgumentParser(description='Choose dataset, architesture and GPU')
parser.add_argument('dataset_name', type=str, default='FreiHAND_pub_v2',
                    help='choose one dataset(FreiHAND_pub_v2/Panoptic/HO3D_v2/SHP).')

args = parser.parse_args()

results_path = tf.io.gfile.glob(args.dataset_name + '/*/' + 'pck_results_pixel.json')
for i in results_path:
    print(i)
# plot it


Markers = [
    '.' , #	point marker
    ',' , #	pixel marker
    'o' , #	circle marker
    'v' , #	triangle_down marker
    '^' , #	triangle_up marker
    '<' #	triangle_left marker
]
Colors = ['b', 'g', 'r', 'c', 'm']
idx = 0
fig = plt.figure()
ax = fig.add_subplot(111)
for path in results_path:
    results = json.load(open(path))
    interval = list(results['pixel_pck'].keys())
    pck_res = list(results['pixel_pck'].values())
    ax.plot(interval,
            pck_res,
            Markers[idx//6]+Colors[idx%5], linewidth=1)
plt.xlabel('Distance threshold / px', fontsize=12)
plt.ylabel('Fraction of frames within distance / %', fontsize=12)
plt.xlim([0.0, 40.])
plt.ylim([0.0, 1.0])
ax.grid(True)

# save if required
fig.savefig(args.dataset_name + '/results.png',
            bbox_extra_artists=None,
            bbox_inches='tight')
plt.show(block=False)