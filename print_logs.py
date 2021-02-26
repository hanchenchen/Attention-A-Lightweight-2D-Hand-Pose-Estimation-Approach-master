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
parser.add_argument('--arch', type=str, default='1',
                    help='ablation studies. ')
args = parser.parse_args()
dire = args.dataset_name + '/'+ ('cpm' if args.arch == 'cpm' else 'arch' + args.arch)
logs = json.load(open(dire + '/logs.json'))
epoch = [i+1 for i in range(len(logs))]
loss = [0 for i in range(len(logs))]
val_loss = [0 for i in range(len(logs))]
pck = [0 for i in range(len(logs))]
for i in range(len(logs)):
    print(logs[i])
    loss[i] = logs[i]['logs']['loss']
    val_loss[i] = logs[i]['logs']['val_loss']
    pck[i] = logs[i]['pck_results']['sigma_pck']['0.2']

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
ax.plot(epoch,
        loss,
        Markers[idx//6]+Colors[idx%5], linewidth=1)
idx += 1
ax.plot(epoch,
        val_loss,
        Markers[idx//6]+Colors[idx%5], linewidth=1)
idx += 1
plt.xlabel('Distance threshold / px', fontsize=12)
plt.ylabel('Fraction of frames within distance / %', fontsize=12)
plt.xlim([0.0, epoch[-1]])
plt.ylim([0.0, 1.0])
ax.grid(True)

# save if required
fig.savefig(dire + '/loss_logs.png',
            bbox_extra_artists=None,
            bbox_inches='tight')
plt.show(block=False)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(epoch,
        pck,
        Markers[idx//6]+Colors[idx%5], linewidth=1)
idx += 1
plt.xlabel('Distance threshold / px', fontsize=12)
plt.ylabel('Fraction of frames within distance / %', fontsize=12)
plt.xlim([0.0, epoch[-1]])
plt.ylim([0.0, 1.0])
ax.grid(True)

# save if required
fig.savefig(dire + '/pck_logs.png',
            bbox_extra_artists=None,
            bbox_inches='tight')
plt.show(block=False)