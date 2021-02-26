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

parser = argparse.ArgumentParser(description='Choose dataset, architesture and GPU')
parser.add_argument('dataset_name', type=str, default='FreiHAND_pub_v2',
                    help='choose one dataset(FreiHAND_pub_v2/Panoptic/HO3D_v2/SHP).')

args = parser.parse_args()

for root, dirs, files in os.walk(args.dataset_name, topdown=False):

    for name in dirs:
        print(os.path.join(root, name))