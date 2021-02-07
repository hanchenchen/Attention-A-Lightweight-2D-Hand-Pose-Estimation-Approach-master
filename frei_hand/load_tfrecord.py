import tensorflow as tf
import numpy as np
import json
import time
import matplotlib.pyplot as plt

AUTO = tf.data.experimental.AUTOTUNE
configs = json.load(open('configs/FreiHAND_pub_v2.json'))
BATCH_SIZE = configs['batch_size']

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  image_feature_description = {
      'name': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.string),
      'image': tf.io.FixedLenFeature([], tf.string),
  }
  return tf.io.parse_single_example(example_proto, image_feature_description)

def load_training_dataset(name = 'trainig'):
    raw_image_dataset = tf.data.TFRecordDataset('frei_hand/tfrecord/'+name+'.tfrecords')
    # Create a dictionary describing the features.
    dataset = raw_image_dataset.map(_parse_image_function, num_parallel_calls=AUTO)
    dataset = dataset.cache()
    dataset = dataset.shuffle(BATCH_SIZE*10)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset

def load_dataset(name):
    raw_image_dataset = tf.data.TFRecordDataset('frei_hand/tfrecord/'+name+'.tfrecords')
    # Create a dictionary describing the features.
    dataset = raw_image_dataset.map(_parse_image_function, num_parallel_calls=AUTO)
    return dataset

import frei_hand
testing = load_dataset('testing')
for image, label in testing:
    frei_hand.show_result(image, label)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(image)
    tf.print(label)
    break




