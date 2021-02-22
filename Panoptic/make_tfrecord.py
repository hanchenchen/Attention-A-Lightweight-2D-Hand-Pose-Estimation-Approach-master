import tensorflow as tf
import numpy as np
import json
import time
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.
import os
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def serialize_example(name, image, label):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
      'name': _bytes_feature(name),
      'image': _bytes_feature(image),
      'label': _bytes_feature(label),
  }
  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def read_label(filename):
    label = json.load(open(filename[:-4]+'.json'))
    return tf.convert_to_tensor(label)[:,:2]
start_time = time.time()
configs = json.load(open('configs/Panoptic.json'))
print('Reading images...')
images_path = tf.io.gfile.glob(configs['images_path'])
num = len(images_path)
import random
os.environ['CUDA_VISIBLE_DEVICES'] = configs['GPU']
random.shuffle(images_path)
dataset = {
    'training' : images_path[:int(num*0.8)],
    'validation' : images_path[int(num*0.8):-int(num*0.1)],
    'testing' : images_path[-int(num*0.1):]
}

print('Reading labels...')
BATCH_SIZE = configs['batch_size']

for name, paths in dataset.items():
    record_file = 'Panoptic/'+ name +'.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
        for path in paths:
            print(path)
            label = read_label(path)
            image_string = open(path, 'rb').read()
            label_string = tf.io.serialize_tensor(label)
            path = path.split('/')[-1].encode('utf-8')
            tf_example = serialize_example(path, image_string, label_string)
            writer.write(tf_example)
end_time = time.time()
print("Takes", end_time - start_time, "sec ")