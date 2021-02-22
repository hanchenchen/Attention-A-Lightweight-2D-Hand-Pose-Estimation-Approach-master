import tensorflow as tf
import numpy as np
import json
import time
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

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
    idx = tf.strings.to_number(tf.strings.substr(filename, -12, 8), out_type=tf.int32)
    idx = idx % 32560
    K, xyz = K_list[idx], xyz_list[idx]
    uv = tf.transpose(tf.matmul(K, tf.transpose(xyz)))
    label_idx = uv[:, :2] / uv[:, -1:]
    return label_idx
start_time = time.time()
configs = json.load(open('configs/FreiHAND_pub_v2.json'))
print('Reading images...')
images_path = tf.io.gfile.glob(configs['images_path'])
num = len(images_path)
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = configs['GPU']
random.shuffle(images_path)
dataset = {
    'training' : images_path[:int(num*0.8)],
    'validation' : images_path[int(num*0.8):-int(num*0.1)],
    'testing' : images_path[-int(num*0.1):]
}


print('Reading labels...')
BATCH_SIZE = configs['batch_size']
K_list = tf.convert_to_tensor(json.load(open(configs['K_path'], 'r')))
xyz_list = tf.convert_to_tensor(json.load(open(configs['xyz_path'], 'r')))
for name, paths in dataset.items():
    record_file = 'FreiHAND_pub_v2/'+ name +'.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
        for path in paths:
            print(path)
            image_string = open(path, 'rb').read()
            label_string = tf.io.serialize_tensor(read_label(path))
            path = path[-12:].encode('utf-8')
            tf_example = serialize_example(path, image_string, label_string)
            writer.write(tf_example)
end_time = time.time()
print("Takes", end_time - start_time, "sec.")