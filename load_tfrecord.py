import tensorflow as tf
import numpy as np
import json
import time
import matplotlib.pyplot as plt

AUTO = tf.data.experimental.AUTOTUNE

def _parse_image_function(example_proto):
  # Parse the input tf.train.Example proto using the dictionary above.
  image_feature_description = {
      'name': tf.io.FixedLenFeature([], tf.string),
      'image': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.string)
  }
  return tf.io.parse_single_example(example_proto, image_feature_description)

def get_name(sample):
    return sample['name']

def get_image(sample, image_type = 'png'):
    if image_type == 'jpg':
        image = tf.io.decode_jpeg(sample['image'], channels=3)
    elif image_type == 'png':
        image = tf.io.decode_png(sample['image'], channels=3)
    else:
        print('Unrecognized type:', image_type)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [224, 224, 3])
    image = tf.image.per_image_standardization(image)
    return image

def get_label(sample):
    label = tf.io.parse_tensor(sample['label'], tf.float32)
    label = tf.reshape(label, [21, 2])
    return label/224.

def image_label(sample):

    return get_image(sample), get_label(sample)

def name_image_label(sample):
    return get_name(sample), get_image(sample), get_label(sample)

def load_training_dataset(dataset_name, name = 'trainig'):
    raw_image_dataset = tf.data.TFRecordDataset(dataset_name + '/' + name + '.tfrecords')
    # Create a dictionary describing the features.
    dataset = raw_image_dataset.map(map_func=_parse_image_function, num_parallel_calls=AUTO)
    dataset = dataset.map(map_func=image_label, num_parallel_calls=AUTO)
    # dataset = dataset.cache()
    configs = json.load(open('configs/' + dataset_name + '.json'))
    BATCH_SIZE = configs['batch_size']
    dataset = dataset.shuffle(BATCH_SIZE*10)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset

# finite and ordered dataset
def load_dataset(dataset_name, name = 'testing'):
    raw_image_dataset = tf.data.TFRecordDataset(dataset_name + '/' + name + '.tfrecords')
    # Create a dictionary describing the features.
    dataset = raw_image_dataset.map(map_func=_parse_image_function, num_parallel_calls=AUTO)
    # dataset = dataset.map(map_func=name_image_label, num_parallel_calls=AUTO)
    return dataset


def add_batch(dataset):
    dataset = dataset.batch(1, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset
def load_xyz_dataset(dataset_name, name = 'testing'):
    name_dataset = tf.data.TFRecordDataset(dataset_name + '/' + name + '.tfrecords')
    image_dataset = tf.data.TFRecordDataset(dataset_name + '/' + name + '.tfrecords')
    label_dataset = tf.data.TFRecordDataset(dataset_name + '/' + name + '.tfrecords')
    # Create a dictionary describing the features.
    name_dataset = name_dataset.map(map_func=_parse_image_function, num_parallel_calls=AUTO)
    image_dataset = image_dataset.map(map_func=_parse_image_function, num_parallel_calls=AUTO)
    label_dataset = label_dataset.map(map_func=_parse_image_function, num_parallel_calls=AUTO)
    name_dataset = name_dataset.map(map_func=get_name, num_parallel_calls=AUTO)
    image_dataset = image_dataset.map(map_func=get_image, num_parallel_calls=AUTO)
    label_dataset = label_dataset.map(map_func=get_label, num_parallel_calls=AUTO)
    return add_batch(name_dataset), add_batch(image_dataset), add_batch(label_dataset) #batch!!!??? 无法理解为什么不加batch就不行

from PIL import Image, ImageDraw

def hand_pose_estimation(im, coordinates, name):
    # Type: list,   Length:21,      element:[x,y]
    save_path = name+'.jpg'
    ori_im = draw_point(coordinates, im)
    print('save output to ', save_path)
    ori_im.save(save_path)
    return coordinates


def draw_point(points, im):
    i = 0
    draw = ImageDraw.Draw(im)

    for point in points:
        x = point[0]
        y = point[1]

        if i == 0:
            rootx = x
            rooty = y
        if i == 1 or i == 5 or i == 9 or i == 13 or i == 17:
            prex = rootx
            prey = rooty

        if i > 0 and i <= 4:
            draw.line((prex, prey, x, y), 'red')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'red', 'white')
        if i > 4 and i <= 8:
            draw.line((prex, prey, x, y), 'yellow')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'yellow', 'white')

        if i > 8 and i <= 12:
            draw.line((prex, prey, x, y), 'green')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'green', 'white')
        if i > 12 and i <= 16:
            draw.line((prex, prey, x, y), 'blue')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'blue', 'white')
        if i > 16 and i <= 20:
            draw.line((prex, prey, x, y), 'purple')
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), 'purple', 'white')

        prex = x
        prey = y
        i = i + 1
    return im

def show_samples():
    testing = load_dataset('SHP','testing')
    i = 50
    for sample in testing:
        if not i:
            break
        i -= 1
        print('!!!!!')
        name = sample['name']
        image = tf.image.decode_png(sample['image']).numpy()
        print(type(image))
        label = tf.io.parse_tensor(sample['label'], tf.float32)
        tf.print(name, label)
        pil_img = Image.fromarray(image)
        hand_pose_estimation(pil_img, label, str(i))
        pil_img.show()
#  '''
# '''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
show_samples() # 注释掉load_dataset中的第二个map
# '''
