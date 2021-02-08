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

def image_label(sample):
    image = tf.image.decode_jpeg(sample['image'])
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [224, 224, 3])
    image = tf.image.per_image_standardization(image)
    label = tf.io.parse_tensor(sample['label'], tf.float32)
    label = tf.reshape(label, [21, 2])
    return image, label/224.

def name_image_label(sample):
    image = tf.image.decode_jpeg(sample['image'])
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [224, 224, 3])
    image = tf.image.per_image_standardization(image)
    label = tf.io.parse_tensor(sample['label'], tf.float32)
    label = tf.reshape(label, [21, 2])
    return sample['name'], image, label/224.

def load_training_dataset(dataset, name = 'trainig'):
    raw_image_dataset = tf.data.TFRecordDataset(dataset + '/' + name + '.tfrecords')
    # Create a dictionary describing the features.
    dataset = raw_image_dataset.map(map_func=_parse_image_function, num_parallel_calls=AUTO)
    dataset = dataset.map(map_func=image_label, num_parallel_calls=AUTO)
    # dataset = dataset.cache()
    configs = json.load(open('configs/' + name + '.json'))
    BATCH_SIZE = configs['batch_size']
    dataset = dataset.shuffle(BATCH_SIZE*10)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset

# finite and ordered dataset
def load_dataset(dataset, name = 'testing'):
    raw_image_dataset = tf.data.TFRecordDataset(dataset + '/' + name + '.tfrecords')
    # Create a dictionary describing the features.
    dataset = raw_image_dataset.map(map_func=_parse_image_function, num_parallel_calls=AUTO)
    dataset = dataset.map(map_func=name_image_label, num_parallel_calls=AUTO)
    return dataset

'''
from PIL import Image, ImageDraw

def hand_pose_estimation(im, coordinates, name):
    # Type: list,   Length:21,      element:[x,y]
    save_path = 'Panoptic/'+name+'.jpg'
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

testing = load_dataset('Panoptic','testing')
i = 50
for sample in testing:
    if not i:
        break
    i -= 1
    print('!!!!!')
    name = sample['name']
    image = tf.image.decode_jpeg(sample['image']).numpy()
    print(type(image))
    label = tf.io.parse_tensor(sample['label'], tf.float32)
    tf.print(name, label)
    pil_img = Image.fromarray(image)
    hand_pose_estimation(pil_img, label, str(i))
    pil_img.show()
'''



