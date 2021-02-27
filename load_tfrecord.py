import tensorflow as tf
import numpy as np
import json
import time
import matplotlib.pyplot as plt

AUTO = tf.data.experimental.AUTOTUNE
# tf.config.run_functions_eagerly(True)
image_type = 'png'
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

def get_image(sample):
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

def get_heatmap(sample):
    label = tf.io.parse_tensor(sample['label'], tf.float32)
    label = tf.reshape(label, [21, 2])
    label /= 8.
    label = computeHeatmaps(label, [224//8, 224//8])
    label = tf.stack([label]*6, axis = -1)
    return label

def image_label(sample):

    return get_image(sample), get_label(sample)
def image_heatmep(sample):
    return get_image(sample), get_heatmap(sample)
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
    global image_type
    image_type = configs['images_path'].split('.')[-1]
    dataset = dataset.shuffle(BATCH_SIZE*10)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset

def load_cpm_dataset(dataset_name, name = 'trainig'):
    raw_image_dataset = tf.data.TFRecordDataset(dataset_name + '/' + name + '.tfrecords')
    # Create a dictionary describing the features.
    dataset = raw_image_dataset.map(map_func=_parse_image_function, num_parallel_calls=AUTO)
    dataset = dataset.map(map_func=image_heatmep, num_parallel_calls=AUTO)
    # dataset = dataset.cache()
    configs = json.load(open('configs/' + dataset_name + '.json'))
    BATCH_SIZE = configs['batch_size']
    global image_type
    image_type = configs['images_path'].split('.')[-1]
    dataset = dataset.shuffle(BATCH_SIZE*10)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset
# finite and ordered dataset
def load_dataset(dataset_name, name = 'testing'):
    configs = json.load(open('configs/' + dataset_name + '.json'))
    global image_type
    image_type = configs['images_path'].split('.')[-1]
    raw_image_dataset = tf.data.TFRecordDataset(dataset_name + '/' + name + '.tfrecords')
    # Create a dictionary describing the features.
    dataset = raw_image_dataset.map(map_func=_parse_image_function, num_parallel_calls=AUTO)
    dataset = dataset.map(map_func=name_image_label, num_parallel_calls=AUTO)
    return dataset


def add_batch(dataset):
    dataset = dataset.batch(1, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset
def load_xyz_dataset(dataset_name, num, name = 'testing'):
    configs = json.load(open('configs/' + dataset_name + '.json'))
    global image_type
    image_type = configs['images_path'].split('.')[-1]
    sample_dataset = tf.data.TFRecordDataset(dataset_name + '/' + name + '.tfrecords')
    if num > 0:
        sample_dataset = sample_dataset.take(num)
    sample_dataset = sample_dataset.shuffle(7)
    sample_dataset = sample_dataset.map(map_func=_parse_image_function, num_parallel_calls=AUTO)

    name_dataset = sample_dataset.map(map_func=get_name, num_parallel_calls=AUTO)
    image_dataset = sample_dataset.map(map_func=get_image, num_parallel_calls=AUTO)
    label_dataset = sample_dataset.map(map_func=get_label, num_parallel_calls=AUTO)
    return add_batch(name_dataset), add_batch(image_dataset), add_batch(label_dataset) #batch!!!??? 无法理解为什么不加batch就不行
# '''
import tensorflow_probability as tfp
tfd = tfp.distributions
# '''
import numpy as np
def getOneGaussianHeatmap(inputs):
    grid = tf.cast(inputs[0], tf.float32)
    mean = tf.cast(inputs[1], tf.float32)
    std = tf.cast(inputs[2], tf.float32)
    # assert std.shape == (1,)
    assert len(grid.shape) == 2
    assert grid.shape[-1] == 2

    mvn = tfd.MultivariateNormalDiag(
        loc=mean,
        scale_identity_multiplier=std)
    prob = mvn.prob(grid) * 2 * np.pi * std * std

    return prob

def computeHeatmaps(kps2D, patchSize, std=5.):
    '''
    gets the gaussian heat map for the keypoints
    :param kps2d:Nx2 tensor
    :param patchSize: hxw
    :param std: standard dev. for the gaussain
    :return:Nxhxw heatmap hxwxN
    '''
    X, Y = tf.meshgrid(tf.range(patchSize[1]), tf.range(patchSize[0]))
    grid = tf.stack([X, Y], axis=2)
    grid = tf.reshape(grid, [-1, 2])
    grid_tile = tf.tile(tf.expand_dims(grid, 0), [kps2D.shape[0], 1, 1])
    heatmaps = tf.map_fn(getOneGaussianHeatmap, (grid_tile, kps2D[:, :2], tf.zeros(kps2D.shape[0], 1) + std), dtype=tf.float32)
    heatmaps = tf.reshape(heatmaps, [kps2D.shape[0], X.shape[0], X.shape[1]])
    heatmap_list = []
    for i in range(heatmaps.shape[0]):
        heatmap_list.append(heatmaps[i])
    heatmaps = tf.stack(heatmap_list, axis = -1)
    return heatmaps
from PIL import Image, ImageDraw
'''heatmaps = computeHeatmaps(tf.fill([1,2], 40/8.), [80/8.,80/8.])
print(heatmaps.shape)
for i in range(heatmaps.shape[-1]):
    print(heatmaps[:,:, i].shape)
    print(heatmaps[5, 5, i])
    pil_img = Image.fromarray(heatmaps[:,:,  i].numpy()*255.)
    pil_img.show()
    tf.print(heatmaps[:,:, i])'''


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

def raw_images(dataset_name, name='testing'):
    configs = json.load(open('configs/' + dataset_name + '.json'))
    global image_type
    image_type = configs['images_path'].split('.')[-1]
    samples = tf.data.TFRecordDataset(dataset_name + '/' + name + '.tfrecords')
    samples = samples.map(map_func=_parse_image_function, num_parallel_calls=AUTO)
    pil_images = []
    for sample in samples:
        if image_type == 'jpg':
            image = tf.io.decode_jpeg(sample['image'], channels=3)
        elif image_type == 'png':
            image = tf.io.decode_png(sample['image'], channels=3)
        else:
            print('Unrecognized type:', image_type)
        image = image.numpy()
        pil_images.append(Image.fromarray(image))
    return pil_images






#  '''
#
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
show_samples() # 注释掉load_dataset中的第二个map
# '''
