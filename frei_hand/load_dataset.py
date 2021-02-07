import tensorflow as tf
import os
import json

AUTO = tf.data.experimental.AUTOTUNE
configs = json.load(open('configs/FreiHAND_pub_v2.json'))
print('Reading labels...')
BATCH_SIZE = configs['batch_size']
K_path = open(configs['K_path'], 'r')
xyz_path = open(configs['xyz_path'], 'r')
K_list = tf.convert_to_tensor(json.load(K_path))
xyz_list = tf.convert_to_tensor(json.load(xyz_path))
K_path.close()
xyz_path.close()
def read_label(filename):
    idx = tf.strings.to_number(tf.strings.substr(filename, -12, 8), out_type=tf.int32)
    K, xyz = K_list[idx], xyz_list[idx]
    uv = tf.transpose(tf.matmul(K, tf.transpose(xyz)))
    label_idx = uv[:, :2] / uv[:, -1:]
    # tf.print(filename, label_idx)
    return label_idx

def read_freihand(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = augmentation(image)
    return image, read_label(filename)/224.

def augmentation(image):
    """
    Augment dataset with random brightness, saturation, contrast, and image quality. Then it is casted to bfloat16 and normalized
    """
    image = tf.cast(image, tf.uint8)
    image = tf.image.random_jpeg_quality(image,min_jpeg_quality=70,max_jpeg_quality=100)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [224, 224, 3])
    image = tf.image.random_brightness(image, max_delta=25/255)
    image = tf.image.random_saturation(image, lower=0.3, upper=1.7)
    image = tf.image.random_contrast(image, lower=0.3, upper=1.7)
    image = tf.cast(image, tf.bfloat16)
    image = tf.image.per_image_standardization(image)
    return image

def load_dataset(filenames):
    """
    Load each TFRecord
    """
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    files = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = files.with_options(ignore_order)
    # dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=512, num_parallel_calls=AUTO)
    dataset = dataset.map(map_func=read_freihand, num_parallel_calls=AUTO)
    return dataset

def get_batched_dataset(filenames):
    """
    Feeds batch to the fit function
    """
    dataset = load_dataset(filenames)
    dataset = dataset.cache()
    dataset = dataset.shuffle(2000)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset

# The same as above but without augmentation to the validation dataset

def read_freihand1(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    # image = augmentation(image)
    return image, read_label(filename)/224.

def augmentation1(image):
    image = tf.cast(image, tf.bfloat16)
    image = tf.image.per_image_standardization(image)
    return image

def load_dataset1(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    files = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = files.with_options(ignore_order)
    # dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=512, num_parallel_calls=AUTO)
    dataset = dataset.map(map_func=read_freihand1, num_parallel_calls=AUTO)
    return dataset

def get_batched_dataset1(filenames):
    dataset = load_dataset1(filenames)
    dataset = dataset.cache()
    dataset = dataset.shuffle(2000)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_training_dataset(filenames):
    return get_batched_dataset1(filenames)

def get_validation_dataset(filenames):
    return get_batched_dataset1(filenames)

def load_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = augmentation1(image)
    return image

def get_testing_dataset(filenames):
    images = [load_image(filename) for filename in filenames]
    labels = [read_label(filename) for filename in filenames]
    return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)/224.

def generate_dataset(training, validation):
    return get_training_dataset(training), get_validation_dataset(validation)# , get_testing_dataset(testing)


