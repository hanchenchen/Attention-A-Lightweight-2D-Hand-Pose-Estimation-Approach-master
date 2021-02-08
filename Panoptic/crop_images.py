import tensorflow as tf
from PIL import Image, ImageDraw
import json
import time
import os
import random
start_time = time.time()
configs = json.load(open('configs/Panoptic.json'))
print('Reading images...')
images_path = tf.io.gfile.glob(configs['origin_images_path'])
def box(im, filename):
    label = json.load(open(filename[:-4]+'.json'))
    midst = label['hand_pts'][9]
    midst[0] += random.uniform(-5.5, 5.5)
    midst[1] += random.uniform(-5.5, 5.5)
    return (int(midst[0] - 112), int(midst[1] - 112), int(midst[0] + 112), int(midst[1] + 112))

i = 0
for path in images_path:
    im = Image.open(path)
    region = im.crop(box(im, path))
    i += 1
    if not i % 300:
        region.show()
    print('croped image', path.split('/')[-1])
    region.save('/HDD/ningbo/fileshare/Panoptic/hand_labels/croped/' + path.split('/')[-1])


