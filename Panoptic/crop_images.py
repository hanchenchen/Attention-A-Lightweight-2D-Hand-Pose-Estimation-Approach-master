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
    labels = json.load(open(filename[:-4]+'.json'))
    label = labels['hand_pts']
    midst = label[9]
    midst[0] += random.uniform(-5.5, 5.5)
    midst[1] += random.uniform(-5.5, 5.5)
    left, top, right, bottom = int(midst[0] - 112), int(midst[1] - 112), int(midst[0] + 112), int(midst[1] + 112)
    for i in range(21):
        label[i][0] -= left
        label[i][1] -= top
    # print(label)
    json.dump(label, open('/HDD/ningbo/fileshare/Panoptic/hand_labels/croped/' + path.split('/')[-1][:-4]+'.json', 'w'))
    return (left, top, right, bottom)

i = 0
for path in images_path:
    im = Image.open(path)
    region = im.crop(box(im, path))
    i += 1
    if not i % 300:
        region.show()
    print('croped image', path.split('/')[-1])
    region.save('/HDD/ningbo/fileshare/Panoptic/hand_labels/croped/' + path.split('/')[-1])


