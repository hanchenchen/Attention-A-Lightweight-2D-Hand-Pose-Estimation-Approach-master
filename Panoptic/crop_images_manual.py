import tensorflow as tf
from PIL import Image, ImageDraw
import json
import time
import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
start_time = time.time()
configs = json.load(open('configs/Panoptic.json'))

print('Reading images...')
images_path = tf.io.gfile.glob(configs['origin_images_path1'])
print(len(images_path))
def get_label(path):
    labels = json.load(open(path[:-4] + '.json'))
    label = labels['hand_pts']
    return label
def box(im, filename, size):
    label = get_label(filename)
    if im.size[0] <= 500:
        w = min(size)
    else:
        w = 224
    midst = label[9]
    midst[0] += random.uniform(-5.5, 5.5)
    midst[1] += random.uniform(-5.5, 5.5)
    left = max(0, min(size[0]-w, int(midst[0] - w/2.)))
    top = max(0, min(size[1]-w, int(midst[1] - w/2.)))
    for i in range(21):
        label[i][0] -= left
        label[i][0] /= w/224
        label[i][1] -= top
        label[i][1] /= w/224
    # print(label)
    print(left, top, left + w, top + w)
    json.dump(label, open('/HDD/ningbo/fileshare/Panoptic/cropped/' + path.split('/')[-1][:-4]+'.json', 'w'))
    return (left, top, left + w, top + w)

i = 0
for path in images_path:
    im = Image.open(path)
    region = im.crop(box(im, path, im.size))
    i += 1
    if not i % 1000:
        region.show()
    region = region.resize((224, 224))
    print(str(i)+'/'+str(len(images_path)) + 'croped image',region.size, path.split('/')[-1])
    region.save('/HDD/ningbo/fileshare/Panoptic/cropped/' + path.split('/')[-1])


