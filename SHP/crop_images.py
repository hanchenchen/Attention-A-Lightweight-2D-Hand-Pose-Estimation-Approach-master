import tensorflow as tf
from PIL import Image, ImageDraw
import json
import time
import os
import random
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
start_time = time.time()
configs = json.load(open('configs/SHP.json'))
print('Reading labels...')
print('Reading images...')
images_path = tf.io.gfile.glob(configs['origin_images_path'])
print('len(images_path)', len(images_path))

import scipy.io as scio
labels_path = tf.io.gfile.glob(configs['labels_dir'])
labels = {}
for path in labels_path:
    labels[path.split('/')[-1]] = scio.loadmat(path)
'''print(labels['B5Random_BB.mat']['handPara'][0][0])
print(len(labels['B5Random_BB.mat']['handPara'][0][0]))
print(len(labels['B5Random_BB.mat']['handPara'][0]))
print(len(labels['B5Random_BB.mat']['handPara']))'''
def get_label(path):
    dir = path.split('/')[-2] + '_BB.mat'
    idx = path.split('/')[-1].split('.')[-2].split('_')
    is_left = 1 if idx[-2] == 'left' else 0
    idx = int(idx[-1])
    label = []
    for i in range(21):
        label.append([labels[dir]['handPara'][0][i][idx],labels[dir]['handPara'][1][i][idx],labels[dir]['handPara'][2][i][idx]])
    # print(label)
    # 仅相差个平移参数baseline，旋转忽略 Reference: https://www.cnblogs.com/wjy-lulu/p/12857249.html#top
    fx = 822.79041
    fy = 822.79041
    tx = 318.47345
    ty = 250.31296
    base = 120.054
    # 增广矩阵计算方便
    R_l = np.asarray([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]])
    R_r = R_l.copy()
    R_r[0, 3] = -base  # 作为平移参数
    # 内参矩阵
    K = np.asarray([
        [fx, 0, tx],
        [0, fy, ty],
        [0, 0, 1]])
    # 世界坐标系点，4*21矩阵，[x,y,z,1]增广矩阵，计算方便
    points =np.concatenate((np.array(label), np.ones((21, 1))), axis = 1).T
    if is_left:
        # 平移+内参
        left_point = np.dot(np.dot(K, R_l), points)
        # 消除尺度z
        image_cood = left_point / left_point[-1, ...]
        image_left = (image_cood[:2, ...].T).astype(np.uint)
        # print(image_left)
        return image_left
    else:
        # 平移+内参
        right_point = np.dot(np.dot(K, R_r), points)
        # 消除尺度z
        image_cood = right_point / right_point[-1, ...]
        image_right = (image_cood[:2, ...].T).astype(np.uint)
        return image_right

def hand_pose_estimation(im, coordinates, name):
    # Type: list,   Length:21,      element:[x,y]
    save_path = name+'.jpg'
    ori_im = draw_point(coordinates, im)
    print('save output to ', save_path)
    # ori_im.save(save_path)
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

def show_samples(path):
    im = Image.open(path)
    label = np.array(get_label(path))
    print(label)
    hand_pose_estimation(im, label, path.split('/')[-1])
    im.show()


def box(im, filename, size):
    label = get_label(filename).tolist()
    if im.size[1] <= 500:
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
    json.dump(label, open('/HDD/ningbo/fileshare/SHP/cropped/' + path.split('/')[-2] + '_' + path.split('/')[-1][:-4]+'.json','w'))
    return (left, top, left + w, top + w)

i = 0
for path in images_path:
    im = Image.open(path)
    region = im.crop(box(im, path, im.size))
    i += 1
    if not i % 500:
        pass
        # region.show()
        # show_samples(path)
    region = region.resize((224, 224))
    print(str(i)+'/'+str(len(images_path)) + 'croped image',region.size,  path.split('/')[-2] + '_' + path.split('/')[-1])
    region.save('/HDD/ningbo/fileshare/SHP/cropped/' + path.split('/')[-2] + '_' + path.split('/')[-1])


