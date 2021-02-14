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
print(type(images_path))
import scipy.io as scio
labels_path = tf.io.gfile.glob(configs['labels_dir'])
labels = {}
for path in labels_path:
    labels[path.split('/')[-1]] = scio.loadmat(path)
print(labels['B5Random_BB.mat']['handPara'][0][0])
print(len(labels['B5Random_BB.mat']['handPara'][0][0]))
print(len(labels['B5Random_BB.mat']['handPara'][0]))
print(len(labels['B5Random_BB.mat']['handPara']))
label = []
for i in range(21):
    label.append([labels['B5Random_BB.mat']['handPara'][0][i][0],labels['B5Random_BB.mat']['handPara'][1][i][0],labels['B5Random_BB.mat']['handPara'][2][i][0]])
    print(labels['B5Random_BB.mat']['handPara'][0][i][0],labels['B5Random_BB.mat']['handPara'][1][i][0],labels['B5Random_BB.mat']['handPara'][2][i][0])


def cv2ProjectPoints(pts3D, isOpenGLCoords=True):
    '''
    TF function for projecting 3d points to 2d using CV2
    :param camProp:  (1) Point Grey Bumblebee2 stereo camera: base line = 120.054 fx = 822.79041 fy = 822.79041 tx = 318.47345 ty = 250.31296
    :param pts3D:
    :param isOpenGLCoords:
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        pts3D = pts3D.dot(coordChangeMat.T)

    fx = 822.79041
    fy = 822.79041
    cx = 318.47345
    cy = 250.31296

    camMat = np.array([[fx, 0, cx], [0, fy, cy], [0., 0., 1.]])

    projPts = pts3D.dot(camMat.T)
    projPts = np.stack([projPts[:,0]/projPts[:,2], projPts[:,1]/projPts[:,2]],axis=1)

    assert len(projPts.shape) == 2

    return projPts
print(cv2ProjectPoints(np.array(label)))
exit()
def get_label(path):
    pass # 重新处理数据集
    data = scio.loadmat(labels_dir + p)
    return labels[int(path.split('/')[-1][:-4])]['joint_self']

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
    json.dump(label, open('/HDD/ningbo/fileshare/Panoptic/cropped/' + '143_' + path.split('/')[-1][:-4]+'.json','w'))
    return (left, top, left + w, top + w)

i = 0
for path in images_path:
    im = Image.open(path)
    region = im.crop(box(im, path, im.size))
    i += 1
    if not i % 500:
        region.show()
    region = region.resize((224, 224))
    print(str(i)+'/'+str(len(images_path)) + 'croped image',region.size, '143_' + path.split('/')[-1])
    region.save('/HDD/ningbo/fileshare/Panoptic/cropped/' + '143_' + path.split('/')[-1])


