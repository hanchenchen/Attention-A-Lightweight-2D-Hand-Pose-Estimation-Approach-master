import tensorflow as tf
from PIL import Image, ImageDraw
import json
import time
import os
import random
import pickle
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
start_time = time.time()
configs = json.load(open('configs/HO3D_v2.json'))
print('Reading images...')
images_path = tf.io.gfile.glob(configs['origin_images_path'])
def projectPoints(camMat, pts3D, isOpenGLCoords=True):
    '''
    TF function for projecting 3d points to 2d using CV2
    :param camProp:
    :param pts3D:
    :param isOpenGLCoords:
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if isOpenGLCoords:
        pts3D = pts3D.dot(coordChangeMat.T)

    projPts = pts3D.dot(camMat.T)
    projPts = np.stack([projPts[:,0]/projPts[:,2], projPts[:,1]/projPts[:,2]],axis=1)

    assert len(projPts.shape) == 2
    return projPts

def loadPickleData(fName):
    with open(fName, 'rb') as f:
        try:
            pickData = pickle.load(f, encoding='latin1')
        except:
            pickData = pickle.load(f)
    return pickData

def get_label(dir, idx):
    id = str(idx)
    while len(id)<4:
        id = '0' + id
    data = loadPickleData('/HDD/ningbo/fileshare/HO3D_v2/train/' + dir + '/meta/' + id + '.pkl')
    '''for i, j in data.items():
        print(i, j)'''
    label = projectPoints(data['camMat'], data['handJoints3D'], True)
    order = [0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]
    label = label[order]
    # print(label)
    return label

def box(im, path):
    label = get_label(path.split('/')[-3], int(path[-8:-4])).tolist()
    midst = label[9]
    midst[0] += random.uniform(-5.5, 5.5)
    midst[1] += random.uniform(-5.5, 5.5)
    left, top, right, bottom = int(midst[0] - 224), int(midst[1] - 224), int(midst[0] + 224), int(midst[1] + 224)
    for i in range(21):
        label[i][0] -= left
        label[i][0] /= 2.
        label[i][1] -= top
        label[i][1] /= 2.
    # print(label)
    json.dump(label, open('/HDD/ningbo/fileshare/HO3D_v2/cropped/' + path.split('/')[-3] + path.split('/')[-1][:-4]+'.json', 'w'))
    return (left, top, left + 448, top + 448)

i = 0
for path in images_path:
    im = Image.open(path)
    region = im.crop(box(im, path))
    i += 1
    region = region.resize((224,224))
    if not i % 300:
        region.show()
    print(i, '/', 66034,  'croped image',region.size, path.split('/')[-3] ,path.split('/')[-1])
    region.save('/HDD/ningbo/fileshare/HO3D_v2/cropped/' + path.split('/')[-3] + path.split('/')[-1])


