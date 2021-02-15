import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import json
from sklearn.metrics import auc
def get_pck_with_sigma(predict_labels_dict, gt_labels, sigma_list = [0.1, 0.15, 0.2, 0.25, 0.3]):
    """
    Get PCK with different sigma threshold
    :param predict_labels_dict:  dict  element:  'img_name':{'prd_label':[list, coordinates of 21 keypoints],
                                                             'resol': origin image size}
    :param gt_labels:            dict  element:  'img_name': [list, coordinates of 21 keypoints ]
    :param sigma_list:       list    different sigma threshold
    :return:
    """
    pck_dict = {}
    for im in predict_labels_dict:
        gt_label = gt_labels[im]        # list    len:21      element:[x, y]
        pred_label = predict_labels_dict[im]['prd_label']  # list    len:21      element:[x, y]
        im_size = predict_labels_dict[im]['resol']
        for sigma in sigma_list:
            if sigma not in pck_dict:
                pck_dict[sigma] = []
            pck_dict[sigma].append(PCK(pred_label, gt_label, im_size/2.2, sigma))
            # Attention!
            # since our cropped image is 2.2 times of hand tightest bounding box,
            # we simply use im_size / 2,2 as the tightest bounding box

    pck_res = {}
    for sigma in sigma_list:
        pck_res[sigma] = sum(pck_dict[sigma]) / len(pck_dict[sigma])
    return pck_res


def PCK(predict, target, bb_size=256, sigma=0.1):
    """
    Calculate PCK
    :param predict: list    len:21      element:[x, y]
    :param target:  list    len:21      element:[x, y]
    :param bb_size: tightest bounding box length of hand
    :param sigma:   threshold, we use 0.1 in default
    :return: scala range [0,1]
    """
    pck = 0
    for i in range(21):
        pre = predict[i]
        tar = target[i]
        dis = np.sqrt((pre[0] - tar[0]) ** 2 + (pre[1] - tar[1]) ** 2)
        if dis < sigma * bb_size:
            pck += 1
    return pck / 21.0
'''import json
predictions = json.load(open('Panoptic/predictions.json'))
groud_truth = json.load(open('Panoptic/ground_truth.json'))
get_pck_with_sigma(predictions, groud_truth)'''

def get2DPCK(predictions, gt, figName=None, showFig=False):
    # calc. PCK
    interval = np.arange(0, 100 + 1, 1)
    errArr = np.zeros((len(interval),), dtype=np.float32)
    projErrs = np.nanmean(np.linalg.norm(gt - predictions, axis=2), axis=1)
    cntr = 0
    for i in interval:
        errArr[cntr] = float(np.sum(projErrs < i)) / float(gt.shape[0]) * 100.
        cntr += 1

    AUC = auc(interval, errArr)
    # plot it
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(interval,
            errArr,
            # c=colors[0],
            linestyle='-', linewidth=1)
    plt.xlabel('Distance threshold / px', fontsize=12)
    plt.ylabel('Fraction of frames within distance / %', fontsize=12)
    plt.xlim([0.0, 100])
    plt.ylim([0.0, 100.0])
    ax.grid(True)

    # save if required
    if figName is not None:
        fig.savefig(figName,
                    bbox_extra_artists=None,
                    bbox_inches='tight')
        with open(figName.split('.')[0] + '.json', 'wb') as f:
            json.dump({'x': interval, 'y': errArr, 'gt': gt, 'est': predictions}, f)

    # show if required
    if showFig:
        plt.show(block=False)
    # plt.close(fig)

    return interval, errArr, AUC