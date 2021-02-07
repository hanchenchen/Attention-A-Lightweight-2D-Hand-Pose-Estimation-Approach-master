import os
import numpy as np
import logging
import matplotlib.pyplot as plt

def get_pck_with_sigma(predict_labels_dict, gt_labels, sigma_list):
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
        pred_label = predict_labels_dict[im]['pred_label']  # list    len:21      element:[x, y]
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
