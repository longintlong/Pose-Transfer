import os
import re
import sys
import cv2
import csv
import shutil
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
# from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from network.rtpose_vgg import get_model
from network.post import decode_pose
from training.datasets.coco_data.preprocessing import (inception_preprocess,rtpose_preprocess,ssd_preprocess, vgg_preprocess)
from network import im_transform
from tool.compute_coordinates import compute_cordinates


# parser = argparse.ArgumentParser()
# parser.add_argument('--vpath', help="the input video path", default="test.mp4")
# parser.add_argument('--spath', help="the source img path", default="source.jpg")
# #parser.add_argument('--pth_file', required=True)
# args = parser.parse_args()

param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}

weight_name = './HPE/weight/pose_model.pth'

#################################
'''just for debug'''
# video = "dancing.mp4"
# source_path = "../demo_data/test/source.jpg"
#################################


def judge_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def get_multiplier(img):
    """Computes the sizes of image at different scales
    :param img: numpy array, the current image
    :returns : list of float. The computed scales
    """
    scale_search = [0.5, 1., 1.5, 2, 2.5]
    return [x * 368. / float(img.shape[0]) for x in scale_search]


def get_outputs(multiplier, img, model, preprocess):
    """Computes the averaged heatmap and paf for the given image
    :param multiplier:
    :param origImg: numpy array, the image being processed
    :param model: pytorch model
    :returns: numpy arrays, the averaged paf and heatmap
    """

    heatmap_avg = np.zeros((img.shape[0], img.shape[1], 19))
    paf_avg = np.zeros((img.shape[0], img.shape[1], 38))
    max_scale = multiplier[-1]
    max_size = max_scale * img.shape[0]
    # padding
    max_cropped, _, _ = im_transform.crop_with_factor(
        img, max_size, factor=8, is_ceil=True)
    batch_images = np.zeros(
        (len(multiplier), 3, max_cropped.shape[0], max_cropped.shape[1]))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        inp_size = scale * img.shape[0]

        # padding
        im_croped, im_scale, real_shape = im_transform.crop_with_factor(
            img, inp_size, factor=8, is_ceil=True)

        if preprocess == 'rtpose':
            im_data = rtpose_preprocess(im_croped)

        elif preprocess == 'vgg':
            im_data = vgg_preprocess(im_croped)

        elif preprocess == 'inception':
            im_data = inception_preprocess(im_croped)

        elif preprocess == 'ssd':
            im_data = ssd_preprocess(im_croped)

        batch_images[m, :, :im_data.shape[1], :im_data.shape[2]] = im_data

    # several scales as a batch
    batch_var = torch.from_numpy(batch_images).cuda().float()
    predicted_outputs, _ = model(batch_var)
    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
    heatmaps = output2.cpu().data.numpy().transpose(0, 2, 3, 1)
    pafs = output1.cpu().data.numpy().transpose(0, 2, 3, 1)

    for m in range(len(multiplier)):
        scale = multiplier[m]
        inp_size = scale * img.shape[0]

        # padding
        im_cropped, im_scale, real_shape = im_transform.crop_with_factor(
            img, inp_size, factor=8, is_ceil=True)
        heatmap = heatmaps[m, :int(im_cropped.shape[0] /
                           8), :int(im_cropped.shape[1] / 8), :]
        heatmap = cv2.resize(heatmap, None, fx=8, fy=8,
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[0:real_shape[0], 0:real_shape[1], :]
        heatmap = cv2.resize(
            heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = pafs[m, :int(im_cropped.shape[0] / 8), :int(im_cropped.shape[1] / 8), :]
        paf = cv2.resize(paf, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        paf = paf[0:real_shape[0], 0:real_shape[1], :]
        paf = cv2.resize(
            paf, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    return paf_avg, heatmap_avg


def extract_pose(video, oriImg, name, filestream, model):
    # Get results of original image
    multiplier = get_multiplier(oriImg)
    with torch.no_grad():
        paf, heatmap = get_outputs(
            multiplier, oriImg, model, 'rtpose')

    pose_cords = compute_cordinates(heatmap, paf, oriImg)
    # coordinate
    print("{}: {}: {}".format(str(name) + ".jpg", str(list(pose_cords[:, 0])), str(list(pose_cords[:, 1]))),
          file=filestream)
    filestream.flush()
    canvas, to_plot, candidate, subset = decode_pose(
        oriImg, param, heatmap, paf)

    save_result(video,oriImg,to_plot,name)


def save_result(video, oriImg, pose, name):
    path = './HPE/result/'+video
    judge_dir(path)
    cv2.imwrite('./demo_data/test/{}.jpg'.format(name), oriImg)
    cv2.imwrite('{}/{}.png'.format(path, name), pose)


def extract_pose_main(video_path, source_path):
#if __name__ == "__main__":

    '''
    :param video_path: viedo's img size size is 256*256
    :param source_path: Image size must be 256*176 and .jpg
    :return:
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'

    model = get_model('vgg19')
    model.load_state_dict(torch.load(weight_name))
    model.cuda()
    model.float()
    model.eval()



    video = video_path
    print("video path is ", video_path)
    source_img = cv2.imread(source_path)
    video_capture = cv2.VideoCapture(video)

    pairLst = "./demo_data/demo-resize-pairs-test.csv"
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print("fps is ",fps)
    a = 0

    while video_capture.isOpened():
        if a == 0:
            result_file = open("./demo_data/demo-resize-annotation-test.csv", 'w')
            print("name:keypoints_y:keypoints_x", file=result_file)
            result_file1 = open("./demo_data/demo-resize-pairs-test.csv", 'w')
            writer = csv.writer(result_file1)
            writer.writerow(["from", "to"])
            extract_pose(video_path.split("/")[-1][:-4], source_img, source_path.split("/")[-1][:-4], result_file, model)
        ret, oriImg = video_capture.read()
        print(oriImg.shape)
        #oriImg = oriImg[27:710,430:900,:]
        oriImg = oriImg[40:808, 8:536, :]
        oriImg = cv2.copyMakeBorder(oriImg, 0, 0, 120, 120, cv2.BORDER_CONSTANT, value=[255,255,255])
        # cv2.imwrite('../demo_data/test/{}.jpg'.format(a), oriImg)
        # break
        oriImg = cv2.resize(oriImg, (256,256), interpolation=cv2.INTER_LINEAR)
        shape_dst = np.min(oriImg.shape[0:2])

        extract_pose(video_path.split("/")[-1][:-4], oriImg, a, result_file, model)

        # pairLst
        writer.writerow([source_path.split("/")[-1], str(a)+".jpg"])
        print("finished {} pics".format(a))

        # cv2.imshow('Video', to_plot)
        a = a + 1
        if a > 100:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    torch.cuda.empty_cache()