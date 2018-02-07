#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Detect object in image by faster-rcnn
# --------------------------------------------------------

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import numpy as np
import os, sys, cv2
import argparse
import caffe

CLASSES = ('__background__', 
           'lianyiqun', 'duanku',
           'nvbao', 'nvxie',
           'shangyi', 'changku',
           'banshenqun', 'maozi', 'mojing')

def parse_args():
    parser = argparse.ArgumentParser(description='demo_detect')
    parser.add_argument('-g', '--gpu', dest='gpu_id', 
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('-c', '--cpu', dest='enable_cpu',
                        help='Swithch on CPU, default GPU',
                        action='store_true')
    parser.add_argument('net_deploy',
                        help='The model deploy proto file')
    parser.add_argument('net_model',
                        help='The trained caffe model')
    parser.add_argument('image_list', 
                        help='Detect image list')
    args = parser.parse_args()
    return args


def detect(net, cv_arr, min_side, max_side):
    res = {}
    ###  Resize the image
    size = cv_arr.shape[0:2]
    minS = min(size)
    maxS = max(size)
    scale = minS / min_side
    if maxS/scale > max_side:
        scale = maxS/max_side

    resized_img = np.zeros((int(round(size[0]/scale)),
                            int(round(size[1]/scale)), 3),
                          dtype=np.uint8)
    cv2.resize(cv_arr, (int(round(size[1]/scale)),
                        int(round(size[0]/scale))),
               resized_img, 0, 0, cv2.INTER_LINEAR)
    
    ### Do detect
    scores, boxes = im_detect(net, resized_img)
    if len(scores)<=0:
        return res
    boxes = boxes*scale
    
    ### Merget the result
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis]))
        dets = dets.astype(np.float32)

        ### Merge with NMS
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        ### Filter with the confidences
        keep = np.where(dets[:, -1] >= CONF_THRESH)[0]
        res["%s"%(cls)] = dets[keep, :]
    return res

def main():
    args = parse_args()
    prototxt = args.net_deploy
    caffemodel = args.net_model
    path = args.image_list
    if not os.path.exists(caffemodel) :
        raise IOError("File not exist: %s"%(caffemodel))
    if not os.path.exists(prototxt):
        raise IOError("File not exist: %s"%(prototxt))

    if args.enable_cpu:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    
    min_side = 500.0
    max_side = 1000.0
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    ### Warmup on a dummy image    
    im = 128 * np.ones((500, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        detect(net, im, min_side, max_side)

    ### Do detection
    path_list = open(path, 'r').readlines()

    frame = 0
    base = "/home/youzai.sf/data/VOCdevkit2012/VOC2012/JPEGImages"

    timer = Timer()
    timer.tic()
    log = open("./log.txt", "w")
    for p in path_list:
        p = p.strip()
        path = "%s/%s.jpg"%(base, p)
        log.write("%s.jpg"%p)
        if not os.path.exists(path):
            continue    
        im = cv2.imread(path)
        res = detect(net, im, min_side, max_side)
        for key in res:
            objs = res[key]
            for o in objs:
                log.write(",%f,%f,%f,%f,%d,%f"%(o[0], o[1],
                                                o[2], o[3],
                                                CLASSES.index(key),
                                                o[4]))
        log.write("\n")
        
    log.close()
    timer.toc()
    print("Elapse average time: %.3fs"%(timer.total_time/len(path_list)))
    
if __name__ == '__main__':
    main()