#!/usr/bin/env python

import numpy as np
import os, sys, cv2
import argparse
import matplotlib.pyplot as plt
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from easydict import EasyDict as edict
import yaml
import json

CLASS_MAP = dict()
LINE_TYPES = ['--', '-', '-.']
COLORS = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black']

def parse_args():
    parser = argparse.ArgumentParser(description='demo_eval_voc')
    parser.add_argument('-d', '--diff', dest='enable_diff', 
                        help='Evaluate the difficult sample',
                        action='store_true')
    parser.add_argument('--is12', dest='is12', 
                        help='Evaluate 2012 method, else 2007',
                        action='store_true')
    parser.add_argument('--train_model',
                        help='Train model file')
    parser.add_argument('--imgs_txt', 
                        help='txt file containing absolute path of images used for evaluattion')
    parser.add_argument('--det_json_root_path', 
                        help='Detection result file.')
    args = parser.parse_args()
    return args

def getClassNameByAnn(label):
    for key in CLASS_MAP:
        if label in CLASS_MAP[key]:
            return key
    return "other"

def getClassName(idx):
    for c in CLASS_MAP:
        if idx in CLASS_MAP[c]:
            return c
    raise Exception("Invalid class index")

def readANN(ann_path):
    ann = {} 
    if not os.path.exists(ann_path):
        return ann
    
    root = ET.parse(ann_path).getroot()
    for subroot in root:
        if "object" != subroot.tag:
            continue
        try:
            label = subroot.find('name').text ## person
        except:
            continue  
        if label != 'person':
            continue
        #label = getClassNameByAnn(label)
        xmin = float(subroot.find('bndbox').find('xmin').text)
        ymin = float(subroot.find('bndbox').find('ymin').text)
        xmax = float(subroot.find('bndbox').find('xmax').text)
        ymax = float(subroot.find('bndbox').find('ymax').text)
        diff = int(subroot.find('difficult').text)
        
        if label not in ann:
            ann[label] = []
        ann[label].append([xmin, ymin, xmax, ymax, diff])
    return ann

def readDET_image(det_json_img): ## detect result of one image
    det = {}
    if not os.path.exists(det_json_img):
        return
    with open(det_json_img, 'r') as f:
        detObjs = json.load(f)
    obj_num = len(detObjs['people'])
    if(obj_num < 1):
        return 
    for i in xrange(obj_num):
        obj = detObjs['people'][i]
        kps = obj['pose_keypoints']
        x = []
        y = []
        scores = []
        if(kps[2] <= 0): ## head point is not detected
            continue
        x.append(kps[0])
        y.append(kps[1])
        scores.append(kps[2])
        if(kps[5] <= 0 and kps[8] <= 0 and kps[11] <= 0):
            continue ## at least tow points must be detected
        for ii in range(1, 4):
            if(kps[3 * ii + 2] > 0):
                x.append(kps[3 * ii])
                y.append(kps[3 * ii + 1])
                scores.append(kps[3 * ii + 2])
        score = sum(scores) / (float(len(scores)))

        name = 'person' ## for debug
        if name not in det:
            det[name] = []
        det[name].append([score, min(x), min(y), max(x), max(y)])

    for key in det:
        arr = np.asarray(det[key], dtype=np.float32)
        ## sort the cofidence
        index = np.argsort(-arr[:, 0]) ## large to small
        det[key] = arr[index]
    return det

"""
def readDET_all(detsJson): ## all images' detect results in single json 
    det = {}
    with open(detJson, 'r') as f:
        detObjs = json.load(f)
    obj_num = len(detObjs)
    for i in xrange(obj_num):
        obj = detObjs[i]
        name = getClassName(obj['category_id'])
        if name not in det:
            det[name] = []
        det[name].append([obj['score']] + obj['bbox'])
    for key in det:
        arr = np.asarray(det[key], dtype=np.float32)
        ## sort the cofidence
        index = np.argsort(-arr[:, 0]) ## large to small
        det[key] = arr[index]
    return det
"""

def isTP(o, gts, isDiff):
    if len(gts)<1:
        return 0
    
    max_idx = -1
    max_iou = 0
    for idx, gt in enumerate(gts):
        bi=[max(o[0],gt[0]), max(o[1],gt[1]), 
            min(o[2],gt[2]), min(o[3],gt[3])]
        in_w = (bi[2]-bi[0]+1)
        in_h = (bi[3]-bi[1]+1)
        if in_w>0 and in_h>0:
            area_in = in_w*in_h*1.0
            area_un = (o[2]-o[0]+1)*(o[3]-o[1]+1) +      \
                      (gt[2]-gt[0]+1)*(gt[3]-gt[1]+1)  - \
                      area_in 
            IoU = area_in/area_un
            if IoU > max_iou:
                max_iou = IoU
                max_idx = idx

    res = -1
    if max_idx!=-1  and max_iou>=0.5:
        if not isDiff and 1==gts[max_idx][4]:
            res = -1
        else:
            res = 1
        del(gts[max_idx])
    else:
        res = 0
    return res
        
def evaluate(ann, det, benchmark, isDiff):
    ### Statistic the ground truth
    for key in ann:
        if isDiff:
            benchmark[key]['GT'] += len(ann[key])
        else:
            for obj in ann[key]:
                if 1 == obj[4]:
                    continue
                benchmark[key]['GT']+=1
        
    ### Statistic the TP and FP
    for key in det:
        objs = det[key]
        obj_num = objs.shape[0]
        if key not in ann:
            benchmark[key]['FP'] +=objs[:,0].tolist()
        else:
            gt = ann[key]
            ### pick the true positive
            for i in xrange(obj_num):
                o = objs[i]
                flag = isTP(o[1:], ann[key], isDiff) 
                if 1==flag:
                    benchmark[key]['TP'].append(o[0])
                elif 0==flag:
                    benchmark[key]['FP'].append(o[0])
    return
    
def drawAP(benchmark, name, ax_ap, ax_pre, ax_rec, lineT, color, is12):
    ap = 0
    gt_num = benchmark['GT']
    TP = np.asarray(benchmark['TP'], dtype=np.float32)
    FP = np.asarray(benchmark['FP'], dtype=np.float32)
    precision = []
    recall = []

    confs = np.hstack((TP, FP))
    confs = np.unique(np.sort(confs))
    ### Compute precision and recall
    for th in confs:
        tp_num = np.sum(TP >= th)*1.0
        fp_num = np.sum(FP >= th)
        recall.append(tp_num/gt_num)
        precision.append(tp_num/(tp_num+fp_num))
    

    ### Use 2012 Evaluation Method
    if is12:
        recall.append(0)
        precision.append(1)

        ### Setting the precision for recall r to the maximum
        ### precision obtained for any recall r*>=r
        for i in xrange(1, len(precision)):
            precision[i]=max(precision[i], precision[i-1])
        
        ### Compute the AP
        for i in xrange(len(recall)-1):
            ind = i+1
            h = recall[i]-recall[i+1]
            w = precision[i] + precision[i+1]
            ap += h*w/2
    else: ### Use 2007 Evaluation Method
        th = 0 
        precision.reverse()
        recall.reverse()
        precision = np.asarray(precision)
        recall = np.asarray(recall)
        for th in range(0,11):
            p = precision[recall>=(th/10.0)]
            if len (p) == 0:
                p=0
            else:
                p = p.max()
            ap = ap + p/11.0
                
    ax_ap.plot(recall, precision,
               linestyle = lineT,
               color = color,
               label = "%s, num=%d, ap=%0.2f%%"%(name, 
                                                 gt_num, 
                                                 ap*100))
    ax_ap.grid()
    ### Draw precision  and recall
    precision = []
    recall = []
    ths = [x/100.0 for x in range(100)]
    for th in ths:
        tp_num = np.sum(TP >= th)*1.0
        fp_num = np.sum(FP >= th)
        recall.append(tp_num/gt_num)
        if 0==fp_num:
            precision.append(1)
        else:
            precision.append(tp_num/(tp_num+fp_num))

    ax_pre.plot(ths, precision,
                linestyle = lineT,
                color = color,
                label = "%s, num=%d"%(name, gt_num))
    ax_rec.plot(ths, recall,
                linestyle = lineT,
                color = color,
                label = "%s, num=%d"%(name, gt_num))
    ax_pre.grid()
    ax_rec.grid()
    return ap*100

def main():
    args = parse_args()
    #train_model = args.train_model    
    det_json_root_path  = args.det_json_root_path
    imgs_txt = args.imgs_txt
    isDiff = args.enable_diff

    is12 = args.is12
    if is12:
        print "Evaluate with 2012 standard"
    else:
        print "Evaluate with 2007 standard"

    if not os.path.exists(imgs_txt):
        raise IOError('imgs_txt not exist: {}'.format(imgs_txt))

    if not os.path.exists(det_json_root_path):
        raise IOError("Detection result json file not exist: %s"%(det_json_root_path)) 
    
    ### Init the benchmark
    CLASS_MAP= {"person": 0}
    benchmark = {}

    for cls in CLASS_MAP:
        #if "other" in cls or "background" in cls:
        #    continue
        benchmark[cls] = {}
        benchmark[cls]["GT"] = 0
        benchmark[cls]['TP'] = []
        benchmark[cls]['FP'] = []

    ### Evaluation
    imgPaths = []
    with open(imgs_txt) as f:
        imgPaths = f.readlines()

    for imgPath in imgPaths:
        imgPath = imgPath.strip()   ## e.g., imgPath = /home/jianchong.zq/sandbox/topdown/topdown_door_A/A_001.jpg
        if not os.path.exists(imgPath):
            continue

        ## Read groun truth, 
        path, img = os.path.split(imgPath)
        path, folder = os.path.split(path)
        img, ext = os.path.splitext(img)
        ann_path = '{}/Annotations/{}/{}.xml'.format(path, folder, img)
        ann = readANN(ann_path) ## e.g., ann_path = /home/jianchong.zq/sandbox/topdown/Annotations/topdown_door_A/A_001.xml

        ## Read detect result
        det_json_path = '{}/json/{}/{}_keypoints.json'.format(det_json_root_path, folder, img)
        det = readDET_image(det_json_path)
        if det:
            evaluate(ann, det, benchmark, isDiff)
      
    ### Draw the curve
    fig_ap = plt.figure(figsize=(16,9), dpi=60)
    ax_ap = fig_ap.add_subplot(111)
    fig_pr = plt.figure(figsize=(16,9), dpi=60)
    ax_pre = fig_pr.add_subplot(211)
    ax_rec = fig_pr.add_subplot(212)

    mAP = 0
    offset = 0
    for key in benchmark:
        lineT = LINE_TYPES[offset%len(LINE_TYPES)]
        color = COLORS[offset/len(LINE_TYPES)]

        if benchmark[key]['GT']==0:
            continue
        mAP += drawAP(benchmark[key], key, 
                      ax_ap, ax_pre, ax_rec,
                      lineT, color, is12)
        offset += 1

    legend_ap = ax_ap.legend(loc='center right', 
                             bbox_to_anchor=(1.3, 0.8))
    ax_ap.set_xlabel("Recall")
    ax_ap.set_ylabel("Precision")
    valid_class_num=len(CLASS_MAP)
    for key in CLASS_MAP:
        if "background" in key.lower():
            valid_class_num -= 1
        elif "other" in key.lower():
            valid_class_num -= 1
        elif 0==benchmark[key]['GT']:
            valid_class_num -= 1
            
    ax_ap.set_title("AP curve, mAP=%0.2f%%"%(mAP/valid_class_num))
    
    legend_pr = ax_pre.legend(loc='center right', 
                              bbox_to_anchor=(1.25, 0.6))
    ax_pre.set_ylabel("Precision")
    ax_pre.set_title("Confidence-Precision")
    
    ax_rec.set_xlabel("Confidence")
    ax_rec.set_ylabel("Recall")
    ax_rec.set_title("Confidence-Recall")

    if plt.get_backend().lower() == 'agg':
        fig_ap.savefig("%s_AP.png"%(det_json_root_path),  
                       bbox_extra_artists=(legend_ap,),
                       bbox_inches='tight')
        fig_pr.savefig("%s_PR.png"%(det_json_root_path),  
                       bbox_extra_artists=(legend_pr,),
                       bbox_inches='tight')
    else:
        fig_ap.show()
        fig_pre_rec.show()

if __name__ == '__main__':
    main()