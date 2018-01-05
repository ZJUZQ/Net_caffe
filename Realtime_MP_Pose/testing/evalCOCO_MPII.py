#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os, sys
cwd = os.getcwd()
sys.path.insert(0, os.path.join(cwd, 'src'))

## path of your caffe
caffe_path = os.environ['HOME'] + '/caffe/' 
sys.path.insert(0, os.path.join(caffe_path, 'python'))

from eval_config import eval_config
from applyModel import applyModel
from connect56LineVec import connect56LineVec


import cv2 as cv 
import numpy as np
import caffe
import time
import argparse
import json
import scipy.io as sio ## This is the package from which loadmat, savemat and whosmat are imported.

def evalCOCO(detectImgs, jsonFile, savedImgPath): 
   
    orderCOCO = [1,0, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4] # openpose's keypoint in coco's order
    # coco keypoints order: ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]

    param = eval_config(0) ## For COCO, mode = 0
    model = param['model'][param['modelID']]
    net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

    pred = [] 
    for i in range(len(detectImgs)): ## iterate all val images
        imgPath = detectImgs[i]

        ## Each dict of pred store the detection results of one image
        pred.append({'annorect': [], 'candidates': [], 'image_id': ''})
        pred[i]['image_id'] = imgPath.split('/')[-1].split('.')[0]

        #candidates, subset = applyModel_Group(net, model, param, imgPath)
        heatMaps = applyModel(imgPath, param, net)
        candidates, subset = connect56LineVec(imgPath, heatMaps, param, savedImgPath)
        
        people_cnt = 0 ## detected people count in current image
        for ridxPred in range(len(subset)):
            point = []
            part_cnt = 0 # joint part couont detected of this people
            for part in range(18):
                if part == 1:
                    continue ## pass neck, which is not labeled in coco.json
                index = subset[ridxPred, part] ## subset: [[], ...]
                if (index > 0):
                    point.append({})
                    ## candidates [[x, y, score, peak_id],...]
                    print('index ', index)
                    point[part_cnt]['x'] = candidates[int(index), 0]
                    point[part_cnt]['y'] = candidates[int(index), 1]
                    point[part_cnt]['score'] = candidates[int(index), 2]
                    ##point[part_cnt]['id'] = orderCOCO[part] - 1
                    point[part_cnt]['id'] = orderCOCO[part]
                    part_cnt = part_cnt + 1

            
            pred[i]['annorect'].append({})
            pred[i]['annorect'][people_cnt]['point'] = point
            pred[i]['annorect'][people_cnt]['score'] = subset[ridxPred, -2] ## the second last number of each row in subset is the score of the overall configuration
            people_cnt = people_cnt + 1
        pred[i]['candidates'] = candidates


    ## convert the format to coco json for evaluation
    json_for_coco_eval = []

    count = 0 ## count of predicted persons
    for j in range(len(pred)): ## number of images     
        for d in range(len(pred[j]['annorect'])): ## number of peoples in current image
            json_for_coco_eval.append({'image_id': [], 'category_id': [], 'keypoints': [], 'score': []})
            json_for_coco_eval[count]['image_id'] = pred[j]['image_id']
            json_for_coco_eval[count]['category_id'] = 1
            json_for_coco_eval[count]['keypoints'] = np.zeros( (3, 17) )
            for p in range(len(pred[j]['annorect'][d]['point'])): ## part points of this person
                point = pred[j]['annorect'][d]['point'][p]
                json_for_coco_eval[count]['keypoints'][0, point['id']] = point['x'] - 0.5
                json_for_coco_eval[count]['keypoints'][1, point['id']] = point['y'] - 0.5
                json_for_coco_eval[count]['keypoints'][2, point['id']] = 1 ## visiable of this part
            #json_for_coco_eval[count]['keypoints'] = np.reshape(json_for_coco_eval[count]['keypoints'], (1, 51), order='F').tolist()
            json_for_coco_eval[count]['keypoints'] = json_for_coco_eval[count]['keypoints'].flatten('F').tolist()
            json_for_coco_eval[count]['score'] = pred[j]['annorect'][d]['score']
            count = count + 1

    #print 'json_for_coco_eval', json_for_coco_eval
    jsonData = json.dumps(json_for_coco_eval, indent = 4)
    #print 'jsonData', jsonData
    fileWriter = open(jsonFile, 'w')
    fileWriter.write(jsonData)
    fileWriter.close()

def evalMPII(jsonFile, savedImgPath):

    param = eval_config(1)
    model = param['model'][param['modelID']]
    net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

    orderMPI = [9,8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5] ## used part of MPII joints, and no 'center' ('center' is where ?)
    ## MPII joint order: joint id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis(骨盆), 7 - thorax(胸部), 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)
    targetDist = 41.0/35
    boxisze = model['boxsize']

    pred = []
    pred_count = -1

    annolist_test = sio.loadmat(model['annolist_test'])
    annolist_test = annolist_test['annolist_test']
    annorect = annolist_test['annorect']
    for i in range(annolist_test['image'].size):
        print('i = %d'%i)

        #pred.append({'annorect': [], 'image_name': ''})

        imgPath = str(annolist_test['image'][0, i+30]['name'][0, 0][0])
        print('Precess: %s'%imgPath)
        imgPath = os.path.join(model['MPII_imageFolder'], imgPath)
        oriImg = cv.imread(imgPath)

        rect = annorect[0, i]
        if(rect.size == 0):
            continue 
        pos = np.zeros((rect.size, 2))   ## rough human position in the image
        scale = np.zeros((rect.size, 1)) ## person scale w.r.t. 200 px height
        for ridx in range(rect.size): ## rectangel index
            pos[ridx, :] = [float(rect['objpos'][0, ridx]['x']), float(rect['objpos'][0, ridx]['y'])]
            scale[ridx, 0] = float(rect['scale'][0, ridx])

        minX = min(pos[:,0])
        minY = min(pos[:,1])
        maxX = max(pos[:,0])
        maxY = max(pos[:,1])

        scale0 = targetDist / np.mean(scale, 0)
        deltaX = boxisze / (scale0 * param['crop_ratio'])
        deltaY = boxisze / (scale0 * 2)

        bbox = 4 * [0]
        dX = deltaX * param['bbox_ratio']
        dY = deltaY * param['bbox_ratio']
        bbox[0] = max(minX-dX, 1)
        bbox[1] = max(minY-dY, 1)
        bbox[2] = min(maxX+dX, oriImg.shape[1])
        bbox[3] = min(maxY+dY, oriImg.shape[0])

        heatMaps = applyModel(imgPath, param, net)
        candidates, subset = connect56LineVec(imgPath, heatMaps, param, savedImgPath)

        pred.append({'annorect': [], 'image_name': ''})
        pred_count = pred_count + 1
        
        people_cnt = 0 ## detected people count in current image
        for ridxPred in range(len(subset)):
            point = []
            sum_x = 0 
            sum_y = 0 
            part_cnt = 0 # joint part couont detected of this people
            for part in range(14):
                index = subset[ridxPred, part] ## subset: [[], ...]
                if (index > 0):
                    point.append({})
                    ## candidates [[x, y, score, peak_id],...]
                    #print 'index ', index
                    point[part_cnt]['x'] = candidates[int(index), 0]
                    point[part_cnt]['y'] = candidates[int(index), 1]
                    sum_x = sum_x + point[part_cnt]['x']
                    sum_y = sum_y + point[part_cnt]['y']
                    point[part_cnt]['score'] = candidates[int(index), 2]
                    point[part_cnt]['id'] = orderMPI[part]
                    part_cnt = part_cnt + 1

            mean_x = sum_x / part_cnt 
            mean_y = sum_y / part_cnt 
            index = subset[ridxPred, 14]
            print('index = %d'%index)
            if( ( mean_x > bbox[0] and mean_x < bbox[2] and mean_y > bbox[1] and mean_y < bbox[3] ) or \
                ( int(index) >= 0 and candidates[int(index), 0] > bbox[0] and candidates[int(index), 0] < bbox[2] and candidates[int(index), 1] > bbox[1] and candidates[int(index), 1] < bbox[3]) ):
                pred[pred_count]['annorect'].append({'annopoints': []})
                pred[pred_count]['annorect'][people_cnt]['annopoints'] = point
                pred[pred_count]['image_name'] = imgPath.split('/')[-1]
                people_cnt = people_cnt + 1

    ## convert the format to coco json for evaluation
    json_for_mpii_eval = []

    count = 0 ## count of predicted persons
    for j in range(len(pred)): ## number of images    
        json_for_mpii_eval.append({'image_name': [], 'annorect': [] }) 
        json_for_mpii_eval[count]['image_name'] = pred[j]['image_name']

        for d in range(len(pred[j]['annorect'])): ## number of peoples in current image
            json_for_mpii_eval[count]['annorect'].append({'annopoints': []})
        
            for p in range(len(pred[j]['annorect'][d]['annopoints'])): ## part points of this person
                point = pred[j]['annorect'][d]['annopoints'][p]
                json_for_mpii_eval[count]['annorect'][d]['annopoints'].append(point)

        count = count + 1

    #print 'json_for_mpii_eval', json_for_mpii_eval
    jsonData = json.dumps(json_for_mpii_eval, indent = 4)
    #print 'jsonData', jsonData
    fileWriter = open(jsonFile, 'w')
    fileWriter.write(jsonData)
    fileWriter.close()


def parse_args():
    parser = argparse.ArgumentParser(description='eval and save result for evaluaton for COCO')

    parser.add_argument('--mode', type = str, default = 'COCO',
                        help='COCO or MPII')

    parser.add_argument('--image', 
                        help='image path to evaluate')

    parser.add_argument('--folder', 
                        help='folder path containing images to evaluate')

    parser.add_argument('--write_COCO_json', type = str, default = 'result.json',
                        help='write coco json result file')

    parser.add_argument('--write_MPII_json', type = str, default = 'result.json',
                        help='write MPII json result file')

    parser.add_argument('--write_render_image', type = str, default = '',
                        help='folder path to store saved image')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if(args.mode != 'COCO' and args.mode != 'MPII'):
        print( '\'python evalCOCO.py -h\' for usage help !' )
        sys.exit(0)

    outFile = ''
    if(args.write_COCO_json):
        outFile = args.write_COCO_json

    if(args.write_MPII_json):
        outFile = args.write_MPII_json

    savedImgPath = ''
    if(args.write_render_image):
        savedImgPath = args.write_render_image

    imagePath = args.image
    folderPath = args.folder 
    detectImgs = []

    if(imagePath): ## input is image path
        _, ext = os.path.splitext(imagePath)
        if ext.lower() in [".jpg", ".jpeg", ".bmp", ".png"]:
            detectImgs.append(imagePath)
    elif(folderPath):
        for f in os.listdir(folderPath):
            file = os.path.join(folderPath, f)
            _, ext = os.path.splitext(file)
            if ext.lower() in [".jpg", ".jpeg", ".bmp", ".png"]:
                detectImgs.append(file)
    #else:
    #    print( 'Please assign which image/images to evaluate !' )
    #    sys.exit(0)

    if (args.mode == 'COCO'):
        print('COCO detection')
        evalCOCO(detectImgs, outFile, savedImgPath)
    elif (args.mode == 'MPII'):
        print('MPII detection')
        evalMPII(outFile, savedImgPath)


if __name__ == "__main__":
    main()


