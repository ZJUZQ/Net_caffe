
# -*- coding: UTF-8 -*-

import cv2 as cv
import numpy as np
import scipy
import PIL.Image
import math
import caffe
import time
import matplotlib
from scipy.ndimage.filters import gaussian_filter

def applyModel_Group(net, model, param, imgPath):
    ### B,G,R order
    oriImg = cv.imread(imgPath) 
    
    """
    scale = max(model['boxsize']/ float(oriImg.shape[0]),
                model['boxsize']/ float(oriImg.shape[1]))
    
    multiplier = [ max( x * model['boxsize'] / oriImg.shape[0],  
                        x * model['boxsize'] / oriImg.shape[1] ) for x in param['scale_search'] ]
    """

    multiplier = [x * model['boxsize'] / oriImg.shape[0] for x in param['scale_search']]

    print multiplier
    

    heatmap_avg = np.zeros( (oriImg.shape[0], oriImg.shape[1], 19) )
    paf_avg = np.zeros( (oriImg.shape[0], oriImg.shape[1], 38) )

    for m in range(len(multiplier)):
        scale = multiplier[m]
        print scale
        resizeImg = cv.resize(oriImg, (0,0), fx=scale, fy=scale,  interpolation=cv.INTER_CUBIC)

        print("Shape after padding:", resizeImg.shape) 
       
        padImg, pad = padRightDownCorner(resizeImg, model['stride'], model['padValue'])
        print("Shape after padding:", padImg.shape) 
        
        net.blobs['data'].reshape(*(1, 3, padImg.shape[0], padImg.shape[1]))

        #net.forward() # dry run
        net.blobs['data'].data[...] = np.transpose( np.float32(padImg[:,:,:,np.newaxis]), (3,2,0,1) )/256 - 0.5;
        start_time = time.time()
        output_blobs = net.forward()
        print('Forward time: %.2f ms.'%(1000 * (time.time() - start_time)))

        print 'debug: 1'
        ## extract outputs, resize, and remove padding
        heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1,2,0)) # output 1 is heatmaps
        heatmap = cv.resize(heatmap, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
        heatmap = heatmap[:padImg.shape[0]-pad[2], :padImg.shape[1]-pad[3], :]
        heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
        
        paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1,2,0)) # output 0 is PAFs
        paf = cv.resize(paf, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
        paf = paf[:padImg.shape[0]-pad[2], :padImg.shape[1]-pad[3], :]
        paf = cv.resize(paf, (oriImg.shape[1],oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

        print 'debug: 2'
        print heatmap.shape, heatmap_avg.shape, len(multiplier)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    ## Extrack Peak
    all_peaks = []
    peak_counter = 0
    for part in range(model['kpt_num']): # note that, kpts dont include background
        x_list = []
        y_list = []
        map_ori = heatmap[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        ## nmp
        peaks_binary=np.logical_and.reduce((map >= map_left,
                                            map >= map_right,
                                            map >= map_up,
                                            map >= map_down,
                                            map>param['thre1']))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) ## note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    print("Total candidate part point number: %d"%(peak_counter))

    ### Find connection in the specified sequence
    #center 29 is in the position 15
    limbSeq = model['linkPair']
    # the middle joints heatmap correpondence
    mapIdx = model['linkPafIdx']

    connection_all = []
    special_k = []
    mid_num = 10 ## sample point number between two point
    for k in range(len(mapIdx)):
        score_mid = paf[:,:,[x-model['kpt_num']-1 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], 
                                      candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])+0.000001
                    vec = np.divide(vec, norm)
                
                    startend=zip(np.linspace(candA[i][0],
                                             candB[j][0],
                                             num=mid_num), 
                                 np.linspace(candA[i][1], 
                                             candB[j][1],
                                             num=mid_num))
                
                    vec_x=np.array([score_mid[int(round(startend[I][1])), 
                                              int(round(startend[I][0])), 0] for I in range(len(startend))])
                    vec_y=np.array([score_mid[int(round(startend[I][1])), 
                                              int(round(startend[I][0])), 1] for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, 
                                                     score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break
            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    
    ## 将属于同一个人的关节进行合并：用subset表示候选人的集和，对于每个关节对partA-partB:
    #  1）如果 partA、partB 没有一个在subset中出现，说明该关节对 属于一个新的人，则subset新加一行
    #  2）如果 partA、partB 中只有一个出现在subset中，说明该关节对 属于subset中的一个人（一行）
    #  3）如果 partA、partB 中两个都出现在subset中：
    #           3.1）如果 partA、partB 属于sbuset中的同一行，则subset该行新加一个关节对
    #           3.2）如果 partA、partB 不属于sbuset中的同一行，则subset该两行merge,并加一个关节对
    #
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    #print special_k, mapIdx, connection_all
    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]

                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i]:
                        # partA is alread in subset
                        subset_idx[0] = j
                        found += 1
                    if subset[j][indexB] == partBs[i]:
                        # partB is alread in subset
                        subset_idx[1] = j
                        found += 1
            
                if found == 1: # 
                    print('found = 1')
                    if subset_idx[0] > -1: # partA is alread in subset
                        j = subset_idx[0]
                        if(subset[j][indexB] != partBs[i]): # find a new part of subset[j][indexB]
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            # candidate[partBs[i].astype(int), 2] 是 partBs[i] 的score
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif subset_idx[1] > -1: # partB is alread in subset
                        j = subset_idx[1]
                        if(subset[j][indexA] != partAs[i] ): # find a new part for subset[j][indexA]
                            subset[j][indexA] = partAs[i]
                            subset[j][-1] +=1
                            # candidate[partAs[i].astype(int), 2] 是 partAs[i] 的score
                            subset[j][-2] += candidate[partAs[i].astype(int), 2] + connection_all[k][i][2]

                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx # partA[i]在人subset[j1]中， partB[i]在人subset[j2]中
                    print "found = 2"
                    membership = ( (subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int) )[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        # subset[j1]和subset[j2]没有其他共同关节对， 通过当前关节对partAs[i]-partBs[i]，将两个subset合并
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2] # 加上当前关节对的score
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        # 如果subset[j1]和subset[j2]存在其他相同的关节 (实际上 j1 == j2)
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 2 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)



    print 'debug: xx'
    # visualize_point
    colors = model['part_color']
    cmap = matplotlib.cm.get_cmap('hsv')
    canvas = cv.imread(imgPath) # B,G,R order

    for i in range(model['kpt_num']):
        rgba = np.array(cmap(1 - float(i)/model['kpt_num'] - 1./(model['kpt_num']*2)))
        rgba[0:3] *= 255
        for j in range(len(all_peaks[i])):
            cv.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
        
    # visualize connect
    stickwidth = 4
    for i in range(len(model['linkPair'])):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])-1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1])**2 + (Y[0]-Y[1])**2)** 0.5
            angle = math.degrees(math.atan2(X[0]-X[1],
                                            Y[0]-Y[1]))
            polygon = cv.ellipse2Poly((int(mY),int(mX)), 
                                      (int(length/2), 
                                       stickwidth), 
                                      int(angle), 0, 360, 1)
            color = colors[i%len(model['part_color'])]
            cv.fillConvexPoly(cur_canvas, polygon, color)
            canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 
                                    0.6, 0)
    res = cv.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
    print 'debug: xxxx'
    if(param['save_image']):
        cv.imwrite("%s_res.jpg"%(imgPath), res)

    return candidate, subset
    

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0] # img.shape = [height, width, depth]
    w = img.shape[1]
    print 'h', h, 'w', w

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right
    print pad

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :], [pad[0], 1, 1])*0 + padValue
    print 'pad_up.shape', pad_up.shape
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    print 'img_padded', img_padded.shape


    pad_left = np.tile(img_padded[:, 0:1, :], [1, pad[1], 1])*0 + padValue
    print 'pad_left.shape', pad_left.shape
    img_padded = np.concatenate((pad_left, img_padded), axis=1)

    pad_down = np.tile(img_padded[-2:-1,:,:], [pad[2], 1, 1])*0 + padValue
    print pad_down.shape
    img_padded = np.concatenate((img_padded, pad_down), axis=0)

    pad_right = np.tile(img_padded[:,-2:-1,:], [1, pad[3], 1])*0 + padValue
    print pad_right.shape
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    print 'img_padded', img_padded.shape
    return img_padded, pad