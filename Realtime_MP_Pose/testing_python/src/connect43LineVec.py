# -*- coding: UTF-8 -*-
import os, sys
import copy
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import math
import cv2 as cv
import matplotlib

def connect43LineVec(imgPath, heatMaps, param, savedImgPath):
	oriImg = cv.imread(imgPath)
	model = param['model'][param['modelID']]
	peak_count = 0
	peak_candidates = []

	## non-maximum suppression for finding joint candidates
	for i in range(param['kpt_num']):
		peaks_with_score_and_id = findPeaks(heatMaps[:,:,i], param['thre1'], peak_counter)
		peak_candidates.append(peaks_with_score_and_id)
		peak_counter = peak_count + len(peaks_with_score_and_id)

	limbSeq = model['linkPair']
	pafIdx = model['linkPafIdx']

	connection_all = [] ## len(connection_all) == limbs
	special_k = [] ## store limb index which has no connection
	mid_num = param['mid_num'] ## sample point number between two point
	for k in range(len(pafIdx)):
		score_mid = heatMaps[:, :, [x-1 for x in pafIdx[k]]] ## heatMaps = [joint; paf]
		candA = peak_candidates[limbSeq[k][0]-1] ## candidates for point A of this limb
		candB = peak_candidates[limbSeq[k][1]-1] ## candidates for point B of this limb
		nA = len(candA)
		nB = len(candB)
		if(nA != 0 and nB != 0):
			connection_k_candidates = []
			for i in range(nA):
				for j in range(nB):
					vec = np.subtract(candB[j][:2], candA[i][:2])
					norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1]) + 0.000001
					vec = np.divide(vec, norm) ## unit vec
					startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
								   np.linspace(candA[i][1], candB[j][1], num=mid_num))

					## x and y in score_mid is inverse
					## mid point score in line A-B
					vec_x=np.array([score_mid[int(round(startend[I][1])), 
											  int(round(startend[I][0])), 0] for I in range(len(startend))])
					vec_y=np.array([score_mid[int(round(startend[I][1])), 
											  int(round(startend[I][0])), 1] for I in range(len(startend))])

					score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
					score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
					criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
					criterion2 = score_with_dist_prior > 0
					if criterion1 and criterion2:
						connection_k_candidates.append([i, j, score_with_dist_prior, 
													    score_with_dist_prior+candA[i][2]+candB[j][2]])

			connection_k_candidates = sorted(connection_k_candidates, key=lambda x: x[2], reverse=True)
			connection_k = np.zeros((0, 5)) ## [idA, idB, s, i, j]
			for c in range(len(connection_k_candidates)):
				i, j, s = connection_k_candidates[c][0:3]
				if( (i not in connection_k[:, 3]) and (j not in connection_k[:, 4]) ):
					connection_k = np.vstack((connection_k, [candA[i][3], candB[j][3], s, i, j]))
					if(len(connection_knet) >= min(nA, nB)):
						break 
			connection_all.append(connection_k)
		else:
			special_k.append(k) ## limb k has no connection
			connection_all.append([])

	## 将属于同一个人的关节进行合并：用subset表示候选人的集和，对于每个关节对partA-partB:
	#  1）如果 partA、partB 没有一个在subset中出现，说明该关节对 属于一个新的人，则subset新加一行
	#  2）如果 partA、partB 中只有一个出现在subset中，说明该关节对 属于subset中的一个人（一行）
	#  3）如果 partA、partB 中两个都出现在subset中：
	#		   3.1）如果 partA、partB 属于sbuset中的同一行，则subset该行新加一个关节对
	#		   3.2）如果 partA、partB 不属于sbuset中的同一行，则subset该两行merge,并加一个关节对
	#
	# last number in each row is the total parts number of that person
	# the second last number in each row is the score of the overall configuration
	subset = -1 * np.ones((0, model['kpt_num']+2))
	peak_candidates_array = np.array([item for sublist in peak_candidates for item in sublist]) # [x, y, s, id]
	for k in range(len(pafIdx)):
		if k not in special_k:
			partAs = connection_all[k][:, 0] ## ids in peak_candidates_array[:, 2]
			partBs = connection_all[k][:, 1]
			indexA, indexB = np.array(limbSeq[k]) - 1 

			for i in range(len(connection_all[k])):
				found = 0 
				subset_idx = [-1, -1] ## subset id contain partAs[i], partBs[i]

				for j in range(len(subset)):
					if subset[j][indexA] == partAs[i]: ## partAs[i] already in subset[j]
						subset_idx[0] = j
						found += 1 
					if subset[j][indexB] == partBs[i]:
						subset_idx[1] = j 
						found += 1 

				if found == 1:
					if subset_idx[0] > -1: # partA is alread in subset
						j = subset_idx[0]
						if(subset[j][indexB] != partBs[i]): # find a new part of subset[j][indexB]
							subset[j][indexB] = partBs[i]
							subset[j][-1] += 1
							# peak_candidates_array[partBs[i].astype(int), 2] 是 partBs[i] 的score
							subset[j][-2] += peak_candidates_array[partBs[i].astype(int), 2] + connection_all[k][i][2]
					elif subset_idx[1] > -1: # partB is alread in subset
						j = subset_idx[1]
						if(subset[j][indexA] != partAs[i] ): # find a new part for subset[j][indexA]
							subset[j][indexA] = partAs[i]
							subset[j][-1] +=1
							# peak_candidates_array[partAs[i].astype(int), 2] 是 partAs[i] 的score
							subset[j][-2] += peak_candidates_array[partAs[i].astype(int), 2] + connection_all[k][i][2]

				elif found == 2: # if found 2 and disjoint, merge them
					j1, j2 = subset_idx # partA[i]在人subset[j1]中， partB[i]在人subset[j2]中
					membership = ( (subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int) )[:-2]
					if len(np.nonzero(membership == 2)[0]) == 0: #two distinct subset, merge
						# subset[j1]和subset[j2]没有其他共同关节对， 通过当前关节对partAs[i]-partBs[i]，将两个subset合并
						subset[j1][:-2] += (subset[j2][:-2] + 1)
						subset[j1][-2:] += subset[j2][-2:]
						subset[j1][-2] += connection_all[k][i][2] # 加上当前关节对的score
						subset = np.delete(subset, j2, 0)
					else: # as like found == 1
						# 如果subset[j1]和subset[j2]存在其他相同的关节 (实际上 j1 == j2)
						subset[j1][indexB] = partBs[i]
						subset[j1][-1] += 1
						subset[j1][-2] += peak_candidates_array[partBs[i].astype(int), 2] + connection_all[k][i][2]

				# if find no partA in the subset, create a new subset
				elif not found and k < 17:
					row = -1 * np.ones(20)
					row[indexA] = partAs[i]
					row[indexB] = partBs[i]
					row[-1] = 2
					row[-2] = sum(peak_candidates_array[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
					subset = np.vstack((subset, row))

	# delete some rows of subset which has few parts occur or low score
	deleteIdx = [];
	for i in range(len(subset)):
		subset[i][-2] = subset[i][-2]/subset[i][-1] ## average score
		#if subset[i][-1] < 2 or subset[i][-2]/subset[i][-1] < 0.4:
		if subset[i][-1] < 2 or subset[i][-2] < 0.4:
			deleteIdx.append(i)
	subset = np.delete(subset, deleteIdx, axis=0)

	if(savedImgPath != ''):
		# visualize_point
		colors = model['part_color']
		cmap = matplotlib.cm.get_cmap('hsv') ## The colormap used to map normalized data values to RGBA colors.
		canvas = cv.imread(imgPath) # B,G,R order

		for i in range(model['kpt_num']):
			rgba = np.array(cmap(1 - float(i)/model['kpt_num'] - 1./(model['kpt_num']*2)))
			rgba[0:3] *= 255
			for j in range(len(peak_candidates[i])):
				cv.circle(canvas, peak_candidates[i][j][0:2], 4, colors[i], thickness=-1)
			
		# visualize connect
		stickwidth = 4
		for i in range(len(model['linkPair'])):
			for n in range(len(subset)):
				index = subset[n][np.array(limbSeq[i])-1] ## get ids in all joint peaks
				if -1 in index: ## limb_i not linked for this subset[n]
					continue
				cur_canvas = canvas.copy()
				Y = peak_candidates_array[index.astype(int), 0]
				X = peak_candidates_array[index.astype(int), 1]
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

		cv.imwrite("%s_res.jpg"%(os.path.join(savedImgPath, imgPath.split('/')[-1].split('.')[0])), res)

	return peak_candidates_array, subset



def findPeaks(_map, thre, peak_counter):
	map_smooth = copy.deepcopy(_map)

	map_aug = np.zeros((map_smooth.shape[0] + 1, map_smooth.shape[1] + 2))
	map_aug1 = np.zeros((map_smooth.shape[0] + 1, map_smooth.shape[1] + 2))
	map_aug2 = np.zeros((map_smooth.shape[0] + 1, map_smooth.shape[1] + 2))
	map_aug3 = np.zeros((map_smooth.shape[0] + 1, map_smooth.shape[1] + 2))
	map_aug4 = np.zeros((map_smooth.shape[0] + 1, map_smooth.shape[1] + 2))

	map_aug[1:(1+map_smooth.shape[0]), 1:(1+map_smooth.shape[1])] = copy.deepcopy(map_smooth)
	# left
	map_aug[1:(1+map_smooth.shape[0]), 0:(0+map_smooth.shape[1])] = copy.deepcopy(map_smooth)
	# right
	map_aug[1:(1+map_smooth.shape[0]), 2:(2+map_smooth.shape[1])] = copy.deepcopy(map_smooth)
	# up
	map_aug[0:(0+map_smooth.shape[0]), 1:(1+map_smooth.shape[1])] = copy.deepcopy(map_smooth)
	# down
	map_aug[2:(2+map_smooth.shape[0]), 1:(1+map_smooth.shape[1])] = copy.deepcopy(map_smooth)

	peaks_binary_aug=np.logical_and.reduce((map_aug > map_aug1,
											map_aug > map_aug2,
											map_aug > map_aug3,
											map_aug > map_aug4,
											map_aug > param['thre1']))
	peaks_binary = peaks_binary_aug[1:(1+map_smooth.shape[0]), 1:(1+map_smooth.shape[1])]
	peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) ## note reverse
	peaks_with_score = [x + (_map[x[1],x[0]],) for x in peaks]
	id = range(peak_counter, peak_counter + len(peaks))
	peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

	return peaks_with_score_and_id