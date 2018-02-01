
# -*- coding: UTF-8 -*-
import cv2 as cv
import math
import numpy as np
import copy
import os, sys

## input: 
#		
#  output:
#  		heatMaps: 

def applyModel(imgPath, param, net):

	## select model and other parameters from variable param
	model = param['model'][param['modelID']] ## modelID = 0 for COCO
	boxsize = int(model['boxsize'])
	kpt_num = model['kpt_num'] ## number of keypoints
	oriImg = cv.imread(imgPath) ## opencv read image in B G R order

	if(1): ## Do contrast limiting adaptive histogram equalization
		lab_image = cv.cvtColor(oriImg, cv.COLOR_BGR2LAB)
		clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
		L_channel = clahe.apply(lab_image[: ,: ,0]) ## apply the CLAHE algorithm to the L channel
		lab_image[:, :, 0] = L_channel
		oriImg = cv.cvtColor(lab_image, cv.COLOR_LAB2BGR)

	"""
	octave = int(param['octave'])
	starting_range = double(param['starting_range'])
	ending_range = double(param['ending_range'])
	assert (starting_range <= ending_range), 'starting ratio should <= ending ratio'
	assert (octave >= 1), 'octave should >= 1'

	starting_scale = boxsize / ( oriImg.shape[0] * ending_range ) # oriImg.shape == [height, width, depth]
	ending_scale = boxsize / ( oriImg.shape[0] * starting_range )
	multiplier = 2**( log(starting_scale, 2):(1/octave):log(ending_scale, 2) )
	#print 'multiplier = \n', multiplier
	"""
	#multiplier = [float(x) * boxsize / min(oriImg.shape[0], oriImg.shape[1]) for x in param['scale_search']]
	multiplier = []
	for x in param['scale_search']:
		plier = float(x) * boxsize / min(oriImg.shape[0], oriImg.shape[1])
		if( plier * max(oriImg.shape[0], oriImg.shape[1]) > model['maxsize'] ):
			plier = float(model['maxsize']) / max(oriImg.shape[0], oriImg.shape[1])
		multiplier.append(plier)
	print( 'multiplier = ', multiplier, '\n' )

	## data container for each scale
	heatMaps_all = []

	for m in range(len(multiplier)):
		scale = multiplier[m]

		resizeImg = cv.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
		#print 'oriImg.shape = ', oriImg.shape, '\n'
		#print 'resizeImg.shape = ', resizeImg.shape, '\n'

		#padImg, pad[m] = padRightDownCorner(imageToTest, model['stride'], model['padValue'])

		padImg, pad = padHeight(resizeImg, model['padValue'], model['stride']); # pad so that height and width is multiples of stride
		print 'padImg.shape = ', padImg.shape, '\n'
		imageToTest = normalizeImg(padImg, 0.5)

		if imageToTest.size > 1080*960*3:
			print 'scaled image too large, ignored !!!\n'
			continue

		if(param['modelID'] == 0):
			heatMaps = applyDNN_COCO(imageToTest, net, model['stride']) ## [H, W, 57] = [heatmap_19; paf_38]
			#print 'heatMaps.shape  = ', heatMaps.shape, '\n'
		elif(param['modelID'] == 1):
			heatMaps = applyDNN_MPII(imageToTest, net, model['stride']) ## [H, W, 44] = [heatmap_16; paf_28]

		heatMaps = resizeIntoImgBeforePadded(heatMaps, pad)
		#print 'heatMaps.shape (after inverse pad) = ', heatMaps.shape, '\n'

		heatMaps = cv.resize( heatMaps, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC )
		#print 'heatMaps.shape (after inverse resize)  = ', heatMaps.shape, '\n'

		heatMaps_all.append(copy.deepcopy(heatMaps))

	print '\nUsed multiplier: {}\n'.format(len(heatMaps_all))
	heatMaps_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], heatMaps_all[0].shape[2]))

	if(len(heatMaps_all) == 0):
		return heatMaps_avg

	for i in range(len(heatMaps_all)):
		heatMaps_avg = heatMaps_avg + heatMaps_all[i] / float(len(heatMaps_all))
	return heatMaps_avg

## input:
#		heatMaps : [H, W, N], N = 57 for COCO
def resizeIntoImgBeforePadded(heatMaps, pad):
    #heatMaps = np.transpose(heatMaps, (1, 0, 2)) ## beaause reversed x, y in preprocess
    #print 'pad = ', pad
    if(pad[0] < 0):
        padup = np.zeros((-pad[0],  heatMaps.shape[1], heatMaps.shape[2]))
        heatMaps = np.vstack((padup, heatMaps)) # pad up, because initially is cropped
    elif(pad[0] > 0):
        heatMaps = heatMaps[pad[0]:, :, :]
    
    
    if(pad[1] < 0):
        padleft = np.zeros((heatMaps.shape[0], -pad[1], heatMaps.shape[2]))
        heatMaps = np.hstack((padleft, heatMaps)) # pad left, because initially is cropped
    elif(pad[1] > 0):
        heatMaps = heatMaps[:, pad[1]:, :]
    
    if(pad[2] < 0):
        paddown = np.zeros((-pad[2],  heatMaps.shape[1], heatMaps.shape[2]))
        heatMaps = np.vstack((heatMaps, paddown)) # pad down, because initially is cropped
    elif(pad[2] > 0):
        heatMaps = heatMaps[:-pad[2], :, :] # crop down
        #print 'heatMaps.shape', heatMaps.shape
    
    if(pad[3] < 0):
        padright = np.zeros((heatMaps.shape[0], -pad[3], heatMaps.shape[2]))
        heatMaps = np.hstack((heatMaps, padright)); # pad right, because initially is cropped
    elif(pad[3] > 0):
        heatMaps = heatMaps[:, :-pad[3], :] # crop right
 
    #heatMaps = np.transpose(heatMaps, (1, 0, 2))
  	#print 'debug: heatMaps.shape = ', heatMaps.shape
    return heatMaps


## input:
#		imageToTest: image after pad and normalize
## output: 
#		heatMaps: [heapmap, PAF]
def applyDNN_COCO(imageToTest, net, stride):

	net.blobs['data'].reshape(*(1, imageToTest.shape[2], imageToTest.shape[0], imageToTest.shape[1]))
	net.blobs['data'].data[...] = np.transpose( imageToTest[:,:,:,np.newaxis], (3,2,0,1) ) ## [N, C, H, W]

	net.forward()

	#L1 = net.blobs('Mconv7_stage6_L1').data ## 1x38x46x46
	## Use np.squeeze to remove single-dimensional entries from the shape of an array.
	paf = np.transpose(np.squeeze(net.blobs['Mconv7_stage6_L1'].data), (1,2,0)) ## [H, W, 38]
	paf = cv.resize(paf, (0, 0), fx = stride, fy = stride, interpolation = cv.INTER_CUBIC)

	#L2 = net.blobs('Mconv7_stage6_L2').data ## 1x19x46x46
	heapmap = np.transpose(np.squeeze(net.blobs['Mconv7_stage6_L2'].data), (1,2,0)) ##[H, W, 19]
	heapmap = cv.resize(heapmap, (0, 0), fx = stride, fy = stride, interpolation = cv.INTER_CUBIC)

	heatMaps = np.concatenate((heapmap, paf), axis = 2) ## [H, W, 57]

	return heatMaps

## input:
#		imageToTest: image after pad and normalize
## output: 
#		heatMaps: [heapmap, PAF]
def applyDNN_MPII(imageToTest, net, stride):

	net.blobs['data'].reshape(*(1, imageToTest.shape[2], imageToTest.shape[0], imageToTest.shape[1]))
	net.blobs['data'].data[...] = np.transpose( imageToTest[:,:,:,np.newaxis], (3,2,0,1) ) ## [N, C, H, W]

	net.forward()

	#L1 = net.blobs('Mconv7_stage6_L1').data ## 1x38x46x46
	## Use np.squeeze to remove single-dimensional entries from the shape of an array.
	paf = np.transpose(np.squeeze(net.blobs['Mconv7_stage6_L1'].data), (1,2,0)) ## [H, W, 28]
	paf = cv.resize(paf, (0, 0), fx = stride, fy = stride, interpolation = cv.INTER_CUBIC)

	#L2 = net.blobs('Mconv7_stage6_L2').data ## 1x19x46x46
	heapmap = np.transpose(np.squeeze(net.blobs['Mconv7_stage6_L2'].data), (1,2,0)) ##[H, W, 16]
	heapmap = cv.resize(heapmap, (0, 0), fx = stride, fy = stride, interpolation = cv.INTER_CUBIC)

	heatMaps = np.concatenate((heapmap, paf), axis = 2) ## [H, W, 57]

	return heatMaps

def padRightDownCorner(resizeImg, padValue, stride):
	h = resizeImg.shape[0]
	w = resizeImg.shape[1]

	pad = 4 * [None]
	pad[0] = 0 # up
	pad[1] = 0 # left
	pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
	pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

	img_padded = copy.deepcopy(resizeImg)
	pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
	img_padded = np.concatenate((pad_up, img_padded), axis=0)
	pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
	img_padded = np.concatenate((pad_left, img_padded), axis=1)
	pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
	img_padded = np.concatenate((img_padded, pad_down), axis=0)
	pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
	img_padded = np.concatenate((img_padded, pad_right), axis=1)

	return img_padded, pad

## input:
#		pad img to size bbox
#  output:
#  		img_padded: 
#  		pad:
def padHeight(resizeImg, padValue, stride): ## # pad so that height and width is multiples of stride
	h = resizeImg.shape[0]
	w = resizeImg.shape[1]
	#print 'h, w = ', h, w, '\n'
	bbox = 2 * [None]
	bbox[0] = h if (h%stride == 0) else h + stride - (h%stride)
	bbox[1] = w if (w%stride == 0) else w + stride - (w%stride)
	#print 'bbox = ', bbox, '\n'

	pad = 4 * [0]
	pad[0] = int(math.floor( (bbox[0] - h)/2 )) #up
	pad[1] = int(math.floor( (bbox[1] - w)/2 )) # left
	pad[2] = bbox[0] - h - pad[0] #down
	pad[3] = bbox[1] - w - pad[1] #right
	#print 'pad = ', pad, '\n'

	img_padded = copy.deepcopy(resizeImg) 

	if pad[0] != 0:
		#pad_up = np.tile(img_padded[0,:,:], (pad[0], 1, 1))*0 + padValue # Construct an array by repeating A the number of times given by reps 
		pad_up = np.zeros((pad[0], img_padded.shape[1], img_padded.shape[2])) + padValue
		img_padded = np.vstack((pad_up, img_padded))
	
	if pad[1] != 0:
		#pad_left = np.tile(img_padded[:,0,:], (1, pad[1], 1))*0 + padValue
		pad_left = np.zeros((img_padded.shape[0], pad[1], img_padded.shape[2])) + padValue
		img_padded = np.hstack((pad_left, img_padded))
	
	if pad[2] != 0:
		#pad_down = np.tile(img_padded[-1,:,:], (pad[2], 1, 1))*0 + padValue
		pad_down = np.zeros((pad[2], img_padded.shape[1], img_padded.shape[2])) + padValue
		img_padded = np.vstack((img_padded, pad_down))
	
	if pad[3] != 0:
		#pad_right = np.tile(img_padded[:,-1,:], (1, pad[3], 1))*0 + padValue
		pad_right = np.zeros((img_padded.shape[0], pad[3], img_padded.shape[2])) + padValue
		#print 'pad_right.shape = ', pad_right.shape
		img_padded = np.hstack((img_padded, pad_right))

	return img_padded, pad


## normalize input img,
#
#	output: 
#		img_out: [h, w, 3]
def normalizeImg(img, mean):
	img_out = img / 256.0
	img_out = img_out - mean 
	#img_out = np.transpose(img_out, (1, 0, 2)) ## reverse x, y
	
	if img_out.shape[2] == 1: ## depth = 1
		img_out_bgr = np.zeros((img_out.shape[0], img_out.shape[1], 3))
		img_out_bgr[:, :, 0] = img_out[:, :, 0]
		img_out_bgr[:, :, 1] = img_out[:, :, 0]
		img_out_bgr[:, :, 2] = img_out[:, :, 0]
		return img_out_bgr
	return img_out


## input:
#		img_size: [W, H]
#		x, y: center location
#  output:
#  		label: 2darray
def produceCenterLabelMap(img_size, x, y): # this function is only for center map in testing
	sigma = 21
	[X, Y] = np.meshgrid(range(0, img_size[0]), range(0, img_size[1]))
	X = X - x 
	Y = Y - y
	D2 = X**2 + Y**2 
	Exponent = D2 / 2.0 / sigma / sigma 
	label = np.exp(-Exponent)
	return label 




