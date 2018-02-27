import scipy.io as sio
import numpy as np
import json
import cv2
import lmdb
import sys, os
# change your caffe path here
sys.path.insert(0, os.path.join('/home/zq/FPN-caffe/caffe-FP_Net', 'python/'))
import caffe
import os.path
import struct
import argparse
from matplotlib.path import Path
sys.path.insert(0, '/home/zq/dataset/COCO/coco/PythonAPI/')
from pycocotools.coco import COCO

## transform coco annotations
## input: 
#		anns: [person_annotation] , person_annotation is the annotation of one person in a image
#  output: 
#		anns_image: [
#					 {'image_id': ''
#					  'annorect': [{person_annotation}, ...]
#					 },
#					 ...
#					]
#		image_annotation is the annotations of one image, containing several persons
def annsTransform(datasetRootPath, annFile):
	coco = COCO(os.path.join(datasetRootPath, annFile)) 			## initialize COCO api for annFile
	annIds = coco.getAnnIds() 		## get all annotation ids
	anns = coco.loadAnns(annIds)

	anns_image = [] 				## output annotations
	imgId_prev = -1
	p_index = 0 					## index of people in a image
	image_index = -1 				## index of image

	for i in range(0, len(anns)):
		print('annsTransform: %d / %d'%(i, len(anns)))
		imgId_curr = anns[i]['image_id']
		
		if imgId_curr == imgId_prev:
			p_index = p_index + 1
			anns_image[image_index]['annorect'].append({}) 			## add one annotation of people for curr image
		else:
			anns_image.append({'image_id': '', 'annorect': [{}]}) 	## add annotation for curr image
			p_index = 0
			image_index = image_index + 1

		anns_image[image_index]['image_id'] = imgId_curr
		anns_image[image_index]['annorect'][p_index]['bbox'] = anns[i]['bbox'] ## [x,y,width,height]
		"""
			The segmentation format depends on whether the instance represents a single object (iscrowd=0 in 
			which case polygons are used) or a collection of objects (iscrowd=1 in which case RLE is used). 
			Note that a single object (iscrowd=0) may require multiple polygons, for example if occluded. 
			Crowd annotations (iscrowd=1) are used to label large groups of objects (e.g. a crowd of people). 
		"""
		anns_image[image_index]['annorect'][p_index]['segmentation'] = anns[i]['segmentation'] 	## RLE or [polygon]
		anns_image[image_index]['annorect'][p_index]['area'] = anns[i]['area'] 					## float
		anns_image[image_index]['annorect'][p_index]['id'] = anns[i]['id'] 						## int
		anns_image[image_index]['annorect'][p_index]['iscrowd'] = anns[i]['iscrowd'] 			## 0 or 1
		"""
			"keypoints" is a length 3k array where k is the total number of keypoints defined for the 
			category. Each keypoint has a 0-indexed location x,y and a visibility flag v defined as 
			v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible. 
			A keypoint is considered visible if it falls inside the object segment. "num_keypoints" indicates 
			the number of labeled keypoints (v>0) for a given object (many objects, e.g. crowds and small objects, 
			will have num_keypoints=0).
		"""
		anns_image[image_index]['annorect'][p_index]['keypoints'] = anns[i]['keypoints'] 			## [x1, y1, v1, ...]
		anns_image[image_index]['annorect'][p_index]['num_keypoints'] = anns[i]['num_keypoints'] 	## int
		anns_image[image_index]['annorect'][p_index]['img_width'] = coco.loadImgs(imgId_curr)[0]['width']
		anns_image[image_index]['annorect'][p_index]['img_height'] = coco.loadImgs(imgId_curr)[0]['height']

		imgId_prev = imgId_curr

	"""
	for i in range(len(anns_image)):
		if len(anns_image[i]['annorect']) != 1:
			print('anns_image[{}] has person number: {}'.format(i, len(anns_image[i]['annorect'])))
	"""
	return (coco, anns_image)


##  obatin the mask images for unlabeled person
##          mask_all:   all segmentations of peoples in a image (num_keypoints >= 0)
##          mask_miss:  those segmentations of peoples in a image with (num_keypoints == 0) 
## generate mask_all and mask_miss for each image 
def writeCOCOMask(coco, annsImage, dataType, year, datasetRootPath):

	for i in xrange(len(annsImage)):
		if dataType == 'val2014':
			img_path  = 'dataset/COCO/images/{0}/COCO_{0}_{1:012d}.jpg'.format(dataType, annsImage[i]['image_id'])
		elif dataType == 'train2017':
			img_path = 'datgaset/COCO/images/{0}/{1:012d}.jpg'.format(dataType, annsImage[i]['image_id'])
		
		img_name1 = 'dataset/COCO/mask{0}/{1}_mask_all_{2:012d}.png'.format(year, dataType, annsImage[i]['image_id'])
		img_name2 = 'dataset/COCO/mask{0}/{1}_mask_miss_{2:012d}.png'.format(year, dataType, annsImage[i]['image_id'])
	
		print('%d / %d' %(i, len(annsImage)))
		if(os.path.isfile(datasetRootPath+img_name1) and os.path.isfile(datasetRootPath+img_name2)):
			continue ## check whether image: img_name1 and img_name2 alerady exist
		else: 
			print('%d / %d' %(i, len(annsImage)))
			ori_image = cv2.imread(os.path.join(datasetRootPath, img_path))
			h, w, c = ori_image.shape
			#cv2.imwrite(os.path.join(datasetRootPath, 'dataset/COCO', img_path.split('/')[-1]), ori_image)

			mask_all = np.zeros((h, w), dtype=np.bool)
			mask_miss = np.zeros((h, w), dtype=np.bool)
			flag = 0 
			for p in range(len(annsImage[i]['annorect'])):
				if(annsImage[i]['annorect'][p]['iscrowd'] == 1): ## iscrowd == 1, segmentation is RLE
					mask_crowd = np.array(coco.decode(annsImage[i]['annorect'][p]['segmentation']), dtype=np.bool)
					temp = np.logical_and(mask_all, mask_crowd)
					mask_crowd = mask_crowd - temp
					flag = flag + 1
					annsImage[i]['mask_crowd'] = mask_crowd
					continue
				else: ## iscrowd == 0
					polygon = annsImage[i]['annorect'][p]['segmentation'][0] ## Note that a single object (iscrowd=0) may require multiple polygons, for example if occluded.
				
				X, Y = np.meshgrid(range(w), range(h))
				X, Y = X.flatten(), Y.flatten()
				points = np.vstack((X, Y)).T
				polygon = np.array(list(polygon))
				polygon = polygon.reshape((-1, 2)) ## with shape Nx2
				#print polygon
				path = Path(polygon)
				mask = path.contains_points(points)
				mask = mask.reshape(h, w) ## binary array
				mask_all = np.logical_or(mask, mask_all)

				if(annsImage[i]['annorect'][p]['num_keypoints'] <= 0):
					mask_miss = np.logical_or(mask, mask_miss)

			if flag == 1:
				## crowd segmentation is added to both mask_miss and mask_all
				mask_miss = np.logical_not(np.logical_or(mask_miss, mask_crowd))
				mask_all = np.logical_or(mask_all, mask_crowd)
			else:
				mask_miss = np.logical_not(mask_miss) 

			annsImage[i]['mask_all'] = mask_all
			annsImage[i]['mask_miss'] = mask_miss 

			img_name = 'dataset/COCO/mask{0}/{1}_mask_all_{2:012d}.png'.format(year, dataType, annsImage[i]['image_id'])
			cv2.imwrite(os.path.join(datasetRootPath, img_name), np.array(mask_all, dtype=np.uint8)*255)
			img_name = 'dataset/COCO/mask{0}/{1}_mask_miss_{2:012d}.png'.format(year, dataType, annsImage[i]['image_id'])
			cv2.imwrite(os.path.join(datasetRootPath, img_name), np.array(mask_miss, dtype=np.uint8)*255)

	return annsImage

## for openpose self validatioin
def getValidationImageIds(annsImage, dataType, validationFile):
	validationImageFile = open(validationFile, 'w')
	for i in range(len(annsImage)):
		if(dataType == 'val2014'):
			if i < 2645:

				validationImageFile.write('COCO_{0}_{1:012d}.jpg\n'.format(dataType, annsImage[i]['image_id']))
			else:
				validationImageFile.close()
				break
		else:
			break

## generate new annotation json file for cpm training
def genJSON(annsImage, dataType, datasetRootPath):

	## In COCO:(1-'nose'	2-'left_eye' 3-'right_eye' 4-'left_ear' 5-'right_ear'
	##          6-'left_shoulder' 7-'right_shoulder'	8-'left_elbow' 9-'right_elbow' 10-'left_wrist'	
	##          11-'right_wrist'	12-'left_hip' 13-'right_hip' 14-'left_knee'	15-'right_knee'	
	##          16-'left_ankle' 17-'right_ankle' )
	
	validationCount = 0
	isValidation = 0
	joint_all = []
	count = 0
	
	for i in range(len(annsImage)):
		numPeople = len(annsImage[i]['annorect'])
		print('prepareJoint: %d/%d (numPeople: %d)\n' %(i, len(annsImage), numPeople))
		prev_center = []

		"""
		if(dataType == 'val2014'):
			if i < 2645:
				validationCount = validationCount + 1 
				print('My validation! %d/2644\n' %i)
				isValidation = 1	
			else:
				isValidation = 0 
		else:
			isValidation = 0 
		"""
		isValidation = 0 

		h = annsImage[i]['annorect'][0]['img_height']
		w = annsImage[i]['annorect'][0]['img_width']

		for p in range(numPeople):
			## skip this person if parts number is too low or if segmentation area is too small
			if annsImage[i]['annorect'][p]['num_keypoints'] < 5 or annsImage[i]['annorect'][p]['area'] < 32*32:
				continue
			## skip this person if the distance to existing person is too small
			person_center = [annsImage[i]['annorect'][p]['bbox'][0] + annsImage[i]['annorect'][p]['bbox'][2]/2.0,
							 annsImage[i]['annorect'][p]['bbox'][1] + annsImage[i]['annorect'][p]['bbox'][3]/2.0]
			flag = 0 
			for k in range(len(prev_center)):
				dist = prev_center[k][0:2] - person_center
				if norm(dist) < prev_center[k][2] * 0.3 :
					flag = 1 
					break 
			if flag == 1:
				continue

			joint_all.append({})
			joint_all[count]['dataset'] = 'COCO_' + dataType
			if dataType in {'train2014', 'val2014'}:
				joint_all[count]['img_paths'] = '{0}/COCO_{0}_{1:012d}.jpg'.format(dataType, annsImage[i]['image_id'])
			elif dataType == 'train2017':
				joint_all[count]['img_paths'] = '{0}/{1:012d}.jpg'.format(dataType, annsImage[i]['image_id'])
			else:
				print('dataType ({}) is now not supported.'.format(dataType))
				sys.exit(0)

			joint_all[count]['isValidation'] = isValidation 
			joint_all[count]['img_width'] = w 
			joint_all[count]['img_height'] = h 
			joint_all[count]['objpos'] = person_center 
			joint_all[count]['image_id'] = annsImage[i]['image_id']
			joint_all[count]['bbox'] = annsImage[i]['annorect'][p]['bbox']
			joint_all[count]['segment_area'] = annsImage[i]['annorect'][p]['area']
			joint_all[count]['num_keypoints'] = annsImage[i]['annorect'][p]['num_keypoints'] 

			anno = annsImage[i]['annorect'][p]['keypoints']
			## set part label: joint_all is (np-3-nTrain)
			## for this very center person 
			joint_all[count]['joint_self'] = np.zeros((17, 3))
			for part in range(17): ## COCO has 17 keypoints annotated
				joint_all[count]['joint_self'][part, 0] = anno[part*3]
				joint_all[count]['joint_self'][part, 1] = anno[part*3+1]
				if(anno[part*3+2] == 2): ## v=2: labeled and visible
					joint_all[count]['joint_self'][part, 2] = 1 
				elif(anno[part*3+2] == 1): ## v=1: labeled but not visible (A keypoint is considered visible if it falls inside the object segment)
					joint_all[count]['joint_self'][part, 2] = 0
				else:
					joint_all[count]['joint_self'][part, 2] = 2
			joint_all[count]['joint_self'] = joint_all[count]['joint_self'].tolist() ## convert the array to a python list
			## set scale
			joint_all[count]['scale_provided'] = annsImage[i]['annorect'][p]['bbox'][3] / 368.0
			#joint_all[count]['scale_provided'] = annsImage[i]['annorect'][p]['area']
			
			## for other person on the same image
			count_other = 0
			joint_all[count]['scale_provided_other'] = []
			joint_all[count]['objpos_other'] = []
			joint_all[count]['bbox_other'] = []
			joint_all[count]['segment_area_other'] = []
			joint_all[count]['num_keypoints_other'] = []
			joint_all[count]['joint_others'] = []

			for op in range(numPeople):
				if (op == p or annsImage[i]['annorect'][op]['num_keypoints'] == 0):
					continue 
				joint_all[count]['scale_provided_other'] += [ annsImage[i]['annorect'][op]['bbox'][3] / 368.0 ] 
				#joint_all[count]['scale_provided_other'] += [ annsImage[i]['annorect'][op]['area'] ]
				joint_all[count]['objpos_other'] += [ [annsImage[i]['annorect'][op]['bbox'][0]+annsImage[i]['annorect'][op]['bbox'][2]/2.0,
													   annsImage[i]['annorect'][op]['bbox'][1]+annsImage[i]['annorect'][op]['bbox'][3]/2.0] ]
				joint_all[count]['bbox_other'] += [ annsImage[i]['annorect'][op]['bbox'] ]
				joint_all[count]['segment_area_other'] += [ annsImage[i]['annorect'][op]['area'] ]
				joint_all[count]['num_keypoints_other'] += [ annsImage[i]['annorect'][op]['num_keypoints'] ]

				anno_other = annsImage[i]['annorect'][op]['keypoints']
				joint_other = np.zeros((17, 3))
				for part in range(17):
					joint_other[part, 0] = anno_other[part*3]
					joint_other[part, 1] = anno_other[part*3+1]

					if(anno_other[part*3+2] == 2): ## v=2: labeled and visible
						joint_other[part, 2] = 1 
					elif(anno[part*3+2] == 1): ## v=1: labeled but not visible (A keypoint is considered visible if it falls inside the object segment)
						joint_other[part, 2] = 0
					else:
						joint_other[part, 2] = 2
				joint_other = joint_other.tolist() ## convert the array to a python list
				joint_all[count]['joint_others'] += [ joint_other ]
			joint_all[count]['annolist_index'] = i 
			joint_all[count]['people_index'] = p 
			joint_all[count]['numOtherPeople'] = len(joint_all[count]['joint_others'])

			prev_center.append( joint_all[count]['objpos'] + [max(annsImage[i]['annorect'][p]['bbox'][2], annsImage[i]['annorect'][p]['bbox'][3])] )
			count = count + 1
			
	json_data = json.dumps(joint_all)
	resJson = os.path.join(datasetRootPath, 'dataset/COCO/json/COCO_') + dataType + '.json'
	fileWriter = open(resJson, 'wb')
	fileWriter.write(json_data)
	fileWriter.close()
	return 'COCO_' + dataType


def writeLMDB(datasets, lmdb_path, year, dataType, datasetRootPath, validation = 0):
	env = lmdb.open(lmdb_path, map_size=int(1e12))
	txn = env.begin(write=True)
	data = []
	numSample = 0

	for d in range(len(datasets)):
		if 'COCO' in datasets[d]:
			print datasets[d]
			with open(os.path.join(datasetRootPath, 'dataset/COCO/json/')+datasets[d]+'.json') as data_file:
				data_this = json.load(data_file)
				#data_this = data_this['root']
				data = data + data_this
			numSample = len(data)
			#print data
			print numSample	

	random_order = np.random.permutation(numSample).tolist() ## Randomly permute a sequence, or return a permuted range
	
	isValidationArray = [data[i]['isValidation'] for i in range(numSample)]; ## validation is used for self validate
	if(validation == 1):  ## totalWriteCount exclude isValidation
		totalWriteCount = isValidationArray.count(0.0)
	else:
		totalWriteCount = len(data)
	print('totalWriteCount = {}'.format(totalWriteCount))
	writeCount = 0

	for count in range(numSample):
		idx = random_order[count]
		if (data[idx]['isValidation'] != 0 and validation == 1):
			print '%d/%d skipped' % (count,idx)
			continue

		img = cv2.imread(os.path.join(datasetRootPath, 'dataset/COCO/images', data[idx]['img_paths']))
		if "COCO" in data[idx]['dataset']:
			mask_all = cv2.imread(os.path.join(datasetRootPath, 'dataset/COCO/mask{0}/{1}_mask_all_{2:012d}.png'.format(year, dataType, data[idx]['image_id'])), 0)
			mask_miss = cv2.imread(os.path.join(datasetRootPath, 'dataset/COCO/mask{0}/{1}_mask_miss_{2:012d}.png'.format(year, dataType, data[idx]['image_id'])), 0)
			#print('mask_all.shape = {}'.format(mask_all.shape))
			#print('mask_miss.shape = {}'.format(mask_miss.shape))

		height = img.shape[0]
		width = img.shape[1]
		if(width < 64):
			img = cv2.copyMakeBorder(img,0,0,0,64-width,cv2.BORDER_CONSTANT,value=(128,128,128))
			print 'saving padded image!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
			cv2.imwrite('padded_img.jpg', img)
			width = 64
			# no modify on width, because we want to keep information
		meta_data = np.zeros(shape=(height,width,1), dtype=np.uint8)
		#print type(img), img.shape
		#print type(meta_data), meta_data.shape
		
		clidx = 0 # current line index, or row index of meta_data
		# dataset name (string)
		for i in range(len(data[idx]['dataset'])):
			meta_data[clidx][i] = ord(data[idx]['dataset'][i])

		clidx = clidx + 1
		# image height, image width
		height_binary = float2bytes(float(data[idx]['img_height']))
		for i in range(len(height_binary)):
			meta_data[clidx][i] = ord(height_binary[i]) ## Given a string of length one, return an integer representing the Unicode code point of the character when the argument is a unicode object, or the value of the byte when the argument is an 8-bit string.
		width_binary = float2bytes(float(data[idx]['img_width']))
		for i in range(len(width_binary)):
			meta_data[clidx][4+i] = ord(width_binary[i])

		clidx = clidx + 1
		# (a) isValidation(uint8), numOtherPeople (uint8), people_index (uint8), annolist_index (float), writeCount(float), totalWriteCount(float)
		meta_data[clidx][0] = data[idx]['isValidation']
		meta_data[clidx][1] = data[idx]['numOtherPeople']
		meta_data[clidx][2] = data[idx]['people_index']
		annolist_index_binary = float2bytes(float(data[idx]['annolist_index']))
		for i in range(len(annolist_index_binary)): # 3,4,5,6
			meta_data[clidx][3+i] = ord(annolist_index_binary[i])
		count_binary = float2bytes(float(writeCount)) # note it's writecount instead of count!
		for i in range(len(count_binary)):
			meta_data[clidx][7+i] = ord(count_binary[i])
		totalWriteCount_binary = float2bytes(float(totalWriteCount))
		for i in range(len(totalWriteCount_binary)):
			meta_data[clidx][11+i] = ord(totalWriteCount_binary[i])
		nop = int(data[idx]['numOtherPeople'])

		clidx = clidx + 1
		# (b) objpos_x (float), objpos_y (float)
		objpos_binary = float2bytes(data[idx]['objpos'])
		for i in range(len(objpos_binary)):
			meta_data[clidx][i] = ord(objpos_binary[i])

		clidx = clidx + 1
		# (c) scale_provided (float)
		scale_provided_binary = float2bytes(data[idx]['scale_provided'])
		for i in range(len(scale_provided_binary)):
			meta_data[clidx][i] = ord(scale_provided_binary[i])

		clidx = clidx + 1
		# (d) joint_self (3*17) (float) (3 line)
		joints = np.asarray(data[idx]['joint_self']).T.tolist() # transpose to 3*17
		for i in range(len(joints)):
			row_binary = float2bytes(joints[i])
			for j in range(len(row_binary)):
				meta_data[clidx][j] = ord(row_binary[j])
			clidx = clidx + 1

		# (e) check nop, prepare arrays
		if(nop!=0):
			if(nop==1):
				joint_other = [data[idx]['joint_others']]
				objpos_other = [data[idx]['objpos_other']]
				scale_provided_other = [data[idx]['scale_provided_other']]
			else:
				joint_other = data[idx]['joint_others']
				objpos_other = data[idx]['objpos_other']
				scale_provided_other = data[idx]['scale_provided_other']
			# (f) objpos_other_x (float), objpos_other_y (float) (nop lines)
			for i in range(nop):
				objpos_binary = float2bytes(objpos_other[i])
				for j in range(len(objpos_binary)):
					meta_data[clidx][j] = ord(objpos_binary[j])
				clidx = clidx + 1
			# (g) scale_provided_other (nop floats in 1 line)
			scale_provided_other_binary = float2bytes(scale_provided_other)
			for j in range(len(scale_provided_other_binary)):
				meta_data[clidx][j] = ord(scale_provided_other_binary[j])
			clidx = clidx + 1
			# (h) joint_others (3*16) (float) (nop*3 lines)
			for n in range(nop):
				joints = np.asarray(joint_other[n]).T.tolist() # transpose to 3*16
				for i in range(len(joints)):
					row_binary = float2bytes(joints[i])
					for j in range(len(row_binary)):
						meta_data[clidx][j] = ord(row_binary[j])
					clidx = clidx + 1
		
		# print meta_data[0:12,0:48] 
		# total 7+4*nop lines
		if "COCO" in data[idx]['dataset']:
			img4ch = np.concatenate((img, meta_data, mask_miss[...,None], mask_all[...,None]), axis=2)
			#img4ch = np.concatenate((img, meta_data, mask_miss[...,None]), axis=2)

		img4ch = np.transpose(img4ch, (2, 0, 1))
		print img4ch.shape
		
		"""
		Converts a 3-dimensional array to datum. If the array has dtype uint8,
		the output data will be encoded as a string. Otherwise, the output data
		will be stored in float format
		"""
		datum = caffe.io.array_to_datum(img4ch, label=0) 
		key = '%07d' % writeCount
		txn.put(key, datum.SerializeToString()) ## txn.put(): Store a record, returning True if it was written, or False to indicate the key was already present and overwrite=False. On success, the cursor is positioned on the new record.
		if(writeCount % 1000 == 0):
			txn.commit()
			txn = env.begin(write=True)
		print '%d/%d/%d/%d' % (count,writeCount,idx,numSample)
		writeCount = writeCount + 1

	txn.commit()
	env.close()

def float2bytes(floats):
	if type(floats) is float:
		floats = [floats]
	return struct.pack('%sf' % len(floats), *floats) ## '3f' == 'fff'; Return a string containing the values v1, v2, ... packed according to the given format. The arguments must match the values required by the format exactly


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'generate lmdb of coco dataset.')
	parser.add_argument('year', type=str, help='e.g., 2014')
	parser.add_argument('annType', type=str, help='instances, captions, or person_keypoints')
	parser.add_argument('dataType', type=str, help='val2014, train2014, train2017 and so on')
	parser.add_argument('datasetRootPath', type=str, help='root path of dataset, e.g., /home/zq/')
	parser.add_argument('lmdbPath', type=str, help='output lmdb path, e.g., /home/zq/dataset/COCO/lmdb')
	args = parser.parse_args()

	annFile = 'dataset/COCO/annotations/{}_{}.json'.format(args.annType, args.dataType)

	if os.path.exists(os.path.join(args.datasetRootPath, 'dataset/COCO/mask') + args.year):
		pass
	else:
		os.mkdir(os.path.join(args.datasetRootPath, 'dataset/COCO/mask') + args.year, 0777) ## store generated mask_full and mask_miss
	if os.path.exists(os.path.join(args.datasetRootPath, 'dataset/COCO/json')):
		pass
	else:
		os.mkdir(os.path.join(args.datasetRootPath, 'dataset/COCO/json'), 0777) ### store transformed json file (contain raw informations needed for training)
	
	coco, annsImage =  annsTransform(args.datasetRootPath, annFile)

	#getValidationImageIds(annsImage, args.dataType, 'validationImageFile.txt')

	annsImage = writeCOCOMask(coco, annsImage, args.dataType, args.year, args.datasetRootPath)

	resJson = genJSON(annsImage, args.dataType, args.datasetRootPath)

	writeLMDB([resJson], args.lmdbPath, args.year, args.dataType, args.datasetRootPath, 0)

