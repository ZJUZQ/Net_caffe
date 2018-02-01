
import argparse
import cv2
import numpy as np
from matplotlib.path import Path
import os, sys
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
	coco = COCO(datasetRootPath + annFile) 			## initialize COCO api for annFile
	annIds = coco.getAnnIds() 		## get all annotation ids
	anns = coco.loadAnns(annIds)

	anns_image = [] 				## output annotations
	imgId_prev = -1
	p_index = 0 					## index of people in a image
	image_index = -1 				## index of image

	for i in range(0, len(anns)):
		print('annsTransform: %d / %d'%(i, len(anns)))
		imgId_curr = anns[i]['image_id']
		if(imgId_curr == imgId_prev):
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

	return (coco, anns_image)


##  obatin the mask images for unlabeled person
##          mask_all:   all segmentations of peoples in a image (num_keypoints >= 0)
##          mask_miss:  those segmentations of peoples in a image with (num_keypoints == 0) 
## generate mask_all and mask_miss for each image 
def writeCOCOMask(coco, annsImage, dataType, year, datasetRootPath):

	for i in range(len(annsImage)):
		img_path  = 'dataset/COCO/images/{0}/COCO_{0}_{1:012d}.jpg'.format(dataType, annsImage[i]['image_id'])
		img_name1 = 'dataset/COCO/mask{0}/{1}_mask_all_{2:012d}.png'.format(year, dataType, annsImage[i]['image_id'])
		img_name2 = 'dataset/COCO/mask{0}/{1}_mask_miss_{2:012d}.png'.format(year, dataType, annsImage[i]['image_id'])
	
		print('%d / %d' %(i, len(annsImage)))
		#mat1 = cv2.imread(datasetRootPath+img_name1)
		#mat2 = cv2.imread(datasetRootPath+img_name2)
		##if(mat1 != NULL and mat2 != NULL):
		if(os.path.isfile(datasetRootPath+img_name1) and os.path.isfile(datasetRootPath+img_name2)):
			continue ## check whether image: img_name1 and img_name2 alerady exist
		else: 
			## cv2.imread() cannot read img_name1 or img_name2
			print('%d / %d' %(i, len(annsImage)))
			h, w, c = cv2.imread(datasetRootPath+img_path).shape
			mask_all = np.zeros((h, w), dtype=np.bool_)
			mask_miss = np.zeros((h, w), dtype=np.bool_)
			flag = 0 
			for p in range(len(annsImage[i]['annorect'])):
				if(annsImage[i]['annorect'][p]['iscrowd'] == 1): ## iscrowd == 1, segmentation is RLE
					mask_crowd = np.array(coco.decode(annsImage[i]['annorect'][p]['segmentation']), dtype=np.bool_)
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

			img_name = 'dataset/COCO/mask{0}/{1}_maks_all_{2:012d}.png'.format(year, dataType, annsImage[i]['image_id'])
			cv2.imwrite(datasetRootPath+img_name, np.array(mask_all, dtype=np.uint8))
			img_name = 'dataset/COCO/maks{0}/{1}_mask_miss_{2:012d}.png'.format(year, dataType, annsImage[i]['image_id'])
			cv2.imwrite(datasetRootPath+img_name, np.array(mask_miss, dtype=np.uint8))

	return annsImage

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
	joint_all = [{}]
	count = 0
	

	for i in range(len(annsImage)):
		numPeople = len(annsImage[i]['annorect'])
		print('prepareJoint: %d/%d (numPeople: %d)\n' %(i, len(annsImage), numPeople))
		prev_center = []

		if(dataType == 'val2014'):
			if i < 2645:
				validationCount = validationCount + 1 
				print('My validation! %d/2644\n' %i)
				isValidation = 1	
			else:
				isValidation = 0 
		else:
			isValidation = 0 

		h = annsImage[i]['annorect'][0]['img_height']
		w = annsImage[i]['annorect'][0]['img_width']

		for p in range(numPeople):
			## skip this person if parts number is too low or if segmentation area is too small
			if annsImage[i]['annorect'][p]['num_keypoints'] < 5 or annsImage[i]['annorect'][p]['area'] < 32*32:
				continue
			## skip this person if the distance to existing person is too small
			person_center = [annsImage[i]['annorect'][p]['bbox'][0]+annsImage[i]['annorect'][p]['bbox'][2]/2.0,
							 annsImage[i]['annorect'][p]['bbox'][1]+annsImage[i]['annorect'][p]['bbox'][3]/2.0]
			flag = 0 
			for k in range(len(prev_center)):
				dist = prev_center[k][0:2] - person_center
				if norm(dist) < prev_center[k][2] * 0.3 :
					flag = 1 
					break 
			if flag == 1:
				continue

			if(dataType == 'train2014'):
				joint_all[count]['dataset'] = 'COCO'
				joint_all[count]['img_paths'] = 'train2014/COCO_train2014_{0:012d}.jpg'.format(annsImage[i]['image_id'])
			else:
				joint_all[count]['dataset'] = 'COCO_val'
				joint_all[count]['img_paths'] = 'val2014/COCO_val2014_{0:012d}.jpg'.format(annsImage[i]['image_id'])

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
				joint_all[count]['joint_others'] += [ joint_other ]
			joint_all[count]['annolist_index'] = i 
			joint_all[count]['people_index'] = p 
			joint_all[count]['numOtherPeople'] = len(joint_all[count]['joint_others'])

			prev_center.append( joint_all[count]['objpos'] + [max(annsImage[i]['annorect'][p]['bbox'][2], annsImage[i]['annorect'][p]['bbox'][3])] )
			count = count + 1
			
	json_data = json.dumps(joint_all)
	fileWriter = open(datasetRootPath+'dataset/COCO/json/COCO.json', 'wb')
	fileWriter.write(json_data)
	fileWriter.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'generate COCO Mask images.')
	parser.add_argument('--year', type=str, default='2014', required=True, help='e.g., 2014 for val2014.')
	parser.add_argument('--annType', type=str, default='person_keypoints', required=True, help='instances, captions, or person_keypoints')
	parser.add_argument('--dataType', type=str, default='val2014', required=True, help='val2014, train2014, and so on')
	parser.add_argument('--datasetRootPath', type=str, default='/home/zq/', required=True, help='root path of dataset')
	args = parser.parse_args()
	annFile = 'dataset/COCO/annotations/{}_{}.json'.format(args.annType, args.dataType)

	if(os.path.exists(args.datasetRootPath+'dataset/COCO/mask2014')):
		pass
	else:
		os.mkdir(args.datasetRootPath+'dataset/COCO/mask2014', 0777) ## store generated mask_full and mask_miss
	if(os.path.exists(args.datasetRootPath+'dataset/COCO/json')):
		pass
	else:
		os.mkdir(args.datasetRootPath+'dataset/COCO/json', 0777) ### store transformed json file (contain raw informations needed for training)
	sys.path.insert(0, args.datasetRootPath+'dataset/COCO/coco/PythonAPI/')

	coco, annsImage =  annsTransform(args.datasetRootPath, annFile)

	#getValidationImageIds(annsImage, args.dataType, 'validationImageFile.txt')

	annsImage = writeCOCOMask(coco, annsImage, args.dataType, args.year, args.datasetRootPath)

	genJSON(annsImage, args.dataType, args.datasetRootPath)

