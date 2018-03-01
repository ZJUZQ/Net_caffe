

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
		if anns[i]['iscrowd'] == 1: ## RLE is used, convert to binary mask
			assert type(anns[i]['segmentation']) == dict 
			anns_image[image_index]['annorect'][p_index]['segmentation'] = coco.annToMask(anns[i])
		else: ## [polygons] are used
			assert type(anns[i]['segmentation']) == list 
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

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'generate lmdb of coco dataset.')
	parser.add_argument('year', type=str, help='e.g., 2014 for train2014')
	parser.add_argument('annType', type=str, help='instances, captions, or person_keypoints')
	parser.add_argument('dataType', type=str, help='val2014, train2014, train2017 and so on')
	parser.add_argument('datasetRootPath', type=str, help='root path of dataset, e.g., /home/zq/')
	args = parser.parse_args()

	annFile = 'dataset/COCO/annotations/{}_{}.json'.format(args.annType, args.dataType)

	if os.path.exists(os.path.join(args.datasetRootPath, 'dataset/COCO/json')):
		pass
	else:
		os.mkdir(os.path.join(args.datasetRootPath, 'dataset/COCO/json'), 0777) ### store transformed json file (contain raw informations needed for training)
	
	coco, annsImage =  annsTransform(args.datasetRootPath, annFile)
