


##  obatin the mask images for unlabeled person
##          mask_all:   all segmentations of peoples in a image (num_keypoints >= 0)
##          mask_miss:  those segmentations of peoples in a image with (num_keypoints == 0) 
## generate mask_all and mask_miss for each image 
def writeCOCOMask(coco, annsImage, dataType, year, datasetRootPath):

	for i in xrange(len(annsImage)):
		if dataType == 'val2014':
			img_path  = 'dataset/COCO/images/{0}/COCO_{0}_{1:012d}.jpg'.format(dataType, annsImage[i]['image_id'])
		elif dataType == 'train2017':
			img_path = 'dataset/COCO/images/{0}/{1:012d}.jpg'.format(dataType, annsImage[i]['image_id'])
		
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
				if annsImage[i]['annorect'][p]['iscrowd'] == 1: 
					## iscrowd == 1, segmentation is RLE, and has been converted to binary mask
					mask_crowd = np.array(annsImage[i]['annorect'][p]['segmentation'], dtype=np.bool)
					temp = np.logical_and(mask_all, mask_crowd)
					mask_crowd = mask_crowd - temp
					flag = flag + 1
					annsImage[i]['mask_crowd'] = mask_crowd
					continue
				## when iscrowd == 0, polygons is used
				X, Y = np.meshgrid(range(w), range(h))
				X, Y = X.flatten(), Y.flatten()
				points = np.vstack((X, Y)).T

				for ii in range(len(annsImage[i]['annorect'][p]['segmentation'])):
					seg = annsImage[i]['annorect'][p]['segmentation'][ii]
					polygon = np.array(list(seg))
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