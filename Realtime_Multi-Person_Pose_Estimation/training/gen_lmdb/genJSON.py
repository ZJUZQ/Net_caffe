



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
				dist = np.array(prev_center[k][0:2]) - np.array(person_center)
				if np.linalg.norm(dist) < prev_center[k][2] * 0.3 :
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