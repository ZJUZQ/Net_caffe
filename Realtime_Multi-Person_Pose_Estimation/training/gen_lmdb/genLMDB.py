import scipy.io as sio
import numpy as np
import json
import cv2
import lmdb
import sys, os
import os.path
import struct

def writeLMDB(datasets, lmdb_path, year, dataType, validation = 0):
	env = lmdb.open(lmdb_path, map_size=int(1e12))
	txn = env.begin(write=True)
	data = []
	numSample = 0

	for d in range(len(datasets)):
		print datasets[d]
		with open(datasets[d]) as data_file:
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
	parser.add_argument('caffe_root', type=str, help='e.g., /home/zq/caffe/')
	parser.add_argument('year', type=str, help='e.g., 2014 for train2014')
	parser.add_argument('dataType', type=str, help='val2014, train2014, train2017 and so on')
	parser.add_argument('jsonFile', type=str, help='input json file, e.g., /home/zq/dataset/COCO/json/train2017.json')
	parser.add_argument('lmdbPath', type=str, help='output lmdb path, e.g., /home/zq/dataset/COCO/lmdb')
	args = parser.parse_args()

	sys.path.insert(0, os.path.join(args.caffe_root, 'python/'))
	import caffe

	writeLMDB([args.jsonFile], args.lmdbPath, args.year, args.dataType, args.datasetRootPath, 0)

