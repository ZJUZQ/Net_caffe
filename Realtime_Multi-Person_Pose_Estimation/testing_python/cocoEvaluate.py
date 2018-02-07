import os
import numpy as np 
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def evaluate_coco(annJson, resJson, annType):
	cocoGt = COCO(annJson)
	cocoDt = cocoGt.loadRes(resJson)
	with open(resJson, 'r') as f:
		jsonResult = json.load(f)
		imgIds_result = [r['image_id'] for r in jsonResult]

	cocoEval = COCOeval(cocoGt, cocoDt, annType)
	cocoEval.params.imgIds = imgIds_result 
	cocoEval.evaluate() 
	cocoEval.accumulate()
	cocoEval.summarize()

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description = 'evaluate through coco method.')
	parser.add_argument('annJson', type = str,
						help = 'ground truth annotation json file.')
	parser.add_argument('resJson', type = str, 
						help = 'result json file, in coco format.')
	parser.add_argument('--annType', type = str, default = 'keypoints',  
						help = 'annotation type: segm, bbox, or keypoints')

	args = parser.parse_args()
	print('annJson = {}'.format(args.annJson))
	print('resJson = {}'.format(args.resJson))
	print('annType = {}'.format(args.annType))

	evaluate_coco(args.annJson, args.resJson, args.annType)