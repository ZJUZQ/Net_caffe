"""
R-CNN is a state-of-the-art detector that classifies region proposals by a finetuned 
Caffe model. For the full details of the R-CNN system and model, refer to its project 
site and the paper:
"""

"""
In this example, we do detection by a pure Caffe edition of the R-CNN model for ImageNet. 
The R-CNN detector outputs class scores for the 200 detection classes of ILSVRC13. 
Keep in mind that these are raw one vs. all SVM scores, so they are not probabilistically 
calibrated or exactly comparable across classes. Note that this off-the-shelf model is 
simply for convenience, and is not the full R-CNN model.
"""
import argparse
import numpy as np 
import os, sys 


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
				usage = '', 
				description = 'example of classification')
	parser.add_argument('caffe_root', type=str, help='e.g., /home/zq/FPN-caffe/caffe-FP_Net')
	parser.add_argument('pretrained_model_root', type=str, help='e.g., ../pretrained_model')

	parser.add_argument('--gpu', type=int, help='use gpu mode, give device number')
	args = parser.parse_args()

	sys.path.insert(0, os.path.join(args.caffe_root, 'python/'))
	import caffe 

	if not os.path.exists(os.path.join(args.pretrained_model_root, 'bvlc_reference_rcnn_ilsvrc13.caffemodel')):
		print('Run ./scripts/download_model_binary.py models/bvlc_reference_rcnn_ilsvrc13 to get the Caffe R-CNN ImageNet model')
		sys.exit(1)

	#######################################################################################
	############################### Selective Search ######################################
	#######################################################################################
	"""
	Selective Search is the region proposer used by R-CNN. The 'selective_search_ijcv_with_python' 
	Python module takes care of extracting proposals through the selective search MATLAB 
	implementation. To install it, download the module and name its directory 
	selective_search_ijcv_with_python, run the demo in MATLAB to compile the necessary 
	functions, then add it to your PYTHONPATH for importing. 

	(If you have your own region proposals prepared, or would rather not bother with this 
	step, 'detect.py' accepts a list of images and bounding boxes as CSV.)
	"""



	"""
	With that done, we'll call the bundled 'detect.py' to generate the region proposals 
	and run the network. For an explanation of the arguments, do './detect.py --help'
	"""
	os.system('echo `pwd`/fish-bike.jpg > det_input.txt')
	os.system( 'python {} --crop_mode=selective_search \
				   		  --pretrained_model={} \
				          --model_def=model/deploy.prototxt \
				          --raw_scale=255 \
				          det_input.txt \
				          det_output.h5'.format(os.path.join(args.caffe_root, 'python/detect.py'),
				   	                            os.path.join(args.pretrained_model_root, 'bvlc_reference_rcnn_ilsvrc13.caffemodel')) )