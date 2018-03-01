import os, sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
# display plots in this notebook
## %matplotlib inline

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap



if __name__ == "__main__":
	parser = argparse.ArgumentParser(
				usage = '', 
				description = 'example of classification')
	parser.add_argument('caffe_root', type=str, help='e.g., /home/zq/FPN-caffe/caffe-FP_Net/python')
	parser.add_argument('pretrained_model_root', type=str, help='e.g., ../pretrained_model/')

	parser.add_argument('--nets_folder', type=str, default='./pascal_multilabel_with_datalayer', help='folder to store train.prototxt, val.prototxt, solver.prototxt')
	parser.add_argument('--gpu', type=int, help='use gpu mode, give device number')
	args = parser.parse_args()

	sys.path.insert(0, os.path.join(args.caffe_root, 'python'))
	import caffe 

	if os.path.isfile(os.path.join(args.pretrained_model_root, 'bvlc_reference_caffenet.caffemodel')):
		print('CaffeNet found.')
	else:
		print('Downloading pre-trained CaffeNet model...')
		from ..scripts import download_model
		download_model.download_caffemodel(args.pretrained_model_root)

	# initialize caffe for gpu mode or cpu mode
	if args.gpu:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
		print('using gpu_mode')
	else:
		caffe.set_mode_cpu()
		print('using cpu_mode')

	model_def = './model_proto/test_net.prototxt'
	model_weights = os.path.join(args.pretrained_model_root, 'bvlc_reference_caffenet.caffemodel')

	net = caffe.Net(model_def, ## defines the structure of the model
					model_weights, ## contains the trained weights
					caffe.TEST ## use test mode (e.g., don't perform dropout)
					)


	
	#################### setup input preprocessing ###########################
	"""
    Set up input preprocessing. 
    (We'll use Caffe's caffe.io.Transformer to do this, but this step is independent of 
    other parts of Caffe, so any custom preprocessing code may be used).

    Our default CaffeNet is configured to take images in BGR format. Values are expected 
    to start in the range [0, 255] and then have the mean ImageNet pixel value subtracted 
    from them. In addition, the channel dimension is expected as the first (outermost) 
    dimension.

    As matplotlib will load images with values in the range [0, 1] in RGB format with the 
    channel as the innermost dimension, we are arranging for the needed transformations here.
	"""

	## load the mean ImageNet image (as distributed with Caffe) for subtraction
	mu = np.load('./ilsvrc_2012_mean.npy') ## a single array is returned, [C, H, W]
	mu = mu.mean(1).mean(1) ## averate over pixels to obtain the mean (BGR) pixel values
	print('mean-subtracted values: ', zip('BGR', mu))

	## create transformer for the input called 'data'
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2, 0, 1)) ## [H, W, C] -> [C, H, W]
	transformer.set_mean('data', mu) ## subtract the dataset-mean value in each channel
	transformer.set_raw_scale('data', 255) ## rescale from [0, 1] to [0, 255]
	transformer.set_channel_swap('data', (2, 1, 0)) ## swap channels from RGB to BGR

	########################## CPU classification ########################
	"""
	Now we're ready to perform classification. Even though we'll only classify one image, 
	we'll set a batch size of 50 to demonstrate batching.
	"""
	
	# set the size of the input (we can skip this if we're happy
	#  with the default; we can also change it later, e.g., for different batch sizes)
	net.blobs['data'].reshape(50,        # batch size
                          	  3,         # 3-channel (BGR) images
                              227, 227)  # image size is 227x227

	## Load an image (that comes with Caffe) and perform the preprocessing we've set up.
	image = caffe.io.load_image('./cat.jpg')	## return an image [H, W, C] with type np.float32 in range [0, 1]
	transformed_image = transformer.preprocess('data', image) ## get [C, H, W] ndarray for input to a Net

	# copy the image data into the memory allocated for the net
	print transformed_image.shape
	net.blobs['data'].data[...] = transformed_image
	print net.blobs['data'].data[...]

	"""
	### perform classification
	output = net.forward()
	output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
	print('predicted class is: {}'.format(output_prob.argmax()))
	"""










