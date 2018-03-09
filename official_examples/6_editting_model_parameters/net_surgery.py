"""
Caffe networks can be transformed to your particular needs by editing the model parameters.
The data, diffs, and parameters of a net are all exposed in pycaffe.

Roll up your sleeves for net surgery with pycaffe!
"""
import numpy as np 
import matplotlib.pyplot as plt 
import os, sys 
import argparse

# configure plotting
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
				usage = '', 
				description = '')
	parser.add_argument('caffe_root', type=str, help='e.g., /home/zq/FPN-caffe/caffe-FP_Net')
	parser.add_argument('pretrained_model_root', type=str, help='e.g., ../pretrained_model')

	parser.add_argument('--gpu', type=int, help='use gpu mode, give device number')
	args = parser.parse_args()

	sys.path.insert(0, os.path.join(args.caffe_root, 'python/'))
	import caffe 
	from caffe import layers as L, params as P

	if args.gpu:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
	else:
		caffe.set_mode_cpu()

	#########################################################################################
	################################# Designer Filters ######################################
	#########################################################################################
	"""
	To show how to load, manipulate, and save parameters we'll design our own filters into a 
	simple network that's only a single convolution layer. This net has two blobs, data for 
	the input and conv for the convolution output and one parameter conv for the convolution 
	filter weights and biases.
	"""
	net = caffe.Net('model/single_conv.prototxt', caffe.TEST)
	print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))

	# load image and prepare as a single input batch for Caffe
	im = np.array(caffe.io.load_image('cat_gray.jpg', color=False)).squeeze()
	print('im.shape: {}'.format(im.shape))

	im_input = im[np.newaxis, np.newaxis, :, :]
	print('im_input.shape: {}'.format(im_input.shape))
	net.blobs['data'].reshape(*im_input.shape)
	net.blobs['data'].data[...] = im_input

	print net.blobs['conv'].data.min()
	print net.blobs['conv'].data.max()

	"""
	The convolution weights are initialized from Gaussian noise while the biases are 
	initialized to zero. These random filters give output somewhat like edge detections.
	"""
	# helper show filter outputs
	def show_filters(net, imageName):
		net.forward()
		plt.figure()
		filt_min, filt_max = net.blobs['conv'].data.min(), net.blobs['conv'].data.max()
		for i in range(3):
			plt.subplot(1, 4, i+1)
			plt.title("filter #{} output".format(i))
			plt.imshow(net.blobs['conv'].data[0, i], vmin=filt_min, vmax=filt_max)
			plt.tight_layout()
			plt.axis('off')
		plt.savefig(imageName)

	# filter the image with initial 
	show_filters(net, 'filters.png')

	"""
	Raising the bias of a filter will correspondingly raise its output
	"""
	# pick first filter output 
	conv0 = net.blobs['conv'].data[0, 0]
	print('pre-surgery output mean {:.2f}'.format(conv0.mean()))
	# set first filter bias to 1
	net.params['conv'][1].data[0] = 1.
	net.forward()
	print('post-surgery output mean {:.2f}'.format(conv0.mean()))

	"""
	Altering the filter weights is more exciting since we can assign any kernel like 
	Gaussian blur, the Sobel operator for edges, and so on. 

	The following surgery turns the 0th filter into a Gaussian blur and the 1st and 
	2nd filters into the horizontal and vertical gradient parts of the Sobel operator.

	See how the 0th output is blurred, the 1st picks up horizontal edges, and the 2nd 
	picks up vertical edges.
	"""
	ksize = net.params['conv'][0].data.shape[2:]
	# make Gaussian blur
	sigma = 1.
	y, x = np.mgrid[-ksize[0]//2 + 1:ksize[0]//2 + 1, -ksize[1]//2 + 1:ksize[1]//2 + 1]
	g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
	gaussian = (g / g.sum()).astype(np.float32)
	net.params['conv'][0].data[0] = gaussian 	# turns the 0th filter into a Gaussian blur

	# make Sobel operator for edge detection
	net.params['conv'][0].data[1:] = 0.
	sobel = np.array((-1, -2, -1, 0, 0, 0, 1, 2, 1), dtype=np.float32).reshape((3,3))
	net.params['conv'][0].data[1, 0, 1:-1, 1:-1] = sobel  # horizontal
	net.params['conv'][0].data[2, 0, 1:-1, 1:-1] = sobel.T  # vertical
	show_filters(net, 'altered_filters.png')

	"""
	With net surgery, parameters can be transplanted across nets, regularized by custom 
	per-parameter operations, and transformed according to your schemes.
	"""


	#########################################################################################
	############### Casting a Classifier into a Fully Convolutional Network #################
	#########################################################################################
	"""
	Let's take the standard Caffe Reference ImageNet model "CaffeNet" and transform it into a 
	fully convolutional net for efficient, dense inference on large inputs. This model generates 
	a classification map that covers a given input size instead of a single classification. 
	In particular a 8 x 8 classification map on a 451 x 451 input gives 64x the output in
	only 3x the time. The computation exploits a natural efficiency of convolutional network 
	(convnet) structure by amortizing the computation of overlapping receptive fields.

	To do so we translate the InnerProduct matrix multiplication layers of CaffeNet into 
	Convolutional layers. This is the only change: the other layer types are agnostic to 
	spatial size. Convolution is translation-invariant, activations are elementwise operations, 
	and so on. The fc6 inner product when carried out as convolution by fc6-conv turns into 
	a 6 x 6 filter with stride 1 on pool5. Back in image space this gives a classification 
	for each 227 x 227 box with stride 32 in pixels. Remember the equation for output 
	map / receptive field size, output = (input - kernel_size) / stride + 1, and work out the 
	indexing details for a clear understanding.
	"""
	os.system('diff model/bvlc_caffenet_full_conv.prototxt model/bvlc_reference_caffenet.prototxt')

	"""
	The only differences needed in the architecture are to change the fully connected classifier 
	inner product layers into convolutional layers with the right filter size -- 6 x 6, since the 
	reference model classifiers take the 36 elements of pool5 as input -- and stride 1 for dense 
	classification. Note that the layers are renamed so that Caffe does not try to blindly load 
	the old parameters when it maps layer names to the pretrained model.
	"""


	# Load the original network and extract the fully connected layers' parameters.
	net = caffe.Net('model/bvlc_reference_caffenet.prototxt', 
					os.path.join(args.pretrained_model_root, 'bvlc_reference_caffenet.caffemodel'), 
					caffe.TEST)
	params = ['fc6', 'fc7', 'fc8']

	# fc_params = {name: (weights, biases)}
	fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

	for fc in params:
		print '{}: weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

	"""
	Consider the shapes of the inner product parameters. The weight dimensions are the 
	'output x input' sizes while the bias dimension is the output size.
	"""


	# Load the fully convolutional network to transplant the parameters.
	net_full_conv = caffe.Net('model/bvlc_caffenet_full_conv.prototxt', 
	          				  os.path.join(args.pretrained_model_root, 'bvlc_reference_caffenet.caffemodel'),
	          				  caffe.TEST)
	params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-conv']
	# conv_params = {name: (weights, biases)}
	conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

	for conv in params_full_conv:
		print '{}: weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

	"""
	The convolution weights are arranged in 'output x input x height x width' dimensions. 
	
	To map the inner product weights to convolution filters, we could roll the flat inner 
	product vectors into 'channel x height x width' filter matrices, but actually these are 
	identical in memory (as row major arrays) so we can assign them directly.

	The biases are identical to those of the inner product.

	Let's transplant!
	"""
	for pr, pr_conv in zip(params, params_full_conv):
		conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
		conv_params[pr_conv][1][...] = fc_params[pr][1]

	# Next, save the new model weights.
	net_full_conv.save('model/bvlc_caffenet_full_conv.caffemodel')


	#########################################################################################
	###################################### Test #############################################
	#########################################################################################
	"""
	To conclude, let's make a classification map from the example cat image and visualize 
	the confidence of "tiger cat" as a probability heatmap. This gives an 8-by-8 prediction 
	on overlapping regions of the 451 x 451 input.
	"""

	# load input and configure preprocessing
	im = caffe.io.load_image('cat.jpg')
	transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
	transformer.set_mean('data', np.load('ilsvrc_2012_mean.npy').mean(1).mean(1))
	transformer.set_transpose('data', (2,0,1))
	transformer.set_channel_swap('data', (2,1,0))
	transformer.set_raw_scale('data', 255.0)

	# make classification map by forward and print prediction indices at each location
	out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
	print out['prob'][0].argmax(axis=0)
	# show net input and confidence map (probability of the top prediction at each location)
	plt.figure()
	plt.subplot(1, 2, 1)
	plt.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))
	plt.subplot(1, 2, 2)
	plt.imshow(out['prob'][0,281])
	plt.savefig('result.png')

	"""
	In this way the fully connected layers can be extracted as dense features across an 
	image (see net_full_conv.blobs['fc6'].data for instance), which is perhaps more useful 
	than the classification map itself.
	"""