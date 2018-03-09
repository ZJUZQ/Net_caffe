import os, sys
import numpy as np
import argparse
import matplotlib.pyplot as plt

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

"""
Since we are dealing with four-dimensional data here, we'll define a helper function 
for visualizing sets of rectangular heatmaps.
"""
def vis_square(data, imgName):
	"""
	Take an array of shape (n, height, width) or (n, height, width, 3)
    and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    """
    # normalize data for display 
   	data = (data - data.min()) / (data.max() - data.min())
    # force the number of filters to be square 
   	n = int(np.ceil(np.sqrt(data.shape[0])))
   	padding = ( ((0, n ** 2 - data.shape[0]),
                (0, 1), 
                (0, 1))                 		 # add some space between filters
                + ((0, 0),) * (data.ndim - 3) )  # don't pad the last dimension (if there is one)
   	data = np.pad(data, padding, mode='constant', constant_values=1) # pad with ones (white)

   	# tile the filters into an image 
   	data = data.reshape( (n, n) + data.shape[1:] ).transpose( (0, 2, 1, 3) + tuple(range(4, data.ndim+1)) )
   	data = data.reshape( (n * data.shape[1], n * data.shape[3]) + data.shape[4:] )
   	plt.imsave(imgName, data)


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


	##########################################################################
	###################### setup input transformer ###########################
	##########################################################################
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
	mu = np.load(os.path.join(args.caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')) ## a single array is returned, [C, H, W]
	mu = mu.mean(1).mean(1) ## averate over pixels to obtain the mean (BGR) pixel values
	print('mean-subtracted values: ', zip('BGR', mu))

	## create transformer for the input called 'data'
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	print net.blobs['data'].data.shape
	transformer.set_transpose('data', (2, 0, 1)) ## [H, W, C] -> [C, H, W]
	transformer.set_mean('data', mu) ## subtract the dataset-mean value in each channel
	transformer.set_raw_scale('data', 255) ## rescale from [0, 1] to [0, 255]
	transformer.set_channel_swap('data', (2, 1, 0)) ## swap channels from RGB to BGR


	######################################################################
	########################## CPU classification ########################
	######################################################################
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
	#print type(transformed_image) ## numpy.ndarray
	
	# copy the image data into the memory allocated for the net
	net.blobs['data'].data[...] = transformed_image	## assign transformed_image to each image of the batching
	#print type(net.blobs['data'].data[...])	## numpy.ndarray

	### perform classification
	output = net.forward()
	output_prob = output['prob'][0]  # the output probability vector for the first image in the batch (50)
	#print type(output['prob'])
	print('predicted class is: {}'.format(output_prob.argmax()))

	# load ImageNet labels
	labels_file = '../data/ilsvrc12/synset_words.txt'
	if not os.path.exists(labels_file):
		os.system('bash ../scripts/get_ilsvrc_aux.sh ../data/ilsvrc12')
	labels = np.loadtxt(labels_file, str, delimiter='\t')
	#print labels.shape
	print 'output label: ', labels[output_prob.argmax()]

	# sort top five predictions from softmax output 
	top_inds = output_prob.argsort()[:5] # reverse sort and take five largest items
	print 'probabilities and labels:\n', zip(output_prob[top_inds], labels[top_inds]) 


	######################################################################
	################## Examining intermediate output #####################
	######################################################################
	"""
    A net is not just a black box; let's take a look at some of the parameters and 
    intermediate activations.

	First we'll see how to read out the structure of the net in terms of activation 
	and parameter shapes.

    For each layer, let's look at the activation shapes, which typically have the 
    form (batch_size, channel_dim, height, width). The activations are exposed as an 
    OrderedDict, net.blobs.
	"""
	print('\nlayer_name' + '\t' + 'activation shape')
	for layer_name, blob in net.blobs.iteritems():
		print(layer_name + '\t' + str(blob.data.shape))

	"""
    Now look at the parameter shapes. The parameters are exposed as another OrderedDict, 
    net.params. We need to index the resulting values with either [0] for weights or [1] for biases.

    The param shapes typically have the form (output_channels, input_channels, filter_height, filter_width) 
    (for the weights) and the 1-dimensional shape (output_channels,) (for the biases).
	"""
	print('\nlayer_name' + '\t' + 'weight param shape' + '\t' + 'bias param shape')
	for layer_name, param in net.params.iteritems():
		print(layer_name + '\t' + str(param[0].data.shape) + '\t' + str(param[1].data.shape))

	"""
	visualize heatmaps
	"""
    # first layer, the parameters are a list of [weights, biases]
   	filters = net.params['conv1'][0].data
   	vis_square(filters.transpose(0, 2, 3, 1), './conv1_weights.png')

   	# first layer output (rectified responses of the filters above, first 36 only)
   	feat = net.blobs['conv1'].data[0, :36] ## first image in the batching
   	vis_square(feat, './conv1_blob.png')

   	# fully connected layer, fc6 (rectified), show the output values and the histogram of positive values
   	feat = net.blobs['fc6'].data[0] 
   	plt.figure(figsize=(15, 8))
   	plt.subplot(2, 1, 1)
   	plt.plot(feat.flat)
   	plt.subplot(2, 1, 2)
   	plt.hist(feat.flat[feat.flat > 0], bins=100)
   	plt.savefig('./fc6_blob.png')

   	# the final probability output, prob 
   	feat = net.blobs['prob'].data[0] 
   	plt.figure(figsize=(15, 3))
   	plt.plot(feat.flat)
   	plt.savefig('./prob_blob.png')









