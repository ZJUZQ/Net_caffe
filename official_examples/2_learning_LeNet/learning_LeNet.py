import os, sys
import numpy as np 
import argparse
import matplotlib.pyplot as plt

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

"""
This network expects to read from pregenerated LMDBs, but reading directly 
from ndarrays is also possible using MemoryDataLayer.
"""
def create_lenet(lmdb, batch_size):
	n = caffe.NetSpec()
	n.data, n.label = L.Data( batch_size=batch_size, backend=P.Data.LMDB, source= lmdb,
							  transform_param=dict(scale=1./255), ntop=2 )
	n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
	n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	n.fc1 = L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
	n.relu1 = L.ReLU(n.fc1, in_place=True)
	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
	n.loss = L.SoftmaxWithLoss(n.score, n.label)
	return n.to_proto()

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
				usage = 'python learning_LeNet.py /home/zq/FPN-caffe/caffe-FP_Net ../pretrained_model', 
				description = 'example of learning LeNet')
	parser.add_argument('caffe_root', type=str, help='e.g., /home/zq/FPN-caffe/caffe-FP_Net')
	parser.add_argument('pretrained_model_root', type=str, help='e.g., ../pretrained_model')

	parser.add_argument('--gpu', type=int, help='use gpu mode, give device number')
	args = parser.parse_args()

	sys.path.insert(0, os.path.join(args.caffe_root, 'python/'))
	import caffe 
	from caffe import layers as L, params as P

	if not os.path.exists('../data/mnist'):
		os.mkdir('../data/mnist')
	if not os.path.exists('../data/mnist/train-images-idx3-ubyte'):
		os.system('bash ../scripts/get_mnist.sh ../data/mnist') ## downlaod mnist data

	# converts the mnist data into lmdb format
	os.system('bash ../scripts/create_mnist.sh {}'.format(args.caffe_root))

	
	#####################################################################################
	############################ Creating the net prototxt ##############################
	#####################################################################################
	"""
	We start by creating the net. We'll write the net in a succinct and natural way as Python 
	code that serializes to Caffe's protobuf model format.
	"""
	if not os.path.exists('./model'):
		os.mkdir('./model')

	# write train net prototxt
	with open('model/lenet_train.prototxt', 'w') as f:
		f.write(str(create_lenet('./mnist_train_lmdb', 64)))

	# write test net prototxt
	with open('model/lenet_test.prototxt', 'w') as f:
		f.write(str(create_lenet('./mnist_test_lmdb', 100)))

	# write solver prototxt
	sys.path.append('../scripts')
	import CaffeSolver 
	solverprototxt = CaffeSolver.CaffeSolver(trainnet_prototxt_path = 'model/lenet_train.prototxt', 
									   		 testnet_prototxt_path = 'model/lenet_test.prototxt')
	solverprototxt.sp['display'] = "100" # Display every 100 iterations
	solverprototxt.sp['base_lr'] = "0.01"
	solverprototxt.sp['gamma'] = "0.0001"
	solverprototxt.sp['lr_policy'] = '"inv"'
	solverprototxt.sp['test_interval'] = "500"
	solverprototxt.write('model/solver.prototxt')

	
	#####################################################################################
	########################## Loading and checking the solver ##########################
	#####################################################################################
	if args.gpu:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
	else:
		caffe.set_mode_cpu()

	#solver = None 
	solver = caffe.SGDSolver('model/solver.prototxt')

	"""
    To get an idea of the architecture of our net, we can check the dimensions of the 
    intermediate features (blobs) and parameters (these will also be useful to refer to 
    when manipulating data later).
	"""
	print('\nlayer_name' + '\t' + 'blob_shape')
	for k, v in solver.net.blobs.items():
		print(k + '\t' + str(v.data.shape))

	print('\nlayer_name' + '\t' + 'weight_shape' + '\t' + 'bias_shape')
	for k, v in solver.net.params.items():
		print(k + '\t' + str(v[0].data.shape) + '\t' + str(v[1].data.shape))

	"""
	Before taking off, let's check that everything is loaded as we expect. We'll run a 
	forward pass on the train and test nets and check that they contain our data.
	"""
	solver.net.forward() # train net
	print solver.net.blobs['loss'].data
	solver.test_nets[0].forward() # test net (there can be more than one test net)
	print solver.test_nets[0].blobs['loss'].data

	# we use a little trick to tile the first eight images
	plt.imsave( 'train_digits.png', 
				solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28),
				cmap='gray' )
	print('train labels: {}'.format(solver.net.blobs['label'].data[:8]))

	plt.imsave( 'test_digits.png', 
				solver.test_nets[0].blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8*28),
				cmap='gray' )
	print('test labels: {}'.format(solver.test_nets[0].blobs['label'].data[:8]))


	#####################################################################################
	############################## Stepping the solver ##################################
	#####################################################################################
	"""
	Both train and test nets seem to be loading data, and to have correct labels.

    Let's take one step of (minibatch) SGD and see what happens
	"""
	solver.step(1)
	# Do we have gradients propagating through our filters? Let's see the updates to the first layer
	print( 'solver.net.params[\'conv1\'][0].data.shape = {}'.format(solver.net.params['conv1'][0].data.shape) )
	print( 'solver.net.params[\'conv1\'][0].diff.shape = {}'.format(solver.net.params['conv1'][0].diff.shape) )
	
	## params: [weight, bias]
	vis_square(solver.net.params['conv1'][0].diff.transpose(0, 2, 3, 1)[:,:,:,0], 'conv1_kernels_diff.png' ) 
	vis_square(solver.net.params['conv1'][0].data.transpose(0, 2, 3, 1)[:,:,:,0], 'conv1_kernels_data.png' )  
	vis_square(solver.net.blobs['conv1'].data[0], 'conv1_features_data.png' )  


	#####################################################################################
	####################### Writing a custom training loop ##############################
	#####################################################################################
	"""
	Something is happening. Let's run the net for a while, keeping track of a few things as 
	it goes. Note that this process will be the same as if training through the caffe binary. 
	In particular:

    	> logging will continue to happen as normal
    	> snapshots will be taken at the interval specified in the solver prototxt (here, 
    	  every 5000 iterations)
    	> testing will happen at the interval specified (here, every 500 iterations)

	Since we have control of the loop in Python, we're free to compute additional things 
	as we go, as we show below. We can do many other things as well, for example:

    	> write a custom stopping criterion
    	> change the solving process by updating the net in the loop
	"""
	niter = 200 
	test_interval = 25 
	# losses will also be stored in the log
	train_loss = np.zeros(niter)
	test_acc = np.zeros( int(np.ceil(niter / test_interval)) )
	output = np.zeros((niter, 8, 10))

	for it in range(niter):
		solver.step(1) # SGD by caffe
		train_loss[it] = solver.net.blobs['loss'].data 

		# store the output on the first test batch 
		# (start the forward pass at conv1 to avoid loading new data)
		solver.test_nets[0].forward(start='conv1')
		output[it] = solver.test_nets[0].blobs['score'].data[:8]

		# run a full test every so often
		# (Caffe can also do this for us and write to a log, but we show here
		#  how to do it directly in Python, where more complicated things are easier.)
		if (it % test_interval) == 0:
			print('Iteration {} testing...'.format(it))
			correct = 0 
			for test_it in range(100):
				solver.test_nets[0].forward()
				correct += sum( solver.test_nets[0].blobs['score'].data.argmax(1) 
								== solver.test_nets[0].blobs['label'].data )
			test_acc[it // test_interval] = float(correct) / 1e4 ## 100 * test_batching

	# Let's plot the train loss and test accuracy
	_, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(np.arange(niter), train_loss)
	ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
	ax1.set_xlabel('iteration')
	ax1.set_ylabel('train loss')
	ax2.set_ylabel('test accuracy')
	ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
	plt.savefig('loss_accuracy.png')

	"""
	Since we saved the results on the first test batch, we can watch how our prediction 
	scores evolved. We'll plot time on the xx axis and each possible label on the yy, with 
	lightness indicating confidence.
	"""
	for i in range(8):
		plt.figure(figsize=(4, 4))
		plt.imsave('test_image_{}.png'.format(i), solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
		plt.figure(figsize=(10, 4))
		plt.imsave('test_image_{}_prediction_scores.png'.format(i), output[:50, i].T, cmap='gray')

	"""
	Note that these are the "raw" output scores rather than the softmax-computed probability 
	vectors. The latter, shown below, make it easier to see the confidence of our net (but 
	harder to see the scores for less likely digits).
	"""
	for i in range(8):
		plt.figure(figsize=(10, 4))
		plt.imsave('test_image_{}_softmax_prediction_scores.png'.format(i), output[:50, i].T, cmap='gray')


	#####################################################################################
	####################### Writing a custom training loop ##############################
	#####################################################################################
	"""
	Now that we've defined, trained, and tested LeNet there are many possible next steps:

    	* Define new architectures for comparison
    	* Tune optimization by setting base_lr and the like or simply training longer
    	* Switching the solver type from SGD to an adaptive method like AdaDelta or Adam

	Feel free to explore these directions by editing the all-in-one example that follows. 
	Look for "EDIT HERE" comments for suggested choice points.

	By default this defines a simple linear classifier as a baseline.

	In case your coffee hasn't kicked in and you'd like inspiration, try out

    	1. Switch the nonlinearity from ReLU to ELU or a saturing nonlinearity like Sigmoid
    	2. Stack more fully connected and nonlinear layers
    	3. Search over learning rate 10x at a time (trying 0.1 and 0.001)
    	4. Switch the solver type to Adam (this adaptive solver type should be less sensitive 
    	   to hyperparameters, but no guarantees...)
    	5. Solve for longer by setting niter higher (to 500 or 1,000 for instance) to better 
    	   show training differences
	"""
	