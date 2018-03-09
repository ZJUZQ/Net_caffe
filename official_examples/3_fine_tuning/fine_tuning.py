"""
In this example, we'll explore a common approach that is particularly useful in 
real-world applications: take a pre-trained Caffe network and fine-tune the parameters 
on your custom data.

The advantage of this approach is that, since pre-trained networks are learned on a 
large set of images, the intermediate layers capture the "semantics" of the general 
visual appearance. Think of it as a very powerful generic visual feature that you can 
treat as a black box. On top of that, only a relatively small amount of data is needed 
for good performance on the target task.
"""
import os, sys
import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import tempfile	# This module generates temporary files and directories.
from copy import copy 

###############################################################################
# Helper function for deprocessing preprocessed images, e.g., for display
def deprocess_net_image(image):
	image = image.copy() 		# don't modify destructively
	image = image[::-1]   # BGR -> RGB
	image = image.transpose(1, 2, 0) # [C, H, W] -> [H, W, C]
	image += [123, 117, 104] # (approximately) undo mean subtraction

	# clamp values in [0, 255]
	image[image < 0], image[image > 255] = 0, 255 

	# round and cast from float32 to uint8 
	image = np.round(image)
	image = np.require(image, dtype=np.uint8)

	return image 

###############################################################################
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2 

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
			  param=learned_param,
			  weight_filler=dict(type='gaussian', std=0.01),
			  bias_filler=dict(type='constant', value=0.1)):
	conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
						 num_output=nout, pad=pad, group=group,
						 param=param,
						 weight_filler=weight_filler,
						 bias_filler=bias_filler)
	return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, 
	        param=learned_param,
	        weight_filler=dict(type='gaussian', std=0.01),
	        bias_filler=dict(type='constant', value=0.1)):
	fc = L.InnerProduct(bottom, num_output=nout,
						param=param,
						weight_filler=weight_filler,
						bias_filler=bias_filler)
	return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
	return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(data, label=None, train=True, num_classes=1000,
			 classifier_name='fc8', learn_all=False):
	"""
	Returns a NetSpec specifying CaffeNet
	"""
	n = caffe.NetSpec()
	n.data = data 
	param = learned_param if learn_all else frozen_param
	n.conv1, n.relu1 = conv_relu(n.data, ks=11, nout=96, stride=4, param=param)
	n.pool1 = max_pool(n.relu1, ks=3, stride=2)
	n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
	n.conv2, n.relu2 = conv_relu(n.norm1, ks=5, nout=256, pad=2, group=2, param=param)
	n.pool2 = max_pool(n.relu2, ks=3, stride=2)
	n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
	n.conv3, n.relu3 = conv_relu(n.norm2, ks=3, nout=384, pad=1, param=param)
	n.conv4, n.relu4 = conv_relu(n.relu3, ks=3, nout=384, pad=1, group=2, param=param)
	n.conv5, n.relu5 = conv_relu(n.relu4, ks=3, nout=256, pad=1, group=2, param=param)
	n.pool5 = max_pool(n.relu5, ks=3, stride=2)
	n.fc6, n.relu6 = fc_relu(n.pool5, nout=4096, param=param)
	if train:
		n.drop6 = fc7_input = L.Dropout(n.relu6, in_place=True)
	else: # for test
		fc7_input = n.relu6 
	n.fc7, n.relu7 = fc_relu(fc7_input, nout=4096, param=param)
	if train:
		n.drop7 = fc8_input = L.Dropout(n.relu7, in_place=True)
	else:
		fc8_input = n.relu7 

	# always learn fc8 (param=learned_param)
	fc8 = L.InnerProduct(fc8_input, num_output=num_classes, param=learned_param)
	# give fc8 the name specified by argument 'classfier_name'
	n.__setattr__(classifier_name, fc8)

	if not train:
		n.probs = L.Softmax(fc8)
	if label is not None:
		n.label = label 
		n.loss = L.SoftmaxWithLoss(fc8, n.label)
		n.acc = L.Accuracy(fc8, n.label)

	"""
	# write the net to a temporary file and return its filename 
	with tempfile.NamedTemporaryFile(delete=False) as f:
		f.write(str(n.to_proto()))
		return f.name 
	"""
	return str(n.to_proto())


"""
Define a function style_net which calls caffenet on data from the Flickr style dataset.

The new network will also have the CaffeNet architecture, with differences in the input 
and output:

	1. the input is the Flickr style data we downloaded, provided by an ImageData layer
	2. the output is a distribution over 20 classes rather than the original 1000 ImageNet 
	   classes
	3. the classification layer is renamed from fc8 to fc8_flickr to tell Caffe not to 
	   load the original classifier (fc8) weights from the ImageNet-pretrained model
"""
def style_net(train=True, learn_all=False, subset=None):
	if subset is None:
		subset = 'train' if train else 'test'
	transform_param = dict(mirror=train, crop_size=227, 
						   mean_file='../data/ilsvrc12/imagenet_mean.binaryproto')
	style_data, style_label = L.ImageData(transform_param=transform_param,
										  source='../data/flickr_style/{}.txt'.format(subset),
										  batch_size=50,
										  new_height=256, new_width=256, 
										  ntop=2)
	return caffenet(data=style_data, label=style_label, train=train, 
		            num_classes=NUM_STYLE_LABELS, 
		            classifier_name='fc8_flickr', 
		            learn_all=learn_all)


###############################################################################
def disp_preds(net, image, labels, k=5, name='ImageNet'):
	#input_blob = net.blobs['data']
	net.blobs['data'].data[0, ...] = image 
	probs = net.forward(start='conv1')['probs'][0]
	top_k = (-probs).argsort()[:k]
	print('top {} predicted {} labels = '.format(k, name))
	for i, p in enumerate(top_k):
		print( '\t({}) {:.2f}% {}'.format(i+1, 100*probs[p], labels[p]) )

def disp_imagenet_preds(net, image, imagenet_labels):
	disp_preds(net, image=image, labels=imagenet_labels, name='ImageNet')

def disp_style_preds(net, image, style_labels):
	disp_preds(net, image=image, labels=style_labels, name='style')



###############################################################################
"""
Now, we'll define a function solver to create our Caffe solvers, which are used to train the 
network (learn its weights). In this function we'll set values for various parameters used 
for learning, display, and "snapshotting" -- see the inline comments for explanations of 
what they mean. You may want to play with some of the learning parameters to see if you can 
improve on the results here!
"""
from caffe.proto import caffe_pb2

def solver(train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    
    s.max_iter = 100000     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 5K iterations -- ten times during training.
    s.snapshot = 5000
    s.snapshot_prefix = 'models'
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    #s.solver_mode = caffe_pb2.SolverParameter.GPU
    s.solver_mode = caffe_pb2.SolverParameter.CPU
    
    """
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name
    """
    return str(s)


"""
For the record, if you want to train the network using only the command line tool, 
this is the command:

	1. build/tools/caffe train \ 
		-solver models/finetune_flickr_style/solver.prototxt \ 
		-weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \ 
		-gpu 0

	2. However, we will train using Python in this example.

	   We'll first define run_solvers, a function that takes a list of solvers and 	
	   steps each one in a round robin manner, recording the accuracy and loss values 
	   each iteration. At the end, the learned weights are saved to a file.
"""
def run_solvers(niter, solvers, disp_interval=10):
	"""
	Run solvers for niter iterations, returning the loss and accuracy recorded each
	iteration.
	'solvers' is a list of (name, solver) tuples
	"""
	blobs = ('loss', 'acc')
	loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
				for _ in blobs)
	for it in range(niter):
		for name, s in solvers:
			s.step(1) # run a single SGD step in Caffe 
			loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
											 for b in blobs)
		if it % disp_interval == 0 or it + 1 == niter:
			loss_disp = '\n'.join('{}: loss = {:.3f}, acc = {:2d}%'.format(name, loss[name][it], int(np.round(100*acc[name][it])))
								  for name, _ in solvers)
			print('iterate: {:3d}\n'.format(it) + loss_disp)

	# Save the learned weights from both nets
	weights = {}
	for name, s in solvers:
		weights[name] = 'model/weights_{}.caffemodel'.format(name)
		s.net.save(weights[name])
	return loss, acc, weights


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

	if args.gpu:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
	else:
		caffe.set_mode_cpu()

	# download pretrained weights
	if not os.path.exists(os.path.join(args.pretrained_model_root, 'bvlc_reference_caffenet.caffemodel')):
		os.path.insert(0, '../scripts')
		import download_model
		download_model.download_caffemodel(args.pretrained_model_root)
	weights = os.path.join(args.pretrained_model_root, 'bvlc_reference_caffenet.caffemodel')
		

	###########################################################################################
	################################ dataset prepare ##########################################
	###########################################################################################
	"""
	We'll download just a small subset of the full dataset for this exercise: just 2000 of the 
	80K images, from 5 of the 20 style categories. (To download the full dataset, 
	set full_dataset = True in the cell below.)
	"""
	full_dataset = False 
	if full_dataset:
		NUM_STYLE_IMAGES = NUM_STYLE_LABELS = -1 
	else:
		NUM_STYLE_IMAGES = 2000 
		NUM_STYLE_LABELS = 5 

	if not os.path.exists('../data/flickr_style/images/train.txt'):
		print('preparing data...')
		os.system('python assemble_data.py -s 1701 -w -1 --images={} --label={}'.format(NUM_STYLE_IMAGES, NUM_STYLE_LABELS))

	if not os.path.exists('../data/ilsvrc12'):
		os.mkdir('./data/ilsvrc12')
	if not os.path.exists('../data/ilsvrc12/synset_words.txt'):
		os.system('bash ../scripts/get_ilsvrc_aux.sh ../data/ilsvrc12') ## downlaod mnist data
	"""
	Load the 1000 ImageNet labels from ilsvrc12/synset_words.txt, and the 5 style labels 
	from style_names.txt.
	"""
	# Load ImageNet labels to imagenet_labels
	imagenet_label_file = os.path.join(os.path.dirname(__file__), '../data/ilsvrc12/synset_words.txt')
	imagenet_labels = list(np.loadtxt(imagenet_label_file, str, delimiter='\t'))
	assert len(imagenet_labels) == 1000 
	print('Loaded ImageNet labels:\n' + '\n'.join(imagenet_labels[:10] + ['...']))

	# Load style labels to style_labels
	style_label_file = './style_names.txt'
	style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
	if NUM_STYLE_LABELS > 0:
		style_labels = style_labels[:NUM_STYLE_LABELS]
	print('\nLoaded style labels:\n' + ', '.join(style_labels))


	###########################################################################################
	######################## Defining and runing the nets #####################################
	###########################################################################################
	"""
	Now, let's create a CaffeNet that takes unlabeled "dummy data" as input, allowing us to set 
	its input images externally and see what ImageNet classes it predicts.
	"""
	dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))
	with open('model/test_net_dummy_data.prototxt', 'w') as f:
		f.write(caffenet(data=dummy_data, train=False))
	imagenet_net = caffe.Net('model/test_net_dummy_data.prototxt', weights, caffe.TEST)

	print('layer_name' + '\t' + 'blob_shape')
	for layer_name, blob in imagenet_net.blobs.iteritems():
		print(layer_name + '\t' + str(blob.data.shape))



	"""
	Use the style_net function defined above to initialize untrained_style_net, a CaffeNet 
	with input images from the style dataset and weights from the pretrained ImageNet model.

	Call forward on untrained_style_net to get a batch of style training data.
	"""
	with open('model/untrained_style_net.prototxt', 'w') as f:
		f.write(style_net(train=False, subset='train'))
	untrained_style_net = caffe.Net('model/untrained_style_net.prototxt', weights, caffe.TEST)
	
	untrained_style_net.forward() # Call forward on untrained_style_net to get a batch of style training data
	style_data_batch = untrained_style_net.blobs['data'].data.copy()
	style_label_batch = np.array(untrained_style_net.blobs['label'].data, dtype=np.int32)

	"""
	Pick one of the style net training images from the batch of 50 (we'll arbitrarily choose #8 
	here). Display it, then run it through imagenet_net, the ImageNet-pretrained network to view 
	its top 5 predicted classes from the 1000 ImageNet classes.

	Below we chose an image where the network's predictions happen to be reasonable, as the image 
	is of a beach, and "sandbar" and "seashore" both happen to be ImageNet-1000 categories. 
	For other images, the predictions won't be this good, sometimes due to the network actually 
	failing to recognize the object(s) present in the image, but perhaps even more often due to 
	the fact that not all images contain an object from the (somewhat arbitrarily chosen) 1000 
	ImageNet categories. Modify the batch_index variable by changing its default setting of 8 to 
	another value from 0-49 (since the batch size is 50) to see predictions for other images in 
	the batch. (To go beyond this batch of 50 images, first rerun the above cell to load a fresh 
	batch of data into style_net.)
	"""
	batch_index = 8 
	image = style_data_batch[batch_index] 
	plt.imsave('training_image_#8.png', deprocess_net_image(image))
	print('actual label = ' + style_labels[style_label_batch[batch_index]])

	# look at imagenet_net's predictions
	disp_imagenet_preds(imagenet_net, image, imagenet_labels)

	# look at untrained_style_net's predictions
	disp_style_preds(untrained_style_net, image, style_labels)

	"""
	We can also verify that the activations in layer fc7 immediately before the classification 
	layer are the same as (or very close to) those in the ImageNet-pretrained model, since both 
	models are using the same pretrained weights in the conv1 through fc7 layers.
	"""
	diff = untrained_style_net.blobs['fc7'].data[0] - imagenet_net.blobs['fc7'].data[0]
	error = (diff ** 2).sum()
	assert error < 1e-8

	# Delete untrained_style_net to save memory.
	# (Hang on to imagenet_net as we will use it again later.)
	del untrained_style_net 
	

	###########################################################################################
	######################## Training the style classifier ####################################
	###########################################################################################
	"""
	Now we'll invoke the solver to train the style net's classification layer.

	Let's create and run solvers to train nets for the style recognition task. We'll create 
	two solvers -- one (style_solver) will have its train net initialized to the 
	ImageNet-pretrained weights (this is done by the call to the copy_from method), and the 
	other (scratch_style_solver) will start from a randomly initialized net.

	During training, we should see that the ImageNet pretrained net is learning faster and 
	attaining better accuracies than the scratch net.
	"""
	niter = 200 # number of iterations to train

	with open('model/train_style_net_with_pretrained.prototxt', 'w') as f:
		f.write(style_net(train=True))
	with open('model/style_solver_with_pretrained.protottxt', 'w') as f:
		f.write(solver(train_net_path='model/train_style_net_with_pretrained.prototxt'))
	
	style_solver_pretrained = caffe.get_solver('model/style_solver_with_pretrained.protottxt')
	style_solver_pretrained.net.copy_from(weights) # pretrained from Imagenet

	# For reference, we also create a solver that isn't initialized from the pretrained
	# ImageNet weights.
	with open('model/train_style_net_without_pretrained.prototxt', 'w') as f:
		f.write(style_net(train=True))
	with open('model/style_solver_without_pretrained.protottxt', 'w') as f:
		f.write(solver(train_net_path='model/train_style_net_without_pretrained.prototxt'))

	style_solver_scratch = caffe.get_solver('model/style_solver_without_pretrained.protottxt')

	print('Running solvers for {} iterations...'.format(niter))
	solvers = [('pretrained', style_solver_pretrained),
		   	   ('scratch', style_solver_scratch)]
	loss, acc, weights = run_solvers(niter, solvers)
	print('Done.')

	train_loss, scratch_train_loss = loss['pretrained'], loss['scratch']
	train_acc, scratch_train_acc = acc['pretrained'], acc['scratch']
	style_weights, scratch_style_weights = weights['pretrained'], weights['scratch']

	# Delete solvers to save memory
	del style_solver_pretrained, style_solver_scratch, solvers 

	"""
	Let's look at the training loss and accuracy produced by the two training procedures. 
	Notice how quickly the ImageNet pretrained model's loss value (blue) drops, and that 
	the randomly initialized model's loss value (green) barely (if at all) improves from 
	training only the classifier layer.
	"""
	plt.figure()
	plt.plot(np.vstack([train_loss, scratch_train_loss]).T)
	plt.xlabel('Iteration #')
	plt.ylabel('Loss')
	plt.savefig('train_loss.png')

	plt.figure()
	plt.plot(np.vstack([train_acc, scratch_train_acc]).T)
	plt.xlabel('Iteration #')
	plt.ylabel('Accuracy')
	plt.savefig('train_accuracy.png')

	"""
	Let's take a look at the testing accuracy after running 200 iterations of training. 
	Note that we're classifying among 5 classes, giving chance accuracy of 20%. We expect 
	both results to be better than chance accuracy (20%), and we further expect the result 
	from training using the ImageNet pretraining initialization to be much better than the 
	one from training from scratch. Let's see.
	"""
	def eval_style_net(weights, test_iters=10):
		if not os.path.exists('model/test_style_net.prototxt'):
			with open('model/test_style_net.prototxt', 'w') as f:
				f.write(style_net(train=False))
		test_net = caffe.Net('model/test_style_net.prototxt', weights, caffe.TEST)
		accuracy = 0
		for it in xrange(test_iters):
			accuracy += test_net.forward()['acc']
		accuracy /= test_iters
		return test_net, accuracy

	test_net, accuracy = eval_style_net(style_weights)
	print 'Test accuracy, trained from ImageNet initialization: %3.1f%%' % (100*accuracy, )
	scratch_test_net, scratch_accuracy = eval_style_net(scratch_style_weights)
	print 'Test accuracy, trained from random initialization: %3.1f%%' % (100*scratch_accuracy, )


	###########################################################################################
	######################## End-to-end finetuning for style ##################################
	###########################################################################################
	"""
	Finally, we'll train both nets again, starting from the weights we just learned. The only 
	difference this time is that we'll be learning the weights "end-to-end" by turning on 
	learning in all layers of the network, starting from the RGB conv1 filters directly applied 
	to the input image. We pass the argument learn_all=True to the style_net function defined 
	earlier in this notebook, which tells the function to apply a positive (non-zero) lr_mult 
	value for all parameters. Under the default, learn_all=False, all parameters in the pretrained 
	layers (conv1 through fc7) are frozen (lr_mult = 0), and we learn only the classifier layer 
	fc8_flickr.

	Note that both networks start at roughly the accuracy achieved at the end of the previous 
	training session, and improve significantly with end-to-end training. To be more scientific, 
	we'd also want to follow the same additional training procedure without the end-to-end 
	training, to ensure that our results aren't better simply because we trained for twice as 
	long. Feel free to try this yourself!
	"""
	with open('model/train_net_style_end_to_end.prototxt', 'w') as f:
		f.write(style_net(train=True, learn_all=True))

	# Set base_lr to 1e-3, the same as last time when learning only the classifier.
	# You may want to play around with different values of this or other
	# optimization parameters when fine-tuning.  For example, if learning diverges
	# (e.g., the loss gets very large or goes to infinity/NaN), you should try
	# decreasing base_lr (e.g., to 1e-4, then 1e-5, etc., until you find a value
	# for which learning does not diverge).
	base_lr = 0.001 

	with open('model/solver_style_end_to_end.prototxt', 'w') as f:
		f.write(solver(train_net_path='model/train_net_style_end_to_end.prototxt', base_lr=base_lr))
	
	style_solver = caffe.get_solver('model/solver_style_end_to_end.prototxt')
	style_solver.net.copy_from(style_weights)

	scratch_style_solver = caffe.get_solver('model/solver_style_end_to_end.prototxt')
	scratch_style_solver.net.copy_from(scratch_style_weights)

	print 'Running solvers for %d iterations...' % niter
	solvers = [('pretrained_end-to-end', style_solver),
	   		   ('scratch_end-to-end', scratch_style_solver)]
	_, _, finetuned_weights = run_solvers(niter, solvers)
	print 'Done.'

	style_weights_ft = finetuned_weights['pretrained_end-to-end']
	scratch_style_weights_ft = finetuned_weights['scratch_end-to-end']

	# Delete solvers to save memory.
	del style_solver, scratch_style_solver, solvers

	"""
	Let's now test the end-to-end finetuned models. Since all layers have been optimized 
	for the style recognition task at hand, we expect both nets to get better results than 
	the ones above, which were achieved by nets with only their classifier layers trained 
	for the style task (on top of either ImageNet pretrained or randomly initialized weights).
	"""
	test_net, accuracy = eval_style_net(style_weights_ft)
	print 'Accuracy, finetuned from ImageNet initialization: %3.1f%%' % (100*accuracy, )
	scratch_test_net, scratch_accuracy = eval_style_net(scratch_style_weights_ft)
	print 'Accuracy, finetuned from   random initialization: %3.1f%%' % (100*scratch_accuracy, )

	"""
	Finally, we'll pick an image from the test set (an image the model hasn't seen) and look 
	at our end-to-end finetuned style model's predictions for it.
	"""
	batch_index = 1
	image = test_net.blobs['data'].data[batch_index]
	plt.imsave('test_image.png', deprocess_net_image(image))
	print 'actual label =', style_labels[int(test_net.blobs['label'].data[batch_index])]
	#  look at the predictions of the network trained from pretrained weights
	disp_style_preds(test_net, image)
	#  look at the predictions of the network trained from scratch
	disp_style_preds(scratch_test_net, image)
	#  look at the ImageNet model's predictions for the above image
	disp_imagenet_preds(imagenet_net, image)










