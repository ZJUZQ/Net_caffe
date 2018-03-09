import sys 
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import argparse

sys.path.append("pycaffe/layers") # the datalayers we will use are in this directory.
sys.path.append("pycaffe") # the tools file is in this folder

## % matplotlib inline
plt.rcParams['figure.figsize'] = (6, 6)


#######################################################################################
"""
Define network prototxts

Let's start by defining the nets using caffe.NetSpec. 
Note how we used the SigmoidCrossEntropyLoss layer. 
This is the right loss for multilabel classification. 
Also note how the data layer is defined.
"""
# helper function for common structures
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

# another helper function
def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

# yet another helper function
def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

# main netspec wrapper
def caffenet_multilabel(module, datalayer, data_layer_params):
    # setup the python data layer 
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module = module, layer = datalayer, 
                               ntop = 2, param_str=str(data_layer_params))
    # module: name of 'pascal_multilabel_datalayers.py', must under $PYTHONPATH
    # layer: class name of layer in module

    # the net itself
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096)
    n.drop6 = L.Dropout(n.relu6, in_place=True)
    n.fc7, n.relu7 = fc_relu(n.drop6, 4096)
    n.drop7 = L.Dropout(n.relu7, in_place=True)
    n.score = L.InnerProduct(n.drop7, num_output=20)
    n.loss = L.SigmoidCrossEntropyLoss(n.score, n.label)
    
    return str(n.to_proto())


#######################################################################################
#
def hamming_distance(gt, est):
	    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))

def check_accuracy(net, num_batches, batch_size = 128):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        ests = net.blobs['score'].data > 0
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)

def predict_result(solver):
	"""
	Look at some prediction results
	"""
	test_net = solver.test_nets[0]
	for image_index in range(5):
	    plt.figure()
	    plt.imshow(transformer.deprocess(copy(test_net.blobs['data'].data[image_index, ...])))
	    gtlist = test_net.blobs['label'].data[image_index, ...].astype(np.int)
	    estlist = test_net.blobs['score'].data[image_index, ...] > 0
	    plt.title('GT: {} \n EST: {}'.format(classes[np.where(gtlist)], classes[np.where(estlist)]))
	    plt.axis('off')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
				usage = '', 
				description = '')
	parser.add_argument('caffe_root', type=str, help='e.g., /home/zq/FPN-caffe/caffe-FP_Net')
	parser.add_argument('pretrained_model_root', type=str, help='e.g., ../pretrained_model')
	
	parser.add_argument('--gpu', type=int, help='use gpu mode, give device number')
	args = parser.parse_args()

	if not os.path.exists('../data/pascal/VOC2012/Annotations'):
		print('Downloading pascal 2012 dataset...')
		os.system('bash ../scripts/get_pascal_2012.sh ../data')

	sys.path.insert(0, os.path.join(args.caffe_root + 'python'))
	import caffe
	from caffe import layers as L, params as P 

	# these are the PASCAL classes, we'll need them later.
	classes = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
						  'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
						  'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 
						  'train', 'tvmonitor'])

	# make sure we have the caffenet weight downloaded.
	if not os.path.exists(os.path.join(args.pretrained_model_root, 'bvlc_reference_caffenet.caffemodel')):
	    print("Downloading pre-trained CaffeNet model...")
	    import download_caffemodel
	    download_caffemodel.download_caffemodel(args.pretrained_model_root)

	# initialize caffe for gpu mode or cpu mode
	if args.gpu:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu)
		print('gpu_mode')
	else:
		caffe.set_mode_cpu()
		print('cpu_mode')


	#####################################################################################
	########################## Define network prototxts #################################
	#####################################################################################
	"""
	Let's start by defining the nets using caffe.NetSpec. Note how we used the SigmoidEntropyLoss 
	layer. This is the right loss for multilabel classification. Also note how the data layer is 
	defined.

	This net use a python datalayer: 'PascalMultilabelDataLayerSync', which is defined in 
	'./pycaffe/layers/pascal_multilabel_datalayers.py'.

	Take a look at the code of python datalayer. It's quite straight-forward, and gives you 
	full control over data and labels.
	"""
	sys.path.append('pycaffe/layers') # add search path of pascal_multilabel_datalayers.py
	
	# write train net.
	with open('model/train_net.prototxt', 'w') as f:
		# provide parameters to the data layer as a python dictionary. Easy as pie!
		data_layer_params = dict(batch_size = 128, 
								 im_shape = [227, 227], 
								 split = 'train', 
								 pascal_root = '../data/pascal/VOC2012')
		f.write(caffenet_multilabel(module='pascal_multilabel_datalayers', 
									datalayer='PascalMultilabelDataLayerSync',
									data_layer_params=data_layer_params))

	# write test net.
	with open('model/test_net.prototxt', 'w') as f:
		data_layer_params = dict(batch_size = 128, 
								 im_shape = [227, 227], 
								 split = 'val', 
								 pascal_root = '../data/pascal/VOC2012')
		f.write(caffenet_multilabel(module='pascal_multilabel_datalayers', 
									datalayer='PascalMultilabelDataLayerSync',
									data_layer_params=data_layer_params))

	# write solver prototxt
	sys.path.append('../scripts')
	import CaffeSolver 
	solverprototxt = CaffeSolver.CaffeSolver(trainnet_prototxt_path = "model/train_net.prototxt", 
									         testnet_prototxt_path = "model/test_net.prototxt")
	solverprototxt.sp['display'] = "10"
	solverprototxt.sp['base_lr'] = "0.0001"
	solverprototxt.sp['snapshot'] = "1000" # We'll snapshot every 2500 iterations
	solverprototxt.sp['snapshot_prefix'] = '"model/multilabel"' # string withing a string, path with prefix to store caffemodel
	solverprototxt.write('model/solver.prototxt')

	#####################################################################################
	################################ Load caffe solver ##################################
	#####################################################################################
	solver = caffe.SGDSolver('model/solver.prototxt')
	solver.net.copy_from(os.path.join(args.pretrained_model_root, 'bvlc_reference_caffenet.caffemodel'))
	solver.test_nets[0].share_with(solver.net)
	solver.step(1)


	#####################################################################################
	############################## Check the loaded data ################################
	#####################################################################################
	"""
	NOTE: we are readin the image from the data layer, so the resolution is lower than the 
	original PASCAL image.
	"""

	import simple_transformer
	transformer = simple_transformer.SimpleTransformer() # This is simply to add back the bias, re-shuffle the color channels to RGB, and so on...
	image_index = 0 # First image in the batch.
	
	plt.figure()
	plt.imshow(transformer.deprocess(copy(solver.net.blobs['data'].data[image_index, ...])))
	gtlist = solver.net.blobs['label'].data[image_index, ...].astype(np.int)
	plt.title('GT: {}'.format(classes[np.where(gtlist)]))
	plt.axis('off')
	plt.savefig('image.png')


	#####################################################################################
	################################## Train a net ######################################
	#####################################################################################
	"""
	Alright, now let's train for a while
	"""
	for it in range(20):
		solver.step(100)
		print('iterations {:4d} \t accuracy: {:.4f}'.format( 100*(it+1), 
															 check_accuracy(solver.test_nets[0], num_batches=50) ))
	
	"""
	Great, the accuray is increasing, and it seems to converge rather quickly. It may seem
	strange that it starts off so high but it is because the ground truth is sparse. There are
	20 classes in PASCAL, and usually only one or two is present. So predicting all zeros yields
	rather high accuray. Let's check to make sure.
	"""
	def check_baseline_accuracy(net, num_batches, batch_size = 128):
	    acc = 0.0
	    for t in range(num_batches):
	        net.forward()
	        gts = net.blobs['label'].data
	        ests = np.zeros((batch_size, len(gts)))
	        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
	            acc += hamming_distance(gt, est)
	    return acc / (num_batches * batch_size)

	print 'Baseline accuracy:{0:.4f}'.format(check_baseline_accuracy(solver.test_nets[0], 5823/128))


	"""
	Look at some prediction results
	"""
	test_net = solver.test_nets[0]
	for image_index in range(5):
		plt.figure()
		plt.imshow(transformer.deprocess(copy(test_net.blobs['data'].data[image_index, ...])))
		gtlist = test_net.blobs['label'].data[image_index, ...].astype(np.int)
		estlist = test_net.blobs['score'].data[image_index, ...] > 0 
		plt.title('GT: {}\nEST: {}'.format(classes[np.where(grlist)], classes[np.where(estlist)]))
		plt.axis('off')
		plt.savefig('test_image{}_result.png'.format(image_index)) 