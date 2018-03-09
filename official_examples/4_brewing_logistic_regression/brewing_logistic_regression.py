"""
While Caffe is made for deep networks it can likewise represent "shallow" models like 
logistic regression for classification. We'll do simple logistic regression on synthetic 
data that we'll generate and save to HDF5 to feed vectors to Caffe. Once that model is 
done, we'll add layers to improve accuracy. That's what Caffe is about: define a model, 
experiment, and then deploy.
"""
import numpy as np 
import matplotlib.pyplot as plt 
import os, sys 
import h5py 

import sklearn
import sklearn.datasets
import sklearn.linear_model

import pandas as pd 
import argparse



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


	######################################################################################
	########################### scikit-learn logistic regression #########################	
	######################################################################################
	"""
	Synthesize a dataset of 10,000 4-vectors for binary classification with 2 informative 
	features and 2 noise features.
	"""
	X, y = sklearn.datasets.make_classification(n_samples=10000,
												n_features=4,
												n_redundant=0,
												n_informative=2,
												n_clusters_per_class=2,
												hypercube=False,
												random_state=0)
	# Split into train and test
	X, Xt, y, yt = sklearn.model_selection.train_test_split(X, y)

	"""
	# Visualize sample of the data 
	ind = np.random.permutation(X.shape[0])[:1000]
	df = pd.DataFrame(X[ind])
	_ = pd.plotting.scatter_matrix(df, figsize=(9, 9), diagonal='kde', marker='o', s=40, alpha=.4, c=y[ind])
	"""

	"""
	Learn and evaluate scikit-learn's logistic regression with stochastic gradient descent 
	(SGD) training. Time and check the classifier's accuracy.
	"""
	# Train and test the scikit-learn SGD logistic regression.
	clf = sklearn.linear_model.SGDClassifier(loss='log', n_iter=1000, penalty='l2', 
											 alpha=5e-4, class_weight='balanced')

	clf.fit(X, y)
	yt_pred = clf.predict(Xt)
	print('Scikit-learn accuracy: {:.3f}'.format(sklearn.metrics.accuracy_score(yt, yt_pred)))


	######################################################################################
	################# Save the dataset to HDF5 for loading in Caffe ######################	
	######################################################################################
	dirname = os.path.abspath('./hdf5_data')
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	train_filename = os.path.join(dirname, 'train.h5')
	test_filename = os.path.join(dirname, 'test.h5')

	# HDF5DataLayer source should be a file containing a list of HDF5 filenames.
	# To show this off, we'll list the same data file twice.
	with h5py.File(train_filename, 'w') as f:
		f['data'] = X 
		f['label'] = y.astype(np.float32)
	with open(os.path.join(dirname, 'train.txt'), 'w') as f:
		f.write(train_filename + "\n")
		f.write(train_filename + "\n")

	# HDF5 is pretty efficient, but can be further compressed.
	comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
	with h5py.File(test_filename, 'w') as f:
		f.create_dataset('data', data=Xt, **comp_kwargs)
		f.create_dataset('label', data=yt.astype(np.float32), **comp_kwargs)
	with open(os.path.join(dirname, 'test.txt'), 'w') as f:
		f.write(test_filename + '\n')

	######################################################################################
	##################### define logistic regression in Caffe ############################	
	######################################################################################
	"""
	Let's define logistic regression in Caffe through Python net specification. This is a 
	quick and natural way to define nets that sidesteps manually editing the protobuf model.
	"""
	def logreg(hdf5_file, batch_size):
		# logistic regression: data, matrix multiplication, and 2-class softmax loss
		n = caffe.NetSpec()
		n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5_file, ntop=2)
		n.ip1 = L.InnerProduct(n.data, num_output=2, weight_filler=dict(type='xavier'))
		n.accuracy = L.Accuracy(n.ip1, n.label)
		n.loss = L.SoftmaxWithLoss(n.ip1, n.label)
		return n.to_proto()

	train_net_path = 'model/train_logreg.prototxt'
	with open(train_net_path, 'w') as f:
		f.write(str(logreg('hdf5_data/train.txt', batch_size=10)))

	test_net_path = 'model/test_logreg.prototxt'
	with open(test_net_path, 'w') as f:
		f.write(str(logreg('hdf5_data/test.txt', batch_size=10)))

	"""
	Now, we'll define our "solver" which trains the network by specifying the locations of 
	the train and test nets we defined above, as well as setting values for various 
	parameters used for learning, display, and "snapshotting".
	"""
	from caffe.proto import caffe_pb2

	def solver(train_net_path, test_net_path=None):
		s = caffe_pb2.SolverParameter()

		# Specify locations of the train and test networks.
		s.train_net = train_net_path
		if test_net_path is not None:
			s.test_net.append(test_net_path)

		s.test_interval = 1000  # Test after every 1000 training iterations.
		s.test_iter.append(250) # Test 250 "batches" each time we test.

		s.max_iter = 10000      # # of times to update the net (training iterations)

		# Set the initial learning rate for stochastic gradient descent (SGD).
		s.base_lr = 0.01        

		# Set `lr_policy` to define how the learning rate changes during training.
		# Here, we 'step' the learning rate by multiplying it by a factor `gamma`
		# every `stepsize` iterations.
		s.lr_policy = 'step'
		s.gamma = 0.1
		s.stepsize = 5000

		# Set other optimization parameters. Setting a non-zero `momentum` takes a
		# weighted average of the current gradient and previous gradients to make
		# learning more stable. L2 weight decay regularizes learning, to help prevent
		# the model from overfitting.
		s.momentum = 0.9
		s.weight_decay = 5e-4

		# Display the current training loss and accuracy every 1000 iterations.
		s.display = 1000

		# Snapshots are files used to store networks we've trained.  Here, we'll
		# snapshot every 10K iterations -- just once at the end of training.
		# For larger networks that take longer to train, you may want to set
		# snapshot < max_iter to save the network and training state to disk during
		# optimization, preventing disaster in case of machine crashes, etc.
		s.snapshot = 10000
		s.snapshot_prefix = 'model/logreg'

		# We'll train on the CPU for fair benchmarking against scikit-learn.
		# Changing to GPU should result in much faster training!
		s.solver_mode = caffe_pb2.SolverParameter.CPU

		return s

	solver_path = 'model/solver_logreg.prototxt'
	with open(solver_path, 'w') as f:
		f.write(str(solver(train_net_path, test_net_path)))


	######################################################################################
	########################### training and evaluate net ################################	
	######################################################################################
	caffe.set_mode_cpu()
	solver = caffe.get_solver(solver_path)
	solver.solve()

	accuracy = 0
	batch_size = solver.test_nets[0].blobs['data'].num
	print('batch_size of test_net is: {}'.format(batch_size))
	test_iters = int(len(Xt) / batch_size)
	for i in range(test_iters):
		solver.test_nets[0].forward()
		accuracy += solver.test_nets[0].blobs['accuracy'].data
	accuracy /= test_iters

	print("Accuracy: {:.3f}".format(accuracy))

	"""
	Do the same through the command line interface for detailed output on the model 
	and solving.
	"""
	# os.system('{} train -solver model/solver_logreg.prototxt'.format(os.path.join(args.caffe_root, 'build/tools/caffe')))


	"""
	If you look at output or the logreg_auto_train.prototxt, you'll see that the model is 
	simple logistic regression. We can make it a little more advanced by introducing a 
	non-linearity between weights that take the input and weights that give the output -- 
	now we have a two-layer network. That network is given in nonlinear_auto_train.prototxt, 
	and that's the only change made in solver_nonlinear_logreg.prototxt which we will now use.

	The final accuracy of the new network should be higher than logistic regression!
	"""
	def nonlinear_net(hdf5, batch_size):
		# one small nonlinearity, one leap for model kind
		n = caffe.NetSpec()
		n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
		# define a hidden layer of dimension 40
		n.ip1 = L.InnerProduct(n.data, num_output=40, weight_filler=dict(type='xavier'))
		# transform the output through the ReLU (rectified linear) non-linearity
		n.relu1 = L.ReLU(n.ip1, in_place=True)
		# score the (now non-linear) features
		n.ip2 = L.InnerProduct(n.ip1, num_output=2, weight_filler=dict(type='xavier'))
		# same accuracy and loss as before
		n.accuracy = L.Accuracy(n.ip2, n.label)
		n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
		return n.to_proto()

	train_net_path = 'model/train_nonlinear_logreg.prototxt'
	with open(train_net_path, 'w') as f:
		f.write(str(nonlinear_net('hdf5_data/train.txt', 10)))

	test_net_path = 'model/test_nonlinear_logreg.prototxt'
	with open(test_net_path, 'w') as f:
		f.write(str(nonlinear_net('hdf5_data/test.txt', 10)))

	solver_path = 'model/solver_nonlinear_logreg.prototxt'
	with open(solver_path, 'w') as f:
		f.write(str(solver(train_net_path, test_net_path)))

