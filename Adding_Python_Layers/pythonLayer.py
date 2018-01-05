
"""
create a python file which contains the definition for your python layer.
"""
import caffe
import random

class BlankSquareLayer(caffe.Layer):
	def setup(self, bottom, top):
		assert len(bottom) == 1, 	'requires a single layer.bottom'
		assert bottom[0].data.ndim >= 3, 	'requires image data'
		assert len(top) == 1, 	'requires a single layer.top'

	def reshape(self, bottom, top):
		## Copy shape from bottom
		top[0].reshape(*bottom[0].data.shape)

	def forward(self, bottom, top):
		## Copy all of the data
		top[0].data[...] = bottom[0].data[...]
		## Then zero-out one fourth of the image
		height = top[0].data.shape[-2]
		width = top[0].data.shape[-1]
		h_offset = random.randrange(height/2)
		w_offset = random.randrange(width/2)
		top[0].data[..., h_offset:(h_offset + height/2), w_offset:(w_offset + width/2),] = 0

	def backward(self, top, propagate_down, bottom):
		pass 



