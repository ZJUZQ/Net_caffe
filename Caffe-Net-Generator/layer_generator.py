__author__ = 'Wei Wen'

from caffe.proto import caffe_pb2

def add_conv_layer(net_msg,name,bottom,num_output,pad,kernel_size,stride,bias_term=True):
    conv_layer = net_msg.layer.add()
    conv_layer.name = name
    conv_layer.type = 'Convolution'
    conv_layer.bottom._values.append(bottom)
    conv_layer.top._values.append(conv_layer.name)
    # param info for weight and bias
    lr_param = caffe_pb2.ParamSpec()
    lr_param.lr_mult = 1
    conv_layer.param._values.append(lr_param)
    if bias_term:
        lr_param = caffe_pb2.ParamSpec()
        lr_param.lr_mult = 2
        conv_layer.param._values.append(lr_param)
    # conv parameters
    conv_layer.convolution_param.num_output = num_output
    conv_layer.convolution_param.pad._values.append(pad)
    conv_layer.convolution_param.kernel_size._values.append(kernel_size)
    conv_layer.convolution_param.stride._values.append(stride)
    conv_layer.convolution_param.weight_filler.type = 'msra'
    conv_layer.convolution_param.bias_term = bias_term
    if bias_term:
        conv_layer.convolution_param.bias_filler.type = 'constant'

def add_relu_layer(net_msg,name,bottom):
    relulayer = net_msg.layer.add()
    relulayer.name = name
    relulayer.type = 'ReLU'
    relulayer.bottom._values.append(bottom)
    relulayer.top._values.append(name)

def add_eltwise_add_layer(net_msg,name,bottom1,bottom2):
    eltlayer = net_msg.layer.add()
    eltlayer.name = name
    eltlayer.type = 'Eltwise'
    eltlayer.bottom._values.append(bottom1)
    eltlayer.bottom._values.append(bottom2)
    eltlayer.top._values.append(name)

def add_BN_layer(net_msg,name,bottom):
    # norm layer
    batchnormlayer = net_msg.layer.add()
    batchnormlayer.name = name+'_norm'
    batchnormlayer.type = 'BatchNorm'
    batchnormlayer.bottom._values.append(bottom)
    batchnormlayer.top._values.append(batchnormlayer.name)
    for i in range(0,3):
        lr_param = caffe_pb2.ParamSpec()
        lr_param.lr_mult = 0
        batchnormlayer.param._values.append(lr_param)
    # scale layer
    scalelayer = net_msg.layer.add()
    scalelayer.name = name+'_scale'
    scalelayer.type = 'Scale'
    scalelayer.bottom._values.append(batchnormlayer.name)
    scalelayer.top._values.append(name)

    lr_param = caffe_pb2.ParamSpec()
    lr_param.lr_mult = 1
    scalelayer.param._values.append(lr_param)
    lr_param = caffe_pb2.ParamSpec()
    lr_param.lr_mult = 2
    lr_param.decay_mult = 0
    scalelayer.param._values.append(lr_param)

    scalelayer.scale_param.bias_term = True
    scalelayer.scale_param.filler.type = 'msra'

def add_global_avg_pooling_layer(net_msg,name,bottom):
    glb_avg_pl_layer = net_msg.layer.add()
    glb_avg_pl_layer.name = name
    glb_avg_pl_layer.type = 'Pooling'
    glb_avg_pl_layer.bottom._values.append(bottom)
    glb_avg_pl_layer.top._values.append(name)
    glb_avg_pl_layer.pooling_param.pool = caffe_pb2.PoolingParameter.AVE
    glb_avg_pl_layer.pooling_param.global_pooling = True

def add_downsampling_layer(net_msg,name,bottom,stride):
    downsampling_layer = net_msg.layer.add()
    downsampling_layer.name = name
    downsampling_layer.type = 'Pooling'
    downsampling_layer.bottom._values.append(bottom)
    downsampling_layer.top._values.append(name)
    downsampling_layer.pooling_param.pool = caffe_pb2.PoolingParameter.MAX
    downsampling_layer.pooling_param.kernel_size = 1
    downsampling_layer.pooling_param.stride = stride

def add_ip_layer(net_msg,name,bottom,num):
    ip_layer = net_msg.layer.add()
    ip_layer.name = name
    ip_layer.type = 'InnerProduct'
    ip_layer.bottom._values.append(bottom)
    ip_layer.top._values.append(name)
    # param info for weight and bias
    lr_param = caffe_pb2.ParamSpec()
    lr_param.lr_mult = 1
    lr_param.decay_mult = 1
    ip_layer.param._values.append(lr_param)
    lr_param = caffe_pb2.ParamSpec()
    lr_param.lr_mult = 2
    lr_param.decay_mult = 0
    ip_layer.param._values.append(lr_param)
    # inner product parameters
    ip_layer.inner_product_param.num_output = num
    ip_layer.inner_product_param.weight_filler.type = 'msra'
    ip_layer.inner_product_param.bias_filler.type = 'constant'
    ip_layer.inner_product_param.bias_filler.value = 0.0

def add_accuracy_layer(net_msg,bottom):
    accuracy_layer = net_msg.layer.add()
    accuracy_layer.name = 'accuracy'
    accuracy_layer.type = 'Accuracy'
    accuracy_layer.bottom._values.append(bottom)
    accuracy_layer.bottom._values.append('label')
    accuracy_layer.top._values.append('accuracy')
    include_param = caffe_pb2.NetStateRule()
    include_param.phase = caffe_pb2.TEST
    accuracy_layer.include._values.append(include_param)

def add_loss_layer(net_msg,bottom):
    loss_layer = net_msg.layer.add()
    loss_layer.name = 'loss'
    loss_layer.type = 'SoftmaxWithLoss'
    loss_layer.bottom._values.append(bottom)
    loss_layer.bottom._values.append('label')
    loss_layer.top._values.append('loss')