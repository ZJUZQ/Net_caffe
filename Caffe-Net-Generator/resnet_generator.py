__author__ = 'wei wen'

import argparse
import caffeparser
import caffe
from caffe.proto import caffe_pb2
from numpy import *
import re
from layer_generator import *
import os

def add_1st_res_layers(net_msg,name,bottom):
    # first layer
    add_conv_layer(net_msg,name=name+'_conv1',bottom=bottom,num_output=16,pad=1,kernel_size=3,stride=1,bias_term=False)
    add_BN_layer(net_msg,name=name+'_bn1',bottom=name+'_conv1')
    add_relu_layer(net_msg,name=name+'_relu1',bottom=name+'_bn1')
    #second conv
    add_conv_layer(net_msg,name=name+'_conv2',bottom=name+'_relu1',num_output=16,pad=1,kernel_size=3,stride=1,bias_term=False)
    add_BN_layer(net_msg,name=name+'_bn2',bottom=name+'_conv2')
    #add layer
    add_eltwise_add_layer(net_msg,name+'_add',bottom,name+'_bn2')
    #final relu
    add_relu_layer(net_msg,name=name,bottom=name+'_add')

def add_2nd_res_layers(net_msg,name,bottom,downsample=False):
    # first layer
    if downsample:
        add_conv_layer(net_msg,name=name+'_conv1',bottom=bottom,num_output=32,pad=1,kernel_size=3,stride=2,bias_term=False)
    else:
        add_conv_layer(net_msg,name=name+'_conv1',bottom=bottom,num_output=32,pad=1,kernel_size=3,stride=1,bias_term=False)
    add_BN_layer(net_msg,name=name+'_bn1',bottom=name+'_conv1')
    add_relu_layer(net_msg,name=name+'_relu1',bottom=name+'_bn1')
    #second conv
    add_conv_layer(net_msg,name=name+'_conv2',bottom=name+'_relu1',num_output=32,pad=1,kernel_size=3,stride=1,bias_term=False)
    add_BN_layer(net_msg,name=name+'_bn2',bottom=name+'_conv2')
    #add layer
    if downsample:
        #add_downsampling_layer(net_msg,name+'_downsampling',bottom,2)
        add_conv_layer(net_msg,name=name+'_downsampling',bottom=bottom,num_output=32,pad=0,kernel_size=1,stride=2,bias_term=False)
        add_eltwise_add_layer(net_msg,name+'_add',name+'_downsampling',name+'_bn2')
    else:
        add_eltwise_add_layer(net_msg,name+'_add',bottom,name+'_bn2')
    #final relu
    add_relu_layer(net_msg,name=name,bottom=name+'_add')

def add_3rd_res_layers(net_msg,name,bottom,downsample=False):
    # first layer
    if downsample:
        add_conv_layer(net_msg,name=name+'_conv1',bottom=bottom,num_output=64,pad=1,kernel_size=3,stride=2,bias_term=False)
    else:
        add_conv_layer(net_msg,name=name+'_conv1',bottom=bottom,num_output=64,pad=1,kernel_size=3,stride=1,bias_term=False)
    add_BN_layer(net_msg,name=name+'_bn1',bottom=name+'_conv1')
    add_relu_layer(net_msg,name=name+'_relu1',bottom=name+'_bn1')
    #second conv
    add_conv_layer(net_msg,name=name+'_conv2',bottom=name+'_relu1',num_output=64,pad=1,kernel_size=3,stride=1,bias_term=False)
    add_BN_layer(net_msg,name=name+'_bn2',bottom=name+'_conv2')
    #add layer
    if downsample:
        #add_downsampling_layer(net_msg,name+'_downsampling',bottom,2)
        add_conv_layer(net_msg,name=name+'_downsampling',bottom=bottom,num_output=64,pad=0,kernel_size=1,stride=2,bias_term=False)
        add_eltwise_add_layer(net_msg,name+'_add',name+'_downsampling',name+'_bn2')
    else:
        add_eltwise_add_layer(net_msg,name+'_add',bottom,name+'_bn2')

    #final relu
    add_relu_layer(net_msg,name=name,bottom=name+'_add')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_template', type=str, required=True)
    parser.add_argument('--n', type=int, required=True)
    #parser.add_argument('--connectivity_mode', type=int, required=True)
    #parser.add_argument('--learn_depth', type=bool, required=False)
    parser.add_argument('--learndepth', dest='learndepth', action='store_true')
    parser.add_argument('--no-learndepth', dest='learndepth', action='store_false')
    parser.set_defaults(learndepth=False)
    args = parser.parse_args()
    net_template = args.net_template
    n = args.n
    learn_depth = args.learndepth
    #connectivity_mode = args.connectivity_mode

    caffe.set_mode_cpu()
    net_parser = caffeparser.CaffeProtoParser(net_template)
    net_msg = net_parser.readProtoNetFile()

    add_conv_layer(net_msg,name='conv1',bottom='data',num_output=16,pad=1,kernel_size=3,stride=1,)
    add_BN_layer(net_msg,name='conv1_bn',bottom='conv1')
    add_relu_layer(net_msg,name='conv1_relu',bottom='conv1_bn')

    for i in range(1,n+1):
        if i==1:
            add_1st_res_layers(net_msg,name='res_grp1_{}'.format(i),bottom='conv1_relu')
        else:
            add_1st_res_layers(net_msg,'res_grp1_{}'.format(i),'res_grp1_{}'.format(i-1))

    for i in range(1,n+1):
        if i==1:
            add_2nd_res_layers(net_msg,name='res_grp2_{}'.format(i),bottom='res_grp1_{}'.format(n),downsample=True)
        else:
            add_2nd_res_layers(net_msg,'res_grp2_{}'.format(i),'res_grp2_{}'.format(i-1),downsample=False)

    for i in range(1,n+1):
        if i==1:
            add_3rd_res_layers(net_msg,name='res_grp3_{}'.format(i),bottom='res_grp2_{}'.format(n),downsample=True)
        else:
            add_3rd_res_layers(net_msg,'res_grp3_{}'.format(i),'res_grp3_{}'.format(i-1),downsample=False)


    #conv_cur_layer.CopyFrom(conv_layer)
    add_global_avg_pooling_layer(net_msg,name='pool1',bottom='res_grp3_{}'.format(n))
    add_ip_layer(net_msg=net_msg,name='ip1',bottom='pool1',num=10)
    add_accuracy_layer(net_msg=net_msg,bottom='ip1')
    add_loss_layer(net_msg=net_msg,bottom='ip1')

    file_split = os.path.splitext(net_template)
    filepath = 'cifar10_resnet_n{}'.format(n)+file_split[1]
    file = open(filepath, "w")
    if not file:
        raise IOError("ERROR (" + filepath + ")!")
    file.write(str(net_msg))
    file.close()

    print net_msg
    print "saved as {}".format(filepath)