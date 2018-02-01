# Caffe-Net-Generator

This is a repo to generate net prototxt for Caffe.

resnet_generator.py is provided to generate ResNet in Cifar10.
1x1 convolutional layers are utilized as the shortcuts.
Original paper: https://arxiv.org/abs/1512.03385

E.G.
```
# --n: number of groups, please refer to the above paper
# --net_template: prototxt template specifies the data layer
python resnet_generator.py --n 3 --net_template resnet_template.prototxt
```
Generated prototxt is `cifar10_resnet_n3.prototxt`
