#!/bin/bash
## usage: bash get_pascal_2012.sh /home/zq/dataset

wget -r -p $1 -O VOCtrainval_2012.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_2012.tar
mv VOCdevkit/VOC2012 $1/
rm -rf VOCdevkit
rm -f VOCtrainval_2012.tar