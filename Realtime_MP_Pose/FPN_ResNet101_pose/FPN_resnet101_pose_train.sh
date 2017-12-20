#!/usr/bin/env sh
/home/dgxuser1/zhouqiang/Realtime_Multi-Person_Pose_Estimation/caffe_train/build/tools/caffe train --solver=FPN_resnet101_pose_solver.prototxt --gpu=1,2,3 --weights=./_trained_ResNet101/ResNet101.caffemodel 2>&1 | tee ./output.txt
