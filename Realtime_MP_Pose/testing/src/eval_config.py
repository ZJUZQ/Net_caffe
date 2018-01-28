import os, sys
import caffe

def eval_config(id): ## id == 0 for coco, id == 1 for MPII
 
    param = {} ## dict

    ## CPU mode or GPU mode
    param['use_gpu'] = 0 
    ## GPU device number (doesn't matter for CPU mode)
    GPUdeviceNumber = 0

    ## Select model (default: 5)
    param['modelID'] = id
    param['mid_num'] = 10 ## sample points when compute paf integral
    param['save_render_image'] = 0

    param['model'] = 2 * [{}] ## list of dict for COCO and MPII

    ## the larger the crop_ratio, the smaller the windowsize
    param['crop_ratio'] = 2.5 #2
    param['bbox_ratio'] = 0.25 #0.5
 
    if id == 0: ## COCO parameter
        param['scale_search'] = [1.0]
        param['thre1'] = 0.1 ## threshould used for supress heatmaps
        param['thre2'] = 0.05 ## used for line integral of paf
        param['thre3'] = 0.5

        param['model'][id]['caffemodel'] = os.environ['HOME'] + '/Net_caffe/Realtime_MP_Pose/_trained_model/coco/pose_iter_440000.caffemodel'
        param['model'][id]['deployFile'] = os.environ['HOME'] + '/Net_caffe/Realtime_MP_Pose/_trained_model/coco/pose_deploy.prototxt'
        param['model'][id]['description'] = 'COCO Pose56 Two-level Linevec'
        param['model'][id]['boxsize'] = 368
        param['model'][id]['maxsize'] = 480
        param['model'][id]['padValue'] = 128
        param['model'][id]['kpt_num'] = 18 ## without background
        param['model'][id]['stride'] = 8
        
        ## keypoints order in coco annotation (17 keypoints)
        """
        {0, "nose",
         1, "left_eye",
         2, "right_eye",
         3, "left_ear",
         4, "right_ear", 
         5, "left_shoulder",
         6, "right_shoulder",
         7, "left_elbow",
         8, "right_elbow",
         9, "left_wrist",
         10, "right_wrist",
         11, "left_hip",
         12, "right_hip",
         13, "left_knee",
         14, "right_knee",
         15, "left_ankle",
         16, "right_ankle"]
        """

        ## keypoints order used in openpose (18 keypoints + background)
        """
        POSE_COCO_BODY_PARTS {
        {0,  "Nose"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "RHip"},
        {9,  "RKnee"},
        {10, "RAnkle"},
        {11, "LHip"},
        {12, "LKnee"},
        {13, "LAnkle"},
        {14, "REye"},
        {15, "LEye"},
        {16, "REar"},
        {17, "LEar"},
        {18, "Background"},
        }
        """
        param['model'][id]['part_str'] = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', \
                                          'Lsho', 'Lelb', 'Lwri', \
                                          'Rhip', 'Rkne', 'Rank', \
                                          'Lhip', 'Lkne', 'Lank', \
                                          'Reye', 'Leye', 'Rear', 'Lear', 'pt19']
        
        ## For the heat maps storing format,  (18 body parts + background + 2 x 19 PAFs) 
        param['model'][id]['linkPair'] = [[2,3], [2,6], [3,4], [4,5], 
                                          [6,7], [7,8], [2,9], [9,10],
                                          [10,11],[2,12],[12,13],[13,14],
                                          [2,1], [1,15],[15,17],[1,16], 
                                          [16,18], [3,17], [6,18]]

        # index of paf in all heatmaps
        param['model'][id]['linkPafIdx'] = [[31,32], [39,40], [33,34], 
                                            [35,36], [41,42], [43,44], 
                                            [19,20], [21,22], [23,24], 
                                            [25,26], [27,28], [29,30], 
                                            [47,48], [49,50], [53,54],
                                            [51,52], [55,56], [37,38], 
                                            [45,46]]

        param['model'][id]['part_color'] =[[255, 0, 0], [255, 85, 0], 
                                           [255, 170, 0], [255, 255, 0], 
                                           [170, 255, 0], [85, 255, 0], 
                                           [0, 255, 0], [0, 255, 85], 
                                           [0, 255, 170], [0, 255, 255],
                                           [0, 170, 255], [0, 85, 255], 
                                           [0, 0, 255], [85, 0, 255], 
                                           [170, 0, 255], [255, 0, 255],
                                           [255, 0, 170], [255, 0, 85]]

    ## MPI parameter
    if id == 1:
        param['scale_search'] = [1.0]
        param['thre1'] = 0.05
        param['thre2'] = 0.01
        param['thre3'] = 3
        param['thre4'] = 0.1
        
        param['model'][id]['caffemodel'] = os.environ['HOME'] + '/Net_caffe/Realtime_MP_Pose/_trained_model/mpii/pose_iter_146000.caffemodel'
        param['model'][id]['deployFile'] = os.environ['HOME'] + '/Net_caffe/Realtime_MP_Pose/_trained_model/mpii/pose_deploy.prototxt'
        param['model'][id]['description'] = 'MPI Pose43 Two-level LineVec'
        param['model'][id]['boxsize'] = 368
        param['model'][id]['padValue'] = 128 
        param['model'][id]['kpt_num'] = 15 
        param['model'][id]['stride'] = 8
        param['model'][id]['MPII_imageFolder'] = os.environ['HOME'] + '/dataset/MPII/images/'
        param['model'][id]['annolist_test'] = os.environ['HOME'] + '/dataset/MPII/mpii_human_pose_v1_u12_1/annolist_test.mat'


        ## openpose_mpii: joint_map: 1x16x46x46
        param['model'][id]['part_str'] = ['Nose', 'Neck', 'Rsho', 'Relb', 'Rwri', \
                                          'Lsho', 'Lelb', 'Lwri', \
                                          'Rhip', 'Rkne', 'Rank', \
                                          'Lhip', 'Lkne', 'Lank', 'center']

        ## find connection in the specified sequence, center 29 is in the position 15 ('center')
        param['model'][id]['linkPair'] =  [[1, 2], [2, 3], [3, 4], [4, 5], [2, 6],
                                           [6, 7], [7, 8], [2, 15], [15, 12], [12, 13],
                                           [13, 14], [15, 9], [9, 10], [10, 11]]

        ## the middle joints heatmap correpondence
        ## openpose_mpii: paf_map: 1x28x46x46
        param['model'][id]['linkPafIdx'] = [[16, 17], [18, 19], [20, 21], [22, 23], [24, 25], 
                                            [26, 27], [28, 29], [30, 31], [38, 39], [40, 41], 
                                            [42, 43], [32, 33], [34, 35], [36, 37]]


        param['model'][id]['part_color'] =[[255, 0, 0], [255, 85, 0], 
                                           [255, 170, 0], [255, 255, 0], 
                                           [170, 255, 0], [85, 255, 0], 
                                           [0, 255, 0], [0, 255, 85], 
                                           [0, 255, 170], [0, 255, 255],
                                           [0, 170, 255], [0, 85, 255], 
                                           [0, 0, 255], [85, 0, 255], 
                                           [170, 0, 255], [255, 0, 255]]


    
    if param['use_gpu']:
        caffe.set_mode_gpu() 
        caffe.set_device(GPUdeviceNumber) 
    #caffe.reset_all() 
    
    return param