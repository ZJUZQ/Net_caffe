%% convert the annotation format from json to mat

dataType = '';
addpath('/home/zq/dataset/COCO/coco/MatlabAPI');

mkdir('/home/zq/dataset/COCO/mat')

annTypes = { 'instances', 'captions', 'person_keypoints' };
annType=annTypes{3}; % specify dataType/annType

%for mode = 0:1
for mode = 1
    
    if mode == 0
        dataType= 'val2014';
        annFile=sprintf('dataset/COCO/annotations/%s_%s.json',annType,dataType);
    else
        dataType = 'train2017';
        annFile=sprintf('/home/zq/dataset/COCO/annotations/%s_%s.json',annType,dataType);
    end
    
    coco=CocoApi(annFile);
    %%
    my_anno = coco.data.annotations;
    %%
    prev_id = -1;
    p_cnt = 1;
    cnt = 0;
    coco_kpt = [];
    
    for i = 1:1:size(my_anno,2)
        
        curr_id = my_anno(i).image_id;
        if(curr_id == prev_id) %% an image may contain several objects/annotations
            p_cnt = p_cnt + 1;
        else
            p_cnt = 1;
            cnt = cnt + 1;
        end
        coco_kpt(cnt).image_id = curr_id;
        coco_kpt(cnt).annorect(p_cnt).bbox = my_anno(i).bbox; %% [x,y,width,height]
        coco_kpt(cnt).annorect(p_cnt).segmentation = my_anno(i).segmentation; %% RLE or [polygon]
        coco_kpt(cnt).annorect(p_cnt).area = my_anno(i).area;
        coco_kpt(cnt).annorect(p_cnt).id = my_anno(i).id;
        coco_kpt(cnt).annorect(p_cnt).iscrowd = my_anno(i).iscrowd; %% Crowd annotations (iscrowd=1) are used to label large groups of objects (e.g. a crowd of people)
        coco_kpt(cnt).annorect(p_cnt).keypoints = my_anno(i).keypoints; %% [x1,y1,v1,...]
        coco_kpt(cnt).annorect(p_cnt).num_keypoints = my_anno(i).num_keypoints;
        coco_kpt(cnt).annorect(p_cnt).img_width = coco.loadImgs(curr_id).width;
        coco_kpt(cnt).annorect(p_cnt).img_height = coco.loadImgs(curr_id).height;
        
        prev_id = curr_id;
        
        fprintf('%d/%d \n', i, size(my_anno, 2));
    end
    %%
    if mode == 0
        coco_val = coco_kpt;
        save('/home/zq/dataset/COCO/mat/coco_val.mat', 'coco_val');
    else
        save('/home/zq/dataset/COCO/mat/coco_kpt.mat', 'coco_kpt');
    end
    
end