#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 18:01:02 2021

@author: fiorapirri
"""


import env


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import config as cfg
import tensorflow as tf
from loader_ytvos import DataLoader 
from model import get_model
import numpy as np
from PIL import Image
import json
from utils import file_reader, prediction_preprocess, prediction_postprocess, unmold_mask_proposals, rle_encoding
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model = get_model(infer=True)

#fine_tuning(model)

model.load_weights('/home/fiorapirri/tracker/weights/model.54--7.149.h5')

model.trainable = False
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DATASET = file_reader(cfg.YT_TRAIN_ANNOTATION_PATH)

IMG_PATH = os.path.join(cfg.YT_TRAIN_FRAMES_PATH)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ANNOTATIONS = DATASET['annotations']

VIDEOS = DATASET['videos']
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ANNOTATIONS_INDEX_DICT = dict()

for annotation_index, annotation in enumerate(ANNOTATIONS):
    try:
        ANNOTATIONS_INDEX_DICT[annotation['video_id']].append(annotation_index)
    except:
        ANNOTATIONS_INDEX_DICT[annotation['video_id']] = [annotation_index]
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#import time
#NEW_DATASET = []
#new_annotation_count = 0
#CONF_THRESH = 0.6
#for video_index, video in enumerate(VIDEOS[1285:]):
#    video_id = video['id']
#    length = video['length']
#    file_names = video['file_names']
#    w, h = video['width'], video['height']
#    annotation_indices = ANNOTATIONS_INDEX_DICT[video_id]
##    start = time.time()
#    for i in range(length):
#        # ground truth
#        new_annotation_count += 1
#        file_name = file_names[i]
#        sub_path = file_name.split('.')[0]
#        gt_ids = []
#        gt_areas = []
#        gt_bboxes = []
#        gt_categories = []
#        gt_segmentations = []
#        num_instances = 0
#        for annotation_index in annotation_indices:
#            annotation = ANNOTATIONS[annotation_index]
#            ids = annotation['id']
#            area = annotation['areas'][i]
#            bbox = annotation['bboxes'][i]
#            category = annotation['category_id']
#            segmentation = annotation['segmentations'][i]
#            if area != None and bbox != None and segmentation != None:
#                num_instances+=1
#                gt_areas.append(area)
#                gt_ids.append(ids)
#                gt_bboxes.append(bbox)
#                gt_categories.append(category)
#                gt_segmentations.append(segmentation)
#        new_annotation = dict()
#        new_annotation['annotation_id'] = new_annotation_count
#        new_annotation['video_id'] = video_id
#        new_annotation['len_sequence'] = length
#        new_annotation['num_sequence'] = i+1
#        new_annotation['file_name'] = os.path.join('JPEGImages', file_name)
#        new_annotation['height'] = h
#        new_annotation['width'] = w
#        new_annotation['gt_instances'] = num_instances
#        new_annotation['gt_ids'] = gt_ids
#        new_annotation['gt_bboxes'] = gt_bboxes
#        new_annotation['gt_areas'] = gt_areas
#        new_annotation['gt_categories'] = gt_categories
#        new_annotation['gt_segmentations'] = gt_segmentations
#        
#        # predictions
#        img = np.array(Image.open(os.path.join(IMG_PATH, file_name)))
#        img, hh, ww = prediction_preprocess(img)
#        box, conf, class_id, mask, features = model.infer(tf.constant(img)[tf.newaxis])
#        pred_instances = int(tf.reduce_sum(tf.cast(conf>=CONF_THRESH,tf.int32)).numpy())
#        pred_confs = []
#        pred_categories = []
#        pred_bboxes = []
#        pred_segmentations = []
#        pred_features = []
#        if pred_instances>0:
#            conf = conf[0,:pred_instances].numpy()
#            features = features[0,:pred_instances].numpy()
#            box = box[0,:pred_instances,:].numpy()
#            class_id = class_id[0,:pred_instances].numpy()
#            mask = mask[0,:pred_instances,:,:].numpy()
#            mask = unmold_mask_proposals((mask, box)).numpy()
#            bboxes, masks = prediction_postprocess(box, mask, hh, ww)
#            pred_confs.extend([round(c,3) for c in conf.tolist()])
#            pred_categories.extend([int(c) for c in class_id.tolist()])
#            pred_bboxes.extend(bboxes.tolist())
#            pred_segmentations.extend([rle_encoding(mask) for mask in masks])
#            feature_paths = [os.path.join('NPYFeatures', sub_path,(f'{j+1:03}')+'.npy') for j in range(pred_instances)] 
#            pred_features.extend(feature_paths)
#            os.makedirs(os.path.join('NPYFeatures', sub_path),exist_ok=True)
#            for j in range(pred_instances):
#                np.save(feature_paths[j], features[j])
#        new_annotation['pred_instances'] = pred_instances
#        new_annotation['pred_categories'] = pred_categories
#        new_annotation['pred_confs'] = pred_confs
#        new_annotation['pred_bboxes'] = pred_bboxes
#        new_annotation['pred_segmentations'] = pred_segmentations
#        new_annotation['pred_features'] = pred_features
#        
#        NEW_DATASET.append(new_annotation)
##    print(time.time()-start)
##    break
#NEW_INSTANCES = dict()
#NEW_INSTANCES['categories'] = cfg.CLASS_YTVIS19
#NEW_INSTANCES['annotations'] = NEW_DATASET
#
#with open('new_instances_5.json', 'w') as f:
#    f.write(json.dumps(A))
#    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#NEW_PATH = 'JPEGImages'
#os.makedirs(NEW_PATH, exist_ok=True)
#for video_index, video in enumerate(VIDEOS):
#    file_names = video['file_names']
#    path = file_names[0].split('/')[0]
#    os.makedirs(os.path.join(NEW_PATH,path), exist_ok=True)
#    break
#    for i in range(len(file_names)):
#        file_name = file_names[i]
#        img = Image.open(os.path.join(IMG_PATH, file_name)).save(os.path.join(file_name))