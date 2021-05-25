#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:16:12 2021

@author: 
"""
from util import file_reader, rle_decoding, draw_bbox, show_image
from PIL import Image
import numpy as np
import os
import config as cfg

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOAD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DATASET_ = file_reader('pred_valid_instances.json')

# ANNOTATIONS KEYS 
#['annotation_id', 'video_id', 'len_sequence', 'num_sequence', 
#'file_name', 'height', 'width', 'gt_instances', 'gt_ids', 'gt_bboxes', 
#'gt_areas', 'gt_categories', 'gt_segmentations', 'pred_instances', 'pred_categories', 
#'pred_bboxes', 'pred_segmentations', 'pred_features']

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# fill the paths if are different from the one in the json
IMG_PATH = '/home/fiorapirri/Documents/workspace/YoutubeVIS21/valid/'

FEAT_PATH = ''

ANNOTATIONS = DATASET_['annotations']

CATEGORIES = cfg.CLASS_YTVIS21

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SHOW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#OFFSET = 1
#FROM = np.random.choice(len(ANNOTATIONS)-OFFSET) # i.e. 48782
#TO = FROM + OFFSET
#
#for a in ANNOTATIONS[:20]:
#    file_name = a['file_name']
#    img = np.array(Image.open(os.path.join(IMG_PATH, file_name)))/255
#    # GROUND TRUTH
##    gt_bboxes = np.array(a['gt_bboxes']).astype(int)
##    gt_bboxes[:,[2,3]] = gt_bboxes[:,[0,1]] + gt_bboxes[:,[2,3]]
##    gt_segmentations = np.array([rle_decoding(m) for m in a['gt_segmentations']])
##    gt_categories = np.array(a['gt_categories'])
##    draw_bbox(img, box=gt_bboxes, conf=None, class_id=gt_categories, mask=gt_segmentations, class_dict=CATEGORIES, mode='PIL')
#    # PREDICTIONS
#    pred_bboxes = np.array(a['pred_bboxes']).astype(int)
#    if len(pred_bboxes)>0:
#        pred_bboxes[:,[2,3]] = pred_bboxes[:,[0,1]] + pred_bboxes[:,[2,3]]
#        pred_segmentations = np.array([rle_decoding(m) for m in a['pred_segmentations']])
#        pred_categories = np.array(a['pred_categories'])
#        draw_bbox(img, box=pred_bboxes, conf=None, class_id=pred_categories, mask=pred_segmentations, class_dict=CATEGORIES, mode='PIL')
#    else:
#        show_image(np.array(img*255,np.uint8))
#    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SHOW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import matplotlib.pyplot as plt
import pandas as pd

TRACKING_ANNOTATIONS = pd.read_json('results-valid-heuristics2.json')

DATASET = file_reader(cfg.YT_VALID_ANNOTATION_PATH)

IMG_PATH = os.path.join(cfg.YT_VALID_FRAMES_PATH)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SHOW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ids, idx = np.unique(TRACKING_ANNOTATIONS['video_id'], return_index = True)

for video in DATASET['videos']:
    video_id = video['id']
    index = ids[ids==video_id]-1
    length = video['length']
    h = video['height']
    w = video['width']
    for i,f in enumerate(video['file_names']):
        img = np.array(Image.open(os.path.join(IMG_PATH, f)))/255
        if len(index)>0:
            FROM = idx[index][0]
            TO = idx[index+1][0]
            instances = TRACKING_ANNOTATIONS[FROM:TO]
            pred_segmentations = np.array([rle_decoding(m[i]) if not m[i]==None else np.zeros((h,w)) for m in instances['segmentations']])
            pred_categories =  instances['category_id'].values
            pred_score =  instances['score'].values
            idd = instances.index.values
            if len(pred_segmentations)>0:
                img = draw_bbox(img, box=None, conf=idd, class_id=pred_categories, mask=pred_segmentations, class_dict=CATEGORIES, mode='plt')
            plt.imshow(img)
            plt.show()
        else:
            plt.imshow(img)
            plt.show()
 #%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SHOW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%