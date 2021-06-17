#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:16:12 2021

@author: fiora
"""
from graph_tracking.util import file_reader, rle_decoding, draw_bbox,  get_whole_single_video

import numpy as np
#import osbboxes
from numpy import matlib
import json

import timeit

from PIL import Image, ImageOps
from matplotlib import pyplot as plt 
import matplotlib.patches as patches
import tensorflow as tf
import tensorflow_probability as tfp
import itertools as itt
# import pandas as pd
from graph_tracking.bipartiteG import bipartite_matching
# from unsup_tracking_VIS_new import *
import copy
import graph_tracking.config_VIS as cfg
from graph_tracking.main_search_DEV import main_search_and_prediction, test_accuracy
import config
import time
import os
from tqdm import tqdm
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOAD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


DATASET_ = file_reader('new_10_pred_val_instances.json')

# ANNOTATIONS KEYS 
#['annotation_id', 'video_id', 'len_sequence', 'num_sequence', 
#'file_name', 'height', 'width', 'gt_instances', 'gt_ids', 'gt_bboxes', 
#'gt_areas', 'gt_categories', 'gt_segmentations', 'pred_instances', 'pred_categories', 
#'pred_bboxes', 'pred_segmentations', 'pred_features']

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# fill the paths if are different from the one in the json
# fill the paths if are different from the one in the json
IMG_PATH = '/home/fiorapirri/Documents/workspace/YoutubeVIS21/valid/'

FEAT_PATH = ''

ANNOTATIONS = DATASET_['annotations']

CATEGORIES = config.CLASS_YTVIS21


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SHOW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# OFFSET = 20



""" Cycles over all videos """

#def mainTrack():
Xds, idx = get_whole_single_video(ANNOTATIONS)
Xds-=1

len_edges = cfg.LEN_EDGES
edges = np.linspace(0,255,len_edges) 
#JJ = np.random.randint(0, max(Xds), 300)
TRACKED_ANNOTATIONS = []
for  FROM_idx in tqdm(Xds, desc='processing: '):
     FROM, TO = idx[FROM_idx], idx[FROM_idx+1] if FROM_idx+1 < len(idx) else len(ANNOTATIONS)-1

     video_id = FROM_idx
     video_X = ANNOTATIONS[FROM:TO]
     
     length = ANNOTATIONS[FROM]['len_sequence']
     video_id = ANNOTATIONS[FROM]['video_id']

     # Prediction
     bboxes =   [X.get('pred_bboxes' ) for X in video_X if len(X)>0]
     categories_all = [np.array(X['pred_categories']) for X in video_X ]
     mask_ps = [X['pred_segmentations'] for X in video_X]
     instances = [X['pred_instances'] for X in video_X]
     confidence_all = [X['pred_confs'] for X in video_X]
     
     
     ### imms
     imms =   [np.array(Image.open(os.path.join(IMG_PATH,X['file_name']))) for X in video_X]
     """only for the tracker if ground truth """
     testAcc = False
#     Dat_dict =[]
     print('video:', video_id)
     start = time.time()
     
     if testAcc:
        ### Ground truth
        gt_bboxes =   [X.get('gt_bboxes' ) for X in video_X if len(X)>0]
        gt_categories_all = [np.array(X['gt_categories']) for X in video_X ]
        gt_mask_ps = [X['gt_segmentations'] for X in video_X]
        gt_instances = [X['gt_instances'] for X in video_X]
        ## pseudo
        confidence_all = [X['pred_confs'] for X in video_X]
         
        dictV, out_classes, schemaSH, cats = main_search_and_prediction(gt_categories_all, gt_bboxes, imms,  gt_mask_ps,
                                      gt_instances, confidence_all, gt = True)
        
        ids_gt =   [X['gt_ids'] for X in video_X]
        acc = test_accuracy(schemaSH, out_classes, ids_gt, cats)
        print('current accuracy:', acc)
     # for prediction
     else:
         dictV, _, _ ,_= main_search_and_prediction(categories_all, bboxes, imms,  mask_ps,
                                      instances, confidence_all, video_id)
     end = time.time()
     
#     Dat_dict.append([video_id, dictV])
     
     for elem in dictV:
         sample = dict()
         sample['video_id'] = video_id
         sample['segmentations'] = [None for i in range(length)]
         tracked_categs = []
         tracked_scores = []
         for attrib in elem['attrib']:
             if list(attrib['rle_mask'])[0] != 'None':
                 tracked_categs.append(attrib['categ'])
                 tracked_scores.extend(list(attrib['conf']))
                 sample['segmentations'][attrib['time']] = list(attrib['rle_mask'])[0]
         if any([i!=None for i in sample['segmentations']]):
             sample['category_id'] = int(np.bincount(tracked_categs).argmax())
             sample['score'] = float(np.mean(tracked_scores))
             TRACKED_ANNOTATIONS.append(sample)
     
     print('time to process', end-start)
   
with open('10_results_valid_new_graph.json', 'w') as f:
    f.write(json.dumps(TRACKED_ANNOTATIONS))
       
            
#if __name__ == "__main__":
#      mainTrack()       
     