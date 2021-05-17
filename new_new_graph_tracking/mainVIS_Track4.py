#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:16:12 2021

@author: fiora
"""
from new_new_graph_tracking.util import file_reader, rle_decoding, draw_bbox,  get_whole_single_video

import numpy as np
import os
from numpy import matlib
import json

import timeit

from PIL import Image, ImageOps
from matplotlib import pyplot as plt 
import matplotlib.patches as patches
import tensorflow as tf
import tensorflow_probability as tfp
import itertools as itt
import pandas as pd
from new_new_graph_tracking.bipartiteG import bipartite_matching
from new_new_graph_tracking.unsup_tracking_VIS_new import *
import copy
import new_new_graph_tracking.config_VIS as cfg
from new_new_graph_tracking.subcompare_newBFBG import subcompare_frames,  main_search 
import config 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOAD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DATASET_ = file_reader('pred_valid_instances.json')

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

TRACKED_ANNOTATIONS = []


""" Cycles over all videos """

ids, idx = get_whole_single_video(ANNOTATIONS)
ids-=1

len_edges = cfg.LEN_EDGES
# len_imm = cfg.LEN_IMM
edges = np.linspace(0,255,len_edges) 
   
   
for FROM_idx in ids:
    start = timeit.timeit()

    """ Get the video"""
    ##np.random.choice(len(idx))
    FROM, TO = idx[FROM_idx], idx[FROM_idx+1] if FROM_idx+1 < len(idx) else len(ANNOTATIONS)-1

    # FROM = np.random.choice(len(ANNOTATIONS)-OFFSET) # i.e. 48782
    # TO = FROM + OFFSET
    video_X = ANNOTATIONS[FROM:TO]
    # print('starting video', video_X[0]['annotation_id'],' ->', FROM_idx,'/',len(idx))
    
    """ Just to see how it works on many instances """
    INSTp = [X['pred_instances'] for X in video_X]
#    INST_gt= [X['gt_instances'] for X in video_X]
   
    """ Get bounding boxes for the current video"""
    bboxes =   [X.get('pred_bboxes' ) for X in video_X if len(X)>0]
    max_len = np.max([len(x) for x in bboxes])
    pred_categories_all = [np.array(X['pred_categories']) for X in video_X ]
#    imms =   [np.array(Image.open(X['file_name'])) for X in video_X]
    imms =   [np.array(Image.open(os.path.join(IMG_PATH,X['file_name']))) for X in video_X]

    ### SHOW
    # for i, (img, a) in enumerate(zip(imms, pred_categories_all)):
    #                              # print(i, a)
    #     pred_bboxes = bboxes[i] 
    #     pred_bboxes = np.array(pred_bboxes)
    #     pred_bboxes[:,[2,3]] = pred_bboxes[:,[0,1]] + pred_bboxes[:,[2,3]]
    #     #pred_segmentations = np.array([rle_decoding(m) for m in a['pred_segmentations']])
    #     pred_categories =  a
    #     draw_bbox(img, i, box=pred_bboxes, conf=None, class_id=pred_categories, 
    #               mask=None, class_dict=CATEGORIES, mode='PIL')

    # print('Loading....')
    instances = [X['pred_instances'] for X in video_X]
    idx_objs, max_preds = make_idx_objs(instances)
    
    idx_objs, results = main_search(bboxes, pred_categories_all, imms, idx_objs, max_preds, edges, len_edges, video_X)
    
 
         
    print(idx_objs)
    end = timeit.timeit()
    print(end - start)
   
    if idx_objs.shape[0]>0 and idx_objs.shape[1]>0:
        """ get varia """
        video_idx = video_X[0]['video_id']
        segments = [X['pred_segmentations'] for X in video_X]
        categs = [X['pred_categories'] for X in video_X]
        confs = [X['pred_confs'] for X in video_X]
        subjects_ids = idx_objs[0,:]
        
#        idx_objs = idx_objs[1:,:] # remove zeros
        
#        idx_objs = idx_objs[:-1,:] # remove zeros
        
        for i in subjects_ids:
            
            indices = idx_objs[:,i]
            
            tracked_masks = []
            tracked_scores = []
            tracked_categs = []
            
            assert len(indices) == len(bboxes)
            assert len(indices) == len(segments)
            assert len(indices) == len(confs)
            assert len(indices) == len(categs)
            
            for j, seg, conf, cat in zip(indices, segments, confs, categs):
                
                if j>-1:
                    if j<len(seg):
                        tracked_masks.append(seg[j])
                        tracked_scores.append(conf[j])
                        tracked_categs.append(cat[j])
                    else:
                        tracked_masks.append(None)
                else:
                    tracked_masks.append(None)
            

            
            if len(tracked_scores)>0:
                new_tracked_instance = {}
                new_tracked_instance['video_id'] = video_idx
                new_tracked_instance['category_id'] = int(np.bincount(tracked_categs).argmax())
                new_tracked_instance['segmentations'] = tracked_masks
                new_tracked_instance['score'] = float(np.mean(tracked_scores))
                TRACKED_ANNOTATIONS.append(new_tracked_instance)
        
with open('results_valid_new_graph.json', 'w') as f:
    f.write(json.dumps(TRACKED_ANNOTATIONS))
       
            
      
     