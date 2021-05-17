# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:02:06 2021

@author: fiora
"""
from graph_tracking.util_track import file_reader, rle_decoding, draw_bbox,  get_whole_single_video

DATASET = file_reader('newinstances.json')

# ANNOTATIONS KEYS 
#['annotation_id', 'video_id', 'len_sequence', 'num_sequence', 
#'file_name', 'height', 'width', 'gt_instances', 'gt_ids', 'gt_bboxes', 
#'gt_areas', 'gt_categories', 'gt_segmentations', 'pred_instances', 'pred_categories', 
#'pred_bboxes', 'pred_segmentations', 'pred_features']

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# fill the paths if are different from the one in the json
IMG_PATH = 'F:\\AVA\\bipartite_tracker\\DaSimone\\'

FEAT_PATH = 'F:\\AVA\\bipartite_tracker\\DaSimone\\'

ANNOTATIONS = DATASET['annotations']

CATEGORIES = dict([(int(i),k) for i,k in zip(DATASET['categories'].keys(),DATASET['categories'].values())])

LEN_EDGES = 120
LEN_IMM = 32*32*3
FF = 8*29*36 # 5*5*36


   
TO_SAVE = 'F:\\AVA\\bipartite_tracker\\ExampleTracker\\Dati'
TO_SAVE_IMM = 'F:\\AVA\\bipartite_tracker\\ExampleTracker\\images'