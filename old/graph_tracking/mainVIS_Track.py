#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:16:12 2021

@author: fiora
"""
from graph_tracking.util_track import file_reader, rle_decoding, draw_bbox,  get_whole_single_video

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
from graph_tracking.bipartiteG import bipartite_matching
from graph_tracking.unsup_tracking_VIS import *
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

LEN_EDGES = 120
LEN_IMM = 32*32*3
FF = 8*29*36 # 5*5*36

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SHOW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# OFFSET = 20
    
def main_ass_track(imms1, imms2, bbx1, bbx2, video_X, edges, len_edges):
    print('Loading....')
    instances = [X['pred_instances'] for X in video_X]
    memX, submemX, idx_objs, max_preds = get_memories(instances)
    Hist = []
#    update = False
    for  jj, (i1, i2, bb1, bb2) in enumerate(zip(imms1,imms2, bbx1, bbx2)):

        print('start processing image '+ str(jj+1))

        if (jj +1 < len(imms1)+1) and (len(bb1) > 0) :

              bb1 =  resort_bbs(bb1,i1.shape)
              bb2 =  resort_bbs(bb2,i2.shape)
              bb1 = bb_embeds(bb1, max_preds, i1.shape)
              bb2 = bb_embeds(bb2, max_preds, i1.shape)
              frame_hist = np.zeros((np.int(np.max([len(bb1), len(bb2)])),len_edges*3-9+4))
              
              frame_hist1 = gen_hist(bb1, i1, frame_hist, edges, len_edges,1)
              frame_hist2 = gen_hist(bb2, i2, frame_hist,  edges, len_edges,2)
              
              """ V returns the similarity"""
              V = compare_frames(frame_hist1, frame_hist2)
              
              # print(V)
              V[V<0] = 0
              """ permx return the choice for each row try on FROM_idx =420 is interesting"""
              permx, vx =  bipartite_matching(V)

              vals = V*permx 
              
              ### superposition
#              rem_update =[]
#              if update:
#                 prev_vals = memX.get(jj-1)
#                 for q in to_update:
#                     bbW = prev_vals[q]['bb']
#                     frame_histP = gen_hist([bbW], i1, frame_hist, edges, len_edges,1)
#                     Vp = compare_frames(frame_histP, frame_hist2)
#                     permP, _ =  bipartite_matching(Vp)
#                     valP = Vp*permP  
#                     if valP > vals:
#                         rem_update.append(1)
#                     else:
#                         rem_update.append(2)
              """ corr generates the association """       
              corr = np.zeros((np.max([len(bb1),len(bb2)]),np.max([len(bb1),len(bb2)])), dtype = np.int32)
              idx_objs[jj+1,:] = -1
#              to_update = []
              numsB = np.max([len(bb1), len(bb2)])
              for kk in range(numsB):
#                  print(vals[kk,:])
                  ttx, = np.where((vals[kk,:] == np.max(vals[kk,:])) & (np.max(vals[kk,:]) > 0.7)  )

                  if len(ttx) > 0:
                     corr[kk,ttx] = 1
                     idx_objs[jj+1,kk] = ttx
                  else:
                      # update =True
                      # to_update.append(kk)
                      # key = jj
                      corr[kk, kk] = -1
                      idx_objs[jj+1,kk] = -1

              """ Want to see correspondence """        
              #show_bb_after(i1,i2, jj, jj+1, bb1, bb2, corr)         
#              if update:
#                   for q in to_update:
#                       submemX.update({q:{'bb': bb1[q]}} )
#                   memX.update({jj: submemX})
              """ The history is used for updating"""
              Hist.append(corr)
        else:
            idx_objs[jj+1,:] = -1
    return Hist, idx_objs, corr
              
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ids, idx = get_whole_single_video(ANNOTATIONS)


""" Cycles over all videos """
TRACKED_ANNOTATIONS = []

for FROM_idx in ids[:1]:
    FROM_idx = FROM_idx-1
    """ Get the video"""
    ##np.random.choice(len(idx))
    FROM = idx[FROM_idx]
    TO =  idx[FROM_idx+1]
    # FROM = np.random.choice(len(ANNOTATIONS)-OFFSET) # i.e. 48782
    # TO = FROM + OFFSET
    video_X = ANNOTATIONS[FROM:TO]
    print('processing video '+str(video_X[0]['video_id']))
    """ Just to see how it works on many instances """
    INSTp = [X['pred_instances'] for X in video_X]
#    INST_gt= [X['gt_instances'] for X in video_X]
    
    max_instP= np.max(INSTp)
#    max_instGT= np.max(INST_gt)   
    
    """ Get bounding boxes for the current video"""
    bboxes =   [X['pred_bboxes'] for X in video_X ] #if len(X)>0
    max_len = np.max([len(x) for x in bboxes])
    bbx1 = bboxes[:-1]
    bbx2 = bboxes[1:]
   
    """ Images and dimension of the histogram/distribution"""
    
    edges = np.linspace(0,255,LEN_EDGES)
    imms =   [np.array(Image.open(os.path.join(IMG_PATH,X['file_name']))) for X in video_X]
    imms1 = imms[:-1]
    imms2 = imms[1:]
    
    '''associate also last frame'''
#    imms1.append(imms[-1])
#    imms2.append(imms[-1])
#
#    bbx1.append(bboxes[-1])
#    bbx2.append(bboxes[-1])
#    
    imms =[]
   
    """For ACcuracy: GT  index of the objects in the ground truth normalized for each video """
#    cat_idx = [X['gt_ids'] for X in video_X]
#    objs_keys = np.unique(np.concatenate(cat_idx).ravel()).astype(np.int)
#    max_idx = max( objs_keys)
#    cat_idx_norm = [list((x//max_idx).astype(np.int)) for x in cat_idx]
#    keys_objs = list(np.unique(np.concatenate(cat_idx_norm).ravel()).astype(np.int))
#    
    
    """PREDS initial data and create a memory"""
    
    hist, idx_objs, corr = main_ass_track(imms1, imms2, bbx1, bbx2, video_X, edges, LEN_EDGES)
    
    if idx_objs.shape[0]>0 and idx_objs.shape[1]>0:
        """ get varia """
        video_idx = video_X[0]['video_id']
        segments = [X['pred_segmentations'] for X in video_X]
        categs = [X['pred_categories'] for X in video_X]
        confs = [X['pred_confs'] for X in video_X]
        subjects_ids = idx_objs[0,:]
        
        idx_objs = idx_objs[1:,:] # remove zeros
        
        idx_objs = idx_objs[:-1,:] # remove zeros
        
        for i in subjects_ids:
            
            indices = idx_objs[:,i]
            
            tracked_masks = []
            tracked_scores = []
            tracked_categs = []
            
            assert len(indices) == len(bboxes[1:])
            assert len(indices) == len(segments[1:])
            assert len(indices) == len(confs[1:])
            assert len(indices) == len(categs[1:])
            
            for j, seg, conf, cat in zip(indices, segments[1:], confs[1:], categs[1:]):
                
                if j>-1:
#                    if j<len(seg):
                    tracked_masks.append(seg[j])
#                    if j<len(conf):
                    tracked_scores.append(conf[j])
#                    if j<len(cat):
                    tracked_categs.append(cat[j])
                else:
                    tracked_masks.append(None)
            

            
            if len(tracked_scores)>0:
                new_tracked_instance = {}
                new_tracked_instance['video_id'] = video_idx
                new_tracked_instance['category_id'] = int(np.bincount(tracked_categs).argmax())
                new_tracked_instance['segmentations'] = tracked_masks
                new_tracked_instance['score'] = float(np.mean(tracked_scores))
                TRACKED_ANNOTATIONS.append(new_tracked_instance)
        
with open('results.json', 'w') as f:
    f.write(json.dumps(TRACKED_ANNOTATIONS))

#%%%%%%%%%%%
    