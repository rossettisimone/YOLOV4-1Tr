#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:09:19 2021

@author: fiorapirri
"""


#%%%%%%%%%%%%%%%%%%%%%%%%%%% BUILD ENV %%%%%%%%%%%%%%%%%%%%%%%%%%%%§%%%%%%
import env


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import config as cfg
import tensorflow as tf
from loader_ytvos import DataLoader 
from model import get_model
import numpy as np
import json

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

GATING_TAU = 99e-2
LONELY_GAMMA = 1e-2

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

def gating(A: np.ndarray) -> (np.ndarray, np.ndarray):
    # Gating: ignore all
    # associations whose cost is
    # higher than a threshold
    m = np.arange(0,A.shape[0])
    n = np.argmin(A,axis=1)
    a_mn = A[m,n]
    # if observations don't pass the gating,
    # it means they are new ones, thus put the 
    # indices in a list
    indices_ko = np.where(a_mn >= GATING_TAU)[0]
    new_indices = np.take(m, indices_ko, axis=0)
    # if observations pass the gating,
    # save the associations
    indices_ok = np.where(a_mn < GATING_TAU)[0]
    m = np.take(m, indices_ok, axis=0)[...,None]
    n = np.take(n, indices_ok, axis=0)[...,None]
    a_mn = np.take(a_mn, indices_ok, axis=0)[...,None]
    associations = np.concatenate([m,n,a_mn],axis=-1)

    return associations, new_indices

def best_friend(associations: np.ndarray, A: np.ndarray) -> (np.ndarray, np.ndarray):
    # Best friends: an association should be the best (i.e. minimum) of
    # both row and column landmark in the state
    if associations.shape[0] == 0 : return np.array([]).reshape(0,2), np.array([])
    # min by columns
    mm = np.argmin(A,axis=0)
    # min by rows
    m, n = associations[:,0].astype(int), associations[:,1].astype(int)
    # take those who are in n 
    mm = np.take(mm,n)
    # take those obs whose a_mn is minimum of rows and cols
    indices = np.where(m==mm)[0]
    associations = np.take(associations, indices, axis=0)
    # doubtful associations are the ones that are not best friend,
    # save unassigned measurements
    indices = np.where(m!=mm)[0]
    pruned = np.take(m, indices, axis=0)
    return associations, pruned

def lonely_best_friend(associations: np.ndarray, A: np.ndarray) -> (np.ndarray, np.ndarray):
    # Lonely best friends: one measurement should be
    # only assigned to one single landmark.
    if associations.shape[0] == 0 : return np.array([]).reshape(0,2), np.array([])
    m, n, a_mn = associations[:,0].astype(int), associations[:,1].astype(int), associations[:,2]
    # exclude current minimum
    A[m,n] = np.inf
    # now take the second minimum
    a_min_row = np.take(np.min(A,axis=1),m)
    a_min_col = np.take(np.min(A,axis=0),n)
    # check weather current associations are in safe reigion w.r.t second minimum
    cond = np.logical_or(a_min_row - a_mn < LONELY_GAMMA, a_min_col - a_mn < LONELY_GAMMA)
    indices = np.where(np.logical_not(cond))[0]
    associations = np.take(associations, indices, axis=0)
    # save observations that didn't pass third gating
    indices = np.where(cond)[0]
    pruned = np.take(m, indices, axis=0)
    return associations, pruned
    
def prune_heuristics(A: np.ndarray = np.identity(2,dtype=np.float32)) -> (np.ndarray, np.ndarray, np.ndarray):
    # Bad associations are EVIL. To avoid the above cases
    # we can use three heuristics
    # 1. Gating
    associations, new_indices = gating(A = A)
    # 2. Best friends
    associations, bf_pruned_indices = best_friend(associations = associations, A = A)
    # 3. Lonely Best Friend
    associations, lbf_pruned_indices = lonely_best_friend(associations = associations, A = A)
    # pruned associations are the ones that may lead to bad associations
    doubtful_indices = np.concatenate([bf_pruned_indices, lbf_pruned_indices],axis=0)

    return associations, new_indices, doubtful_indices

def associate(mask1: np.ndarray, mask2: np.ndarray, verbose: int = 1) -> (np.ndarray, np.ndarray, np.ndarray):
    # with a greedy algorithm compute Omega L2 Norm cost matrix
    A = 1-compute_overlaps_masks(mask1, mask2) # minimum problem
#    print(A)
    # use heuristics to avoid bad associations
    associations, new_indices, doubtful_indices = prune_heuristics(A = A)

    if verbose :
        print(f'match={associations.shape[0]}, new={new_indices.shape[0]}, pruned={doubtful_indices.shape[0]}')
    
    return associations, new_indices, doubtful_indices

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import matplotlib.pyplot as plt
from loader_ytvos import DataLoader 
#import time
from utils import file_reader, show_infer, compute_mAP, draw_bbox, show_mAP, encode_labels, crop_and_resize,xyxy2xywh, decode_ground_truth, unmold_mask_batch, rle_encoding, rle_decoding
 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from typing import Optional, List

class State(object):
    def __init__(self, rle_mask: dict() or np.ndarray, conf: float, cat: int):
        self.rle = rle_mask
        self.conf = conf
        self.cat = cat
        if isinstance(self.rle, np.ndarray):
            self.rle = rle_encoding(self.rle)
    
    def get_mask(self):
        return rle_decoding(self.rle)
        
class Instance(object):
    def __init__(self, video_id: int, idx: int, state: State, t: int, T: int):
        self.video_id = video_id
        self.id = idx
        self.state = state
        self.history = []
        self.valid = 0
        self.t = 0
        self.T = T
        self.fill(t-1)
        self.update(state)
        
    def __len__(self):
        return len(self.history)
    
    def update(self, state: Optional[State] = None) -> None:
        self.history.append(state)
        self.t += 1
        if (state != None):
            self.valid += 1
            self.state = state
    
    def fill(self, t: int) -> None:
        if t > 0:
            for i in range(t):
                self.update(None)
    
    def get_mask(self) -> np.ndarray:
        return self.state.get_mask()
    
    def get_state(self) -> State:
        return self.state
    
    def end(self) -> None:
        for i in range(self.T-self.t):
            self.update(None)
            
    def get_summary(self, rle_format: bool = True) -> dict:
        summary = dict()
        summary['video_id'] = int(self.video_id)
        summary['category_id'] = int(np.bincount([s.cat for s in self.history if s!=None]).argmax())
        summary['segmentations'] = list([s.rle if s!=None else s for s in self.history])
        summary['score'] = float(np.mean([s.conf for s in self.history if s!=None]))
        return summary
         
class InstanceDealer(object):
    def __init__(self, video_id: int, T: int):
        self.instances = []
        self.video_id = video_id
        self.T = T
        self.t = 0
    
    def __len__(self):
        return len(self.instances)
        
    def add(self, state: State) -> None:
        self.instances.append(Instance(self.video_id, len(self.instances), state, self.t, self.T))
    
    def get_states(self) -> List[State]:
        return list([s.get_state() for s in self.instances])
        
    def update(self, states: List[np.ndarray]) -> None:
        self.t += 1
        for index,state in enumerate(states):
            self.instances[index].update(state)
    
    def end_states(self) -> None:
        for instance in self.instances:
            instance.end()

    def print_states(self) -> None:
        intro = ('='*2 + ' video %d -> id: states ' + '='*2)%(self.video_id)
        print('='*len(intro))
        print(intro)
        print('='*len(intro))
        for index,instance in enumerate(self.instances):
            print('= %d: %d'%(instance.id, instance.valid))
        print('='*len(intro))
        print('='*len(intro))
    
    def get_summaries(self, tolerance: int = 1, min_score: float = 0.5) -> List[dict]:
        summaries = [i.get_summary() for i in self.instances if i.valid > tolerance]
        summaries = [obj for obj in summaries if obj['score'] > min_score]
        return summaries
        
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
SEGMENTS = file_reader('pred_test_instances.json')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd

def get_whole_single_video(annot):
    df = pd.DataFrame(annot)
    video_num = df["video_id"].to_numpy()
    ids, idx = np.unique(video_num, return_index = True)
    return ids, idx

ids, idx = get_whole_single_video(SEGMENTS['annotations'])

ids-=1

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TRACKED = []

for idd in ids:
    
    FROM, TO = idx[idd], idx[idd+1] if idd+1 < len(idx) else len(SEGMENTS['annotations'])-1
    
    length_video = TO-FROM
    
    video_id = SEGMENTS['annotations'][FROM]['video_id']
    
    dealer = InstanceDealer(video_id, length_video)
    
    for i, annotation in enumerate(SEGMENTS['annotations'][FROM:TO]):
        
        new_states = [State(s,c,d) for s,c,d in zip(annotation['pred_segmentations'],\
                      annotation['pred_confs'], annotation['pred_categories']) if not s==None]
                
        old_states = dealer.get_states()
        
        if len(new_states)>0:
            
            if len(old_states)>0:
                associations, new_indices, doubtful_indices = associate(np.array([s.get_mask() for s in new_states]).transpose(1,2,0), \
                                                                        np.array([s.get_mask() for s in old_states]).transpose(1,2,0), verbose = 0)
#                print(associations)
                for row in associations:
                    old_states[int(row[1])] = new_states[int(row[0])]
                for index in list(set(range(len(old_states))).difference(list(associations[:,1].astype(int)))):
                    old_states[index] = None
                dealer.update(old_states)
                for index in new_indices:
                    dealer.add(new_states[index])
            else:
                for s in new_states:
                    dealer.add(s)
        else:
            for index in range(len(old_states)):
                old_states[index] = None
                dealer.update(old_states)
        
    dealer.end_states()
    
    dealer.print_states()
    
    instances = dealer.get_summaries(tolerance = 3, min_score = 0.6)
    
    TRACKED.extend(instances)
    
with open('results-test-heuristics2.json', 'w') as f:
    
    f.write(json.dumps(TRACKED))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for state in dealer.instances[1].history:
    if state != None:
        plt.imshow(state.get_mask())
        plt.show()
    else:
        print('None')
        