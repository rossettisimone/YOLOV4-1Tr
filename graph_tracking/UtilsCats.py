# -*- coding: utf-8 -*-
"""
Created on Mon May 24 08:18:48 2021

@author: Utente
"""

import copy
import PIL
from PIL import Image
import itertools
from collections import Counter
from operator import itemgetter
from graph_tracking.bipartiteG import bipartite_matching
from collections import deque
import random
import string
from functools import reduce
# from unsup_tracking_VIS_new import *
from graph_tracking.compare_masks_bbs import *
edges = np.linspace(0,255,cfg.LEN_EDGES)
len_edges = cfg.LEN_EDGES  
#
def initialize_vars(schemai):
        #VX =[]
        m,n =  schemai.shape
        locs= np.arange(0,n,1)
        visited = []
        tovisit = deque()
        time_locations = []
        first_row_data = [x for x in schemai[0,:] if x != -1]
        vn = locs[0:len(first_row_data)]
        nv = locs[len(first_row_data):n]
        visited.append(vn)
        tovisit.extend(nv)
        # tl = [i for i, j in enumerate(vn) if j in vn]
        # time_locations.append(tl)
        # removed = [-1]
        # Mem =[]
        return visited, tovisit  #, removed, Mem, time_locations, VX
            
#   
def make_idx_objs(instances, Tt, Tb,  max_preds, schemaC ):
    T = [len(x) for x in Tt]
    n = max(T)
    idx_objs = np.ones((len(instances), n), dtype = np.int)*-1
    tt, = np.where(schemaC[0,:]>0)
    init = np.array(Tb[0]) 
    idx_objs[0,tt] = tt
    return idx_objs
#
def gen_mem_prob(instances, Tt, Tb, schemaC):
    obj_probs = np.zeros((len(schemaC), np.max(instances)))
    tt, =np.where(schemaC[0,:] > 0)
    obj_probs[0,tt]  =1 
    
    # Tt = tree_transitions[c]
    # Tx = [x for y  in Tt for x in y]
    for j in range(len(Tt)):
        bp = Tt[j]
        if len(bp) > 0:
            info_curr = len(Tb[j])
            info_next = len(Tb[j+1])
            
            fromc, toc = np.where(bp > 0)
            fromc = fromc.tolist()
            toc = toc.tolist()
            for x, y in zip(fromc, toc):
                obj_probs[j+1,y] = bp[x,y]
            missed = np.setdiff1d(Tb[j+1],np.nonzero(obj_probs[j+1,:]) )
            
    return obj_probs
            
##
def count_catg(predC, max_preds):
    schema = np.ones((len(predC)+1, max_preds), dtype = np.int)*-1
    schema[0,:] = np.arange(0,max_preds,1)
    for j, preds in enumerate(predC):
        schema[j+1,0:len(preds)] = preds
    ## to add in first line 
    schemaSH = schema[1:,:]
    return  schemaSH

def make_square(V):
    a,b =  V.shape
    V1 = np.zeros((np.max([a,b]), np.max([a,b])))
    if a >b :
        V1[:,0:b] =V
    else:
        V1[0:a,:] = V
    return V1

def dice_sorens(mask1,mask2):
    m1C = 1-mask1
    m2C = 1-mask2
    TP = np.count_nonzero(mask1*mask2)
    TN = 0.05*np.count_nonzero(m1C*m2C)
    FP= 0.05*np.count_nonzero(m1C*mask2)
    FN= 0.05*np.count_nonzero(mask1*m2C)   
    
    ## metrics
    dice = 2*TP/(FP + 2*TP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    # diff_area = ((TP + FP) - (TP + FN)) 
    return  precision

# vvs, vvh, 
def get_B_matrix(bb1,bb2, mms1, mms2, i1, i2, distrpV,  vel, cval, ns,  edges, len_edges, VX, lev1, lev2):
      # frame_hist = np.zeros((np.int(np.max([len(bb1), len(bb2)])),len_edges*3-9+4))
      ## generate score for masks
      t_mask1, t_mask2, i_mask1, i_mask2, U = compare_masks_new(mms1, mms2, i1,i2, lev1, lev2)
      # U = compare_masks_new(mms1, mms2, i1,i2, lev1, lev2)
      if len(U)==0:
          U = np.array([[],[]])
      U1 = make_square(U)
      
      ## generate scores for featuress in bbs
      frame_hist1 = gen_hist(bb1, i_mask1, t_mask1,  i1,  edges, len_edges,1)
      frame_hist2 = gen_hist(bb2, i_mask2, t_mask2,  i2,  edges, len_edges,2)
     
      V = compare_frames_new(frame_hist1, frame_hist2, lev1, lev2)
      if len(V)==0:
          V = np.array([[],[]])
      V1 = make_square(V)
      
      
                  
      score_trans = 1-np.array(distrpV.cdf(vel))
      trans_mat = np.zeros_like(U1)
      for hi in range(len(lev1)):
              for hj in range(len(lev2)):
                  # print(score_trans[hi,hj])
                  trans_mat[lev1[hi],lev2[hj]] = score_trans[hi,hj]
                  
      # score_trans2 = 1-np.array(distrpD.cdf(theta))
      # trans_matd = np.zeros_like(U1)
      # for hi in range(len(lev1)):
      #         for hj in range(len(lev2)):
      #             # print(score_trans[hi,hj])
      #             trans_matd[lev1[hi],lev2[hj]] = score_trans2[hi,hj]
      w = i_mask1 + i_mask2
      size_people = [len(x) for x in w]            
      if (cval[0] == 26  and   np.mean(size_people) >= 35) or (ns <= 40) :  
          # print('not using motion')
          no_motionX = np.maximum.reduce([V1,U1]) 
          permz, vz =  bipartite_matching(no_motionX)
          VX = permz*no_motionX
            
      else:
          # print('using motion')
          motionX = np.maximum.reduce([V1*trans_mat,U1*trans_mat]) 
          permz, vz =  bipartite_matching(motionX)
          VX = permz*motionX
      VX = np.clip(VX,0,1)
      
      
      return VX, U1, V1

       
def id_generator(size=2, chars=string.ascii_uppercase):
    return ''.join(random.choice(chars) for _ in range(size))

def centroidX(bb):
    cc = 0.5*(np.array([bb[2]-bb[0], bb[3]-bb[1]]))
    center = np.array([bb[0]+cc[0], bb[1]+cc[1]])
    return center

# ''.join(random.choices(string.ascii_uppercase, k=2))
def get_bbs(bboxes, L, ti):
            level1 = L[ti]
            level2 = L[ti+1]
            bbox1 = bboxes[ti]
            bbox2 = bboxes[ti+1]
            bbi1 = [np.array(bbox1[h]) for h in level1]
            bb1 = [np.concatenate([bbx[0: 2], bbx[2:4] + bbx[0:2]]) for bbx in bbi1]
            bbi2 = [np.array(bbox2[h]) for h in level2]
            bb2 = [np.concatenate([bbx[0: 2], bbx[2:4] + bbx[0:2]]) for bbx in bbi2]
            centr1 = list(map(centroidX, bb1))
            centr2 = list(map(centroidX, bb2))
            # theta = [[((y2-y1)/((x2-x1)+0.01))*180/np.pi for (x2,y2) in centr2] for (x1,y1) in centr1]
            vel = [[np.linalg.norm(x-y) for y in centr2] for x in centr1]
          
            return bb1, bb2, level1, level2, centr1, centr2, vel

##
def simple_IOU(im1,im2):
   IOU =  np.sum(im1*im2)/(0.1*np.sum(im1+im2))
   return IOU

#
##
def get_masks_pred(masks, confs, L, ti):
            level1 = L[ti]
            level2 = L[ti+1]
            mmask1 = masks[ti]
            mmask2 = masks[ti+1]
            mms1 = [np.array(mmask1[h]) for h in level1]
            mms2 = [np.array(mmask2[h]) for h in level2]
            conf1 =  confs[ti]
            conf2 =  confs[ti+1]
            conf1 = [np.array(conf1[h]) for h in level1]
            conf2 = [np.array(conf2[h]) for h in level2]
            # IOU =  [[simple_IOU(x,y) for x in mms1] for y in mms2]
            return mms1, mms2, conf1, conf2, level1, level2
##
def get_masks(masks, confs, L, ti):
            level1 = L[ti]
            level2 = L[ti+1]
            mmask1 = masks[ti]
            mmask2 = masks[ti+1]
            mms1 = [np.array(mmask1[h]) for h in level1]
            mms2 = [np.array(mmask2[h]) for h in level2]
             
            conf1 = [np.random.rand(h) for h in level1]
            conf2 = [np.random.rand(h) for h in level2]
            # IOU =  [[simple_IOU(x,y) for x in mms1] for y in mms2]
            return mms1, mms2, conf1, conf2, level1, level2
##
#### Obtain the masks
def build_masks(m0):
        h, w = m0.get('size',[720,1280])
        rle_arr = m0.get('counts',[0])
        rle_arr = np.cumsum(rle_arr)
        indices = []
        extend = indices.extend
        list(map(extend, map(lambda s,e: range(s, e), rle_arr[0::2], rle_arr[1::2])));
        binary_mask = np.zeros(h*w, dtype=np.uint8)
        binary_mask[indices] = 1     
        BM = binary_mask.reshape((w, h)).T  
        return BM
    
    
def get_all_masks(masks_ps):
    el_of_masks = list( map(lambda x: map( lambda y: build_masks(y), x) , masks_ps))
    all_masks = [list(x) for x in el_of_masks]
    return all_masks
    
def get_rle(mask_rle, L, ti):
            level1 = L[ti]
            mrle =mask_rle[ti]
            mmrle1 = [mrle[h] for h in level1]
            return mmrle1, level1
## confidence

    
    