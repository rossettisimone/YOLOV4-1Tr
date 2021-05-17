# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:45:53 2021

@author: Utente
"""
import copy
import itertools
from collections import Counter
from operator import itemgetter
from new_new_graph_tracking.bipartiteG import bipartite_matching
from new_new_graph_tracking.unsup_tracking_VIS_new import *
edges = np.linspace(0,255,cfg.LEN_EDGES)
len_edges = cfg.LEN_EDGES  

def which_faked(L):
    h = []
    j = []
    L1 = [list(x) for x in L]
    for i, x in enumerate(L1):
        if sum(x) == -2:
           h.append(x)
           j.append(i)
    return h, j
         
def update_list(trace, idx, bb2, i2, bboxesn, newL, kk):
    trace.append(idx) 
    if  type(bb2) == list:    
         bbL =  bb2[idx]     
    else:             
      bbL = bb2[idx].astype(np.float)
    bb1 = [bbL]
    i1 = i2.copy()
    if len(bbL)>0:
       newL = add_vals(bbL, bboxesn, newL, kk)      
    return idx, bb1, i1, bboxesn, newL, trace 

def add_vals(bbL, bboxesn, newL, kk):
    if type(bbL) != list:
       bbLi = np.concatenate([bbL[0: 2], bbL[2:4] - bbL[0:2]] )
    else: bbLi = bbL
    if list(bbLi) in bboxesn[kk]:
       # A = [x for x in bboxesn[kk-1] if x!= list(bbLi) else [-1,-1,-1,-1]]]
       A = [[-1,-1, 0, 0] if x == list(bbLi)  else x for x in bboxesn[kk] ]
       newL.append(A)
       return newL

def make_square(V):
    a,b =  V.shape
    V1 = np.zeros((np.max([a,b]), np.max([a,b])))
    if a >b :
        V1[:,0:b] =V
    else:
        V1[0:a,:] = V
    return V1

def associate(result,  schemaSH, kk, idx_objs, i = None):
     print(i)
     if kk ==0:
            index_idx = result[kk,i]
            index_val = idx_objs[kk,i]
            cats = schemaSH[kk, i]
     else:
        index_idx, = np.where(result[kk-1,:] != 0)
        index_val, = np.where(idx_objs[kk-1,:] != -1)
        cats = schemaSH[kk-1, index_idx]
     return index_idx, index_val, cats


# result = np.row_stack([schemaSH[0,:],idx_objs])
# result = np.zeros_like(schemaSH)
## cats = np.sort(schemaSH[0,:])
# result[0,:] = schemaSH[0,:]
# idx_objs, max_preds = make_idx_objs(instances)

#  bboxesn =newList.copy()
#  j=0
def subcompare_frames( bboxesn, schema, schemaSH, assoc, result, imms, idx_objs, edges, len_edges):
        j =0
        i1 = imms[j]
        bb1 = resort_bbs(bboxesn[j], i1.shape)
        
        for kk in range(j+1, len(bboxesn)):
            print(kk)     
            i2 = imms[kk]
            bb2 =  resort_bbs(bboxesn[kk],i2.shape)
           
            ## n non riferisce
            if (len(bb2) > 0) :
                 
                  frame_hist = np.zeros((np.int(np.max([len(bb1), len(bb2)])),len_edges*3-9+4))
                  frame_hist1 = gen_hist(bb1, i1, frame_hist, edges, len_edges,1)
                  frame_hist2 = gen_hist(bb2, i2, frame_hist,  edges, len_edges,2)
                       # tt, idu = which_faked(bb2)
                 
                  V = compare_frames(frame_hist1, frame_hist2,kk)
                 
                  V1 = make_square(V)
                  # print(V1)
                
                #bipartite
                  permx, vx =  bipartite_matching(V1)
                  vals = V1*permx   
                  idx_objs[kk, idx_objs[kk,:] >=0] = -1
                  index_idx, index_val, cats =  associate(result, schemaSH, kk, idx_objs)
                  for i, v in enumerate(vals):
                         
                        if (sum(v) > 0)  and any(v>0.5):
                            print(i, v)
                            ## i e' l' indice delle categorie di kk -1 e x di dove si muovono
                            idx_map_to, = np.where(v > 0.5)
                            if i < len(index_idx):
                                idx_map_from =  index_idx[i]  #index_curr[i][1]
                                val_map_from =  index_val[i]  #index_curr[i][0] 
                                cat_from     =  cats[i]
                                idx_objs[kk,idx_map_to] = val_map_from
                                result[kk, idx_map_to]  =   cat_from
                            else:
                                cat_from, map_from, cats0 =  associate(result, schemaSH, 0, idx_objs,i)
                                idx_objs[kk,idx_map_to] = map_from
                                result[kk, idx_map_to]  =   cat_from
                           
                  curr_cat =  schemaSH[kk,schemaSH[kk,:] != -1]               
                  i1 = i2.copy()
                  bb1 = bb2.copy()
                  j = j+1 
        return idx_objs, result   


    
    
def count_catg(predC, max_preds):
    schema = np.ones((len(predC)+1, max_preds), dtype = np.int)*-1
    schema[0,:] = np.arange(0,max_preds,1)
    for j, preds in enumerate(predC):
        schema[j+1,0:len(preds)] = preds
    L = len(predC[0])
    init = max_preds - L
    ## to add in first line 
    schemaSH =schema[1:,:]
    fr =  predC[0]
    #to_add =  
    # for i, x in enumerate(schemaSH):
    #      c = list((Counter(x) - Counter(fr)).elements())
    #      if (-1) not in c:
    #         schema[1,-len(c):] = np.array(c)
    result = np.zeros_like(schemaSH)
    result[0,:] = schemaSH[0,:]
    L = list(schemaSH[0])
    assoc = schema[:2]
    
    return schema, assoc, schemaSH, result
    
# predC =pred_categories_all
 # bboxesn, schema, schemaSH, assoc, result, imms, idx_objs, edges, len_edges)
def main_search(bboxes, predC, imms, idx_objs, max_preds, edges, len_edges, video_X):
       
        schema, assoc, schemaSH, result = count_catg(predC, max_preds)
        idx_obs, result = subcompare_frames(bboxes, schema, schemaSH, assoc, result, imms, idx_objs, edges, len_edges)
                  
        return idx_objs, result