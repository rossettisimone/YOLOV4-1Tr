# -*- coding: utf-8 -*-
"""
Created on Sun May 23 22:03:54 2021

@author: Utente
"""

#import copy
import tensorflow_probability as tfp
tfd = tfp.distributions
import itertools
import math
from collections import Counter
from operator import itemgetter
from graph_tracking.bipartiteG import bipartite_matching
import random
import string
from functools import reduce
from graph_tracking.compare_masks_bbs import *
from graph_tracking.UtilsCats import *
# from bridge_gap import bridge
from graph_tracking.infer_prediction import prediction_with_mem, bridge, compute_paths 
# from appoStructure import groundtruth_MATS, testing
from graph_tracking.structureXtest import groundtruth_MATS, testing

""" Initialization """
edges = np.linspace(0,255,cfg.LEN_EDGES)
len_edges = cfg.LEN_EDGES  
is_same = lambda x,y: set(x) == set(y)
is_in = lambda x,y: any(set(x) & set(y))
is_subset_eq = lambda x,y: (set(x).issubset(set(y)))
is_subset = lambda x,y: (set(x).issubset(set(y))) and not((set(y).issubset(set(x))))
is_different = lambda x, y: not(is_same(x,y))
difference = lambda x,y: list(set(x).difference(y))
flatten = lambda l: [item for sublist in l for item in sublist]
exists_unique = lambda x, y: True if ( (not all(y)) and (np.count_nonzero(y)== 1) and (any(y> x) ) ) else False   
# ## how many classes in all videos? at most 4
# ### Structures

def Initializing_video_parms(categories_all, mask_ps, instances, confidence_all):
    predC = categories_all.copy()
    masks = get_all_masks(mask_ps)
    mask_rle = [list(x) for x in mask_ps]
    max_preds = max(instances) 
    confidence = confidence_all.copy()
    ### Assuming that we have a chaess_board in which the number of cells in a row
    ### corresponds to the number of subjects/itemms unknown at step T0
    ### schema SH provides the distribution of each class on the chess_board
    schemaSH = count_catg(predC, max_preds)
    ### Categories
    cats = np.unique(np.ravel(schemaSH))
    cats = [x for x in cats if x>0]

    f  = np.arange(0,len(schemaSH[0,:]),1)
    ## position in schemaH of each bb
    loc_ids = np.matlib.repmat(f, len(schemaSH), 1)
    # Name_Ids = [id_generator() for k in Loc_Ids[0,:]]
    return loc_ids, cats, max_preds, masks, confidence, schemaSH, mask_rle


""" Local functions """
 

## make the tree

def tree_of_classes(cats, loc_ids, schemaSH):
    treesClass_names = [[c]  for c in cats]

    ## position of bounding boxes in schemaSH
    treesB =  [[] for _ in range(len(cats))]   
    for i,L in enumerate(treesB):
        L = treesB[i]
        c = cats[i]
        for ti, row in enumerate(schemaSH):
            tx = [h for h in loc_ids[ti,:]  if row[h] == c]
            L.append(tx)
    return treesClass_names, treesB


### ## build states and distributions
def make_embedding(bb1, conf1, rl1, ll):
        bb_list = [(j, bb) for (j,bb) in zip(ll,bb1)]
        conf_list = [(j,cc) for (j,cc) in zip(ll,conf1)]
        rl_list = [(j,mm) for (j,mm)  in zip(ll, rl1)]
        keys = ['bb', 'conf', 'rle_mask' ]
        values = [bb_list, conf_list, rl_list]
        return dict(zip(keys, values))

def graph_probs(cats, treesB, bboxes, masks, confs, mask_rle, imms, gt):
    tree_probs = [[] for _ in range(len(cats))]
    prob_samples = [[] for _ in range(len(cats))]
    embeddings = [ [] for _ in range(len(cats))]
    # samples_acc =  [[] for _ in range(len(cats))]
    # prob_samples_dir = [[] for _ in range(len(cats))]
    for i, (L,TP) in enumerate(zip(treesB, tree_probs)):
        TP = tree_probs[i]
        L = treesB[i]
        trans_mot = prob_samples[i]
        embeddingI = embeddings[i]
        # acc = samples_acc[i]
        # trans_angle =prob_samples_dir[i]
        for  ti in range(len(L)-1):   #, ll in enumerate(zip(range(0,len(L)-1),L)):
                   
                    ## lev1 e lev2 sono le posizioni delle bb al tempo t e al tempo t+1
                    # print('probability graph: branch', i , 'level', ti)
                    bb1, bb2, lev1, lev2, centr1, centr2, vel = get_bbs( bboxes, L, ti)
                    if gt:
                       mms1, mms2, conf1, conf2, lev1, lev2  = get_masks( masks, confs, L, ti)
                    else:
                       mms1, mms2, conf1, conf2, lev1, lev2  = get_masks_pred( masks, confs, L, ti)
                    
                    rl1, lev1 = get_rle(mask_rle,L,ti)
                    embeddingI.append(make_embedding(bb1, conf1,  rl1, L[ti]))
                    i1 = imms[ti]
                    i2 = imms[ti+1]
                    
                    trans_mot.append(np.ravel(np.array([x for x in vel])))
                    # acc.append(np.ravel(np.array([x for y in iou for x in y])))
                    # trans_angle.append(np.ravel(np.array([x for x in theta] )))
                    state1 = [lev1, centr1, vel]
                    state_vals1 = [bb1, mms1, conf1, i1]
                    state2 = [lev2, centr2,  0]
                    state_vals2 = [bb2, mms2, conf2, i2]
                    TP.append([ti, state1, state_vals1,  ti+1, state2, state_vals2, rl1])
    return tree_probs, embeddings,  prob_samples


### Build transitions
## 
def graph_transitions(tree_probs, treesClass_names, prob_samples, cats, ss):
    if len(prob_samples)> 5:
       distrpV = make_distribution(prob_samples, cats)
    else: 
       prob_samples = [[] for _ in range(len(cats))]
       prob_samples = [[np.random.uniform(0.0,200.0,1) for j in range(10)] for k in range(len(cats))]
       distrpV = make_distribution(prob_samples, cats)
    
    # distrpA = make_distribution(acc, cats)
    # distrpD = make_distribution(prob_samples_vel, cats)
    tree_transitions = [[] for _ in range(len(cats))]
    U1V1 = [[] for _ in range(len(cats))]
    VX = [[] for _ in range(len(cats))]
    for i, (TP, Ttrans, Vx) in enumerate(zip(tree_probs, tree_transitions, VX)): #len(tree_probs) -1):
        Tp = tree_probs[i]
        Ttrans = tree_transitions[i]
        Vx = VX[i]
        uv = U1V1[i]
        cval =  treesClass_names[i]
        ns =  len(prob_samples[i])
        for  ti in range(len(Tp)-1):
             ## lev1 e lev2 sono le posizioni delle bb al tempo t e al tempo t+1
             # print('transition graph: branch', i, 'level', ti)
             lev1, centr1, vel1 = Tp[ti][1]
             bb1, mms1, cf1, i1 = Tp[ti][2]
             lev2, centr2, vel2 = Tp[ti+1][1]
             bb2, mms2, cf2, i2 = Tp[ti+1][2]
             # (bb1,bb2, mms1, mms2, i1, i2, distrpV,  vel, cval, ns,  edges, len_edges, VX, lev1, lev2)
             Vx, u1, v1 = get_B_matrix(bb1,bb2,  mms1, mms2, i1, i2, 
                               distrpV,  vel1,  cval, ns, edges, len_edges, Vx, lev1, lev2) 
             Ttrans.append(Vx)
             uv.append([u1, v1])
    return distrpV, tree_transitions, VX, U1V1    
         

###########################################################################################
def make_distribution(Prob, cats):
   P = []
   for c in range(len(cats)):
       probsc =   np.concatenate( Prob[c], axis=0 )
       P.append(probsc)
   probs = np.concatenate(P, axis=0 )
   distrp = tfp.distributions.Empirical(samples = probs)
   return distrp



# 1-distrp.cdf(200)
# 1-distrp.cdf(10)
### Prediction

### MAIN build all graphs
def main_search_and_prediction(categories_all, bboxes, imms,  mask_ps, instances, confidence_all, video_id, gt = False):
    ### build memories and all  the graphs
   if any(bboxes):
       loc_ids, cats, max_preds, masks, confs, schemaSH, mask_rle =  Initializing_video_parms(categories_all, 
                                                                             mask_ps, instances, confidence_all)
       treesClass_names, treesB = tree_of_classes(cats, loc_ids, schemaSH)
       tree_probs, embeddings, prob_samples = graph_probs(cats, treesB, bboxes, masks, confs, mask_rle, imms, gt = False)
       distrp,  tree_transitions, VX,  U1V1 = graph_transitions(tree_probs,treesClass_names,  
                                                                   prob_samples, cats, imms[0].shape)
    
       # Main prediction 
    
       out_classes = [[] for _ in range(len(cats))]
       for c in range(len(cats)):
             idx_objs = prediction_with_mem(treesClass_names,instances, max_preds,
                                      schemaSH, c, tree_transitions, treesB, distrp)
        
             ### bridge the gaps
             idx_objsc = (idx_objs + 1).copy()
             X = bridge(idx_objsc, tree_probs, distrp, c, treesClass_names[c], len(prob_samples[c]))
             ### save for testing
             out_classes[c] = X-1
       
       # build path for each subject
       #(embeddings, video_id, cats, out_classes, schemaSH)
       dictionaryV, out_classes, schemaSH, cats, IS, S =  compute_paths(embeddings, video_id,  cats, out_classes, schemaSH)
       print('number of subjects', IS-1)
       print('path_length', S)
       return dictionaryV, out_classes, schemaSH, cats
   return [], [],[],[]
   ## Testing
    ### generate the ground truth matrices, one for class
    
def  test_accuracy(schemaSH, out_classes, ids_gt, cats):
   GT_mats = groundtruth_MATS(schemaSH, ids_gt, cats)
   cost = np.zeros((len(cats)))
   
   for c in range(len(cats)):
       pred_mat = out_classes[c]
       gt_mat = GT_mats[c]
       Y, mp, ngt = testing(gt_mat, pred_mat, c)
       Y = Y/mp
       cost[c] =np.sum(Y)
       
   cost = np.sum(cost)/len(cats)
   print('cost')
   acc = 1-cost
   return acc
   
   