# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 21:14:49 2021

@author: fiora
"""

import numpy as np
from functools import reduce
from graph_tracking.compare_masks_bbs import *
from graph_tracking.UtilsCats import *

from itertools import groupby
# from bridge_gap import bridge
# from appoStructure import groundtruth_MATS, testing
 
""" Infer Prediction """

exists_unique = lambda x, y: True if ( (not all(y)) and (np.count_nonzero(y)== 1) and (any(y> x) ) ) else False   
def initilize_graph_preds(treesClass_names, tree_transitions, instances, 
                          max_preds, schemaSH, treesB,  classX):
   
    
    """ salvare la probabilita in memoria"""
    T0 = treesB[classX] 
    rr = range(0,len(T0)-1)
    c = treesClass_names[classX]
    ttx, tty = np.where(schemaSH == c)
    schemaC = np.ones_like(schemaSH)*-1
    schemaC[ttx, tty] = c
    
    Tt = tree_transitions[classX]

    size_tb = np.max([len(x) for x in T0])
    idx_objs = make_idx_objs(instances,Tt, T0, max_preds, schemaC )
    """ salvare la probabilita in memoria"""
    idx_objs_probs =gen_mem_prob(instances, Tt, T0, schemaC)
   
    """   """
    all_nodes = np.unique([x for y in T0 for x in y])
    list_to_add = np.arange(np.max(all_nodes)+1, np.max(all_nodes)+20,1)
    all_nodes = np.insert(all_nodes,len(all_nodes),list_to_add)
    visited  =  list(filter(lambda i: i >=0 , idx_objs[0,:])) 
    tovisit = deque()
    nv = [x for x in all_nodes if x not in visited ]
    tovisit.extend(nv)
    return T0, visited, tovisit, idx_objs
##

### Call prediction_with_mem from main search

def prediction_with_mem(treesClass_names, instances, max_preds, 
                        schemaSH, classX, tree_transitions, treesB, distrp):
    T0, visited, tovisit, idx_objs = initilize_graph_preds(treesClass_names,
                                                      tree_transitions,
                                                      instances,  max_preds, 
                                                      schemaSH, treesB, classX)
    # print('visited', visited)
    
    Tt = tree_transitions[classX]
   
    # print(len(schemaSH)-1, 'Tt=', len(Tt))
    # for i, (H, V) in enumerate(zip(Tt, Vx1)):
    for ii in range(len(schemaSH)-1):
        if ii <= len(Tt)-1:
            V = Tt[ii]
            if ii == 0:
                idx_objs[0, visited]= visited
            previous = T0[ii-1]
            fromX = T0[ii]  
            toX = T0[ii+1]
            Vx = np.zeros_like(V)
            for k in range(len(V)):
                if exists_unique(0.2, V[k,:]):
                    Vx[k,:] = V[k,:]
                else:
                    index = np.argmax(V[k,:])
                    Vx[k,index] = V[k,index]
                
            fromc, toc = np.where(Vx > 0.2)
            fromc = fromc.tolist()
            toc = toc.tolist()
            for x, y in zip(fromc, toc):
                  U = idx_objs[ii, x]
                  ## Expand
                  if U == -1: #and not bridge(i,x,y, idx_objs[0:i+1,:]):
                      new_node = tovisit.popleft() 
                      idx_objs[ii,x] = new_node
                      idx_objs[ii+1,y] = new_node
                  else:
                      new_node = U
                      idx_objs[ii+1,y] = new_node
        # else:
        #     while ii <= len(schemaSH)-1:
        #           idx_objs[ii-1,:]   
        #           idx_objs[ii,:] = idx_objs[ii-1,:]    
        #           ii = ii+1
    return idx_objs

""" BRIFGE THE GAP BETWEEN nodes checking similarity """
#### Bridge Gaps
#
def update_row(delta, row, col, v_curr, v_next ):
    found = false
    k = 1
    row_curr = row
    prevc = row_curr - k
    v = idx_objsc[prevc,:]
    v_curr = idx_objsc[row_curr,:]
    v_next = idx_objsc[row_curr:, :]
    delta_up = delta
    while (delta_up <= 5) and (not found):
        v_prev = idx_objsc[prevc,:]
        condition = ((v_prev != 0) and (v_prev in v) and (v_prev not in v_curr) 
                      and (v_prev not in v_next)) 
        if condition:
            found = True
            return v_prev, prevc, col
        else:
           k = k+1
           prevc = row_curr - k
           v = idx_objsc[prevc,:]
           delta_up = delta_up + 1
#
def check_prev(idx_objsc, row, col, delta):
    ### by induction on row
    past_limit = 6    
    k = 1
    row_curr = row
    prevc = row_curr - k
    v = idx_objsc[prevc,:]
    v_curr = idx_objsc[row_curr,:]
    v_next = idx_objsc[row_curr:, :]
    delta_up = delta
    found = False
    """ step 1 go up following the column"""
    while (delta_up <= past_limit) and (not found):
        v_prev = idx_objsc[prevc,col]
        condition = ((v_prev != 0) and (v_prev in v) and (v_prev not in v_curr) 
                      and (v_prev not in v_next)) 
        if condition:
            found = True
            return v_prev, prevc, col
        else:
           k = k+1
           prevc = row_curr - k
           v = idx_objsc[prevc,:]
           delta_up = delta_up + 1
    if not found:
        print('choose something in the previous row and check', row, col)
        # update_row(delta, row, col, v_curr, v_next )
    return -1,-1,-1, 

def update_memory(col, prev_i, next_i, idx_objsc ):
      fromc = idx_objsc[prev_i, col]
      toc = idx_objsc[next_i, col]
      tx, ty = np.where(idx_objsc == toc)
      txx = tx >= next_i
      idx_objsc[tx[txx],ty[txx]] = fromc  
      return idx_objsc
  
    
def evaluate_gap(g, col, idx_objsc, Tp, distrp, c, cval, ns):
    past_lim = 6    
    # col = g[0]
    tg= g[1]
      
    for i, tp in enumerate(tg):
        row = tp[0]
        if tp[1] < idx_objsc.shape[0]:
           delta = tp[1] -tp[0]
           val = idx_objsc[tp[1], col]
           prev_line = idx_objsc[row, :]
           condition = ((delta <= past_lim) and (row != 0 ) and 
                   ( tp[1] != idx_objsc.shape[0]) and idx_objsc[row-1,col] != idx_objsc[tp[1],col ] and
                       (val not in prev_line))
           if condition:

              probs = np.zeros((np.int(delta/2),1))
              row = tp[0]
              next_i_v = idx_objsc[tp[1], col]
              next_i = tp[1]
              
              v_prev, prev_i, fromC = check_prev(idx_objsc, row, col, delta)
              if prev_i != -1:
                  lev1, centr1, vel1 = Tp[prev_i][1]
                  bb1, mms1, cf1,  i1 = Tp[prev_i][2]
                  tt = lev1.index(fromC)
                  centr1, bb1, mms1, lev1= [centr1[tt]], [ bb1[tt]], [mms1[tt]], [lev1[tt]]
                  bb2, mms2, cf2,  i2 = Tp[next_i][2]
                  lev2, centr2, vel = Tp[next_i][1]
                  ti = lev2.index(col)
                  centr2, bb2, mms2, lev2 = [centr2[ti]], [ bb2[ti]], [mms2[ti]], [lev2[ti]]
                  vel1 =  [[np.linalg.norm(x-y) for y in centr2] for x in centr1]
                # vvs, vvh,
                  Ux = 0
                  Ux,a,b = get_B_matrix(bb1,bb2,  mms1, mms2, i1, i2, 
                          distrp,  vel1, cval, ns, edges, len_edges, Ux, lev1, lev2) #vvs, vvh,
                  probs = Ux[lev1[0],lev2[0]]
                  if probs > 0.3:
                      idx_objsc = update_memory(col, prev_i, next_i, idx_objsc )              
    return idx_objsc
##


##
def find_gaps(ijs):
    """ a gap occurs if:
        there is a change in label on a column and in between the change
        of values there are zeros( or -1 if we did not update idx_objs to idx_objs+1)
        example: cols =[0,1,1,0,0,3,0,0]
        gap = [(0,1),(3,5),(6,7)]
    """
    gaps = []
    
    for i, cols in enumerate(zip(*ijs)):
        cols = np.array(cols, dtype = np.int)
        t1, = np.where(cols > 0)
        if len(t1) > 0:
          iszero = np.concatenate(([0], np.equal(cols, 0).view(np.int8), [0]))
          absdiff = np.abs(np.diff(iszero))
          gap = np.where(absdiff == 1)[0].reshape(-1, 2)
          gaps.append([i,gap])
    return gaps    

### call bridge from main search        
def bridge(idx_objsc, tree_probs, distrp, c, cval, ns):
      
    """ the gap is not filled but if a similarity is found the possible
        label change is solved
    """
    Tp = tree_probs[c]
    gapx = find_gaps(idx_objsc)

    for  g in gapx:
         col = g[0]  
         if len(g[1]) > 0:
              
             idx_objsc = evaluate_gap(g, col, idx_objsc, Tp, distrp, c, cval, ns)
 
    return idx_objsc


""" PATH FINDER """
#### Compute the final set of paths  for the current video

unpack = lambda x: x[0]

def renameL(leveli, levelj, cmm):
       u = max(leveli)
       w = np.min(cmm)
       k = levelj.index(w)
       # s,h = u - w if u != w else w
       U = [x+(u-w)  if w != u else x+w for x in levelj[k:]]
       return list(np.ravel(U))

def common_el(L,H):
    C =[]
    common = [x for x in H if x in L]
    C.append(common)
    return  C

def get_all_names(out_classes,cats):
    names = []
    names.append([ np.unique(
                         list(filter(lambda i: i >=0 , 
                          np.ravel(out_classes[c])))).tolist() 
                          for c in range(len(cats))]
                           )
    names = unpack(names)
    return names

def renaming_paths(out_classes, cats):
    new_pred_mats = [[] for _ in cats]
    names =[]
    for c in range(len(cats)):
        names.append(get_all_names(out_classes))
    new_list = []
    new_list.append(names[0].copy())
    for i in range(1, len(names)):
       x = common_el(names[i], names[i-1])
       if  any(x):
           new_list.append(renameL(names[i-1], names[i], x))
       else: new_list.append(names[i]) 
    
    for c in range(len(cats)):
        pred_matC = out_classes[c]
        pred_mat_vals = names[c]
        if not (list(set(pred_mat_vals) & set(new_list[c]))):
           L = new_list[c]
           for k, ( x,y ) in enumerate(zip(pred_mat_vals, L)):
               # print(k, x, y, L)
               ttx,tty  = np.where(pred_matC == x)
               pred_matC[ttx,tty]  = y
           new_pred_mats[c].append(pred_matC)     
        else:
           new_pred_mats[c].append(pred_matC)  
           
    return new_pred_mats, new_list, names


def searchG( ttx, tty,  embeddingc, categ, idx_subj):
    attrib =[]
    keys = ['time', 'subj', 'categ', 'bb','conf','rle_mask',]
    for j, (timex, y) in enumerate(zip(ttx,tty)):
         # Tb, Tc, Tm = True, True, True
         embeddingj = embeddingc[j]
         (b,c,m) = embeddingj['bb'], embeddingj['conf'], embeddingj['rle_mask']
         if len(b) > 0 and any(b[k][0] == y for k in range(len(b))):
             bx = [x for x in b if x[0] == y ]
             bx = list(zip(*bx))
             bx = bx[1]
         else: 
             bx =['None']
             # Tb = False
         if len(c) > 0 and any(c[k][0] == y for k in range(len(c))):
             cx = [x for x in c if x[0] == y ]
             cx = list(zip(*cx))
             cx = cx[1]
         else: 
             cx = ['None']
             # Tc = False
         if len(m) > 0 and any(m[k][0] == y for k in range(len(m))):
             mx = [x for x in m if x[0] == y ]
             mx = list(zip(*mx))
             mx = mx[1]
         else: 
             mx = ['None']
         A = [timex, idx_subj, categ, bx, cx, mx]
         all_S = dict(zip(keys, A))
         attrib.append(all_S)
    return attrib     
              

   
def compute_paths(embeddings, video_id, cats, out_classes, schemaSH):
    # new_pred_mats, subjects_id, names = renaming_paths(out_classes, cats)
    idx_subj = 1
    allDictionaries = []
    S = []
    all_id_names =  get_all_names(out_classes,cats)
    for i, categ  in enumerate(cats):
        embeddingc = embeddings[i]
        pred_mat =  out_classes[i] 
        prev_idx = all_id_names[i]
        for  k, sub in enumerate(prev_idx):
                # print(k, sub)
                ttx, tty = np.where(pred_mat == sub)
                attribs = searchG(ttx, tty,  embeddingc, categ,  idx_subj)
                dx = {'video_id': video_id, 'attrib': attribs}
                # dx = {'video_id': video_id, 'subj':idx_subj,'category':c, 'attrib': attribs}
                idx_subj = idx_subj +1   
                allDictionaries.append(dx)
                S.append(len(tty))
    return allDictionaries, out_classes, schemaSH, cats, idx_subj, S
            