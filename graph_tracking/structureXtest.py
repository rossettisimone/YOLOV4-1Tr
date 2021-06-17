# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:26:54 2021

@author: Utente
"""
import numpy as np
from graph_tracking.bipartiteG import bipartite_matching

def groundtruth_MATS(schemaSH, ids_gt, cats):
    GT_mats = [[] for _ in range(len(cats))]
    idsS = np.unique([i for j in ids_gt for i in j])
    
    
    for i, c in enumerate(range(len(cats))):
        gt_matrix = np.ones_like(schemaSH)*-1 
        X = np.ones_like(gt_matrix)*-1
        for k, (row, ll) in enumerate(zip(schemaSH,ids_gt)):
            # print(row, ll)
            gt_matrix[k,0:len(ll)] = [x - idsS[0] for x in ll ]  # 
            
        
        for j, row in enumerate(zip(schemaSH, gt_matrix)):  
            for h in range(len(ids_gt[j])):
                if schemaSH[j,h] == cats[c]:
                    X[j,h] = gt_matrix[j,h]
                    
        GT_mats[i] = X
    return GT_mats 
           
        
def find_path(val, matrixX):
    pathx, pathy = np.where(matrixX == val)
    return pathx, pathy
    
    
def testing(gt_mat, pred_mat,c):
    uu = np.unique(np.ravel(gt_mat))
    valuesGT = list(filter(lambda i: i >=0 , uu)) 
    # we find the path of each value
    pathGT =  [[] for _ in range(len(valuesGT))]
    
    for ii, val in enumerate(valuesGT):
        pathxg, pathyg = find_path(val, gt_mat)
        matG = np.column_stack([pathxg,pathyg])
        pathGT[ii].append(matG)
        
        
    vv = np.unique(np.ravel(pred_mat))
    valuesP = list(filter(lambda i: i >=0 , vv)) 
    # we find the path of each value
    pathP =  [[] for _ in range(len(valuesP))]
    for ii, val in enumerate(valuesP):
        pathx, pathy = find_path(val, pred_mat)
        matP = np.column_stack([pathx,pathy])
        pathP[ii].append(matP)
     
    mm = np.int(np.max([len(pathP),len(pathGT)]))
    M = np.zeros((mm,mm))
   
    for i , s in enumerate(pathP):
        for j, t in enumerate(pathGT):
            M[i,j]   = compare_paths(s[0],t[0])
    # here we add the maximum to either a row or a column of all ones indicating
    # that the number of paths are different, so we add the penalty

    if len(M) > 1:
        test_numPathG = np.sum(M,axis =0)
        test_numPathP = np.sum(M,axis =1)
        tg, = np.where( test_numPathG ==0)
        tp,=  np.where( test_numPathP ==0)
        for columns in tg:
            M[:,tg] =1
        for columns in tp:
            M[tp, :] =1
    Mc = np.max(M) - M
    B,u = bipartite_matching(Mc)
    return M*B, len(pathP),  len(pathGT)

def compare_paths(sc,tc):
    """ evaluation of errors on a path:
        score: 
             intersection between the two path
    """
    # sc = np.column_stack([s[0], s[1]])
    # tc = np.column_stack([t[0], t[1]])   
    m, n = len(sc), len(tc)
    cost = 0.0
    delta = np.argwhere((tc[:,None,0] == sc[:,0]) & (tc[:,None,1] == sc[:,1]))
    if len(delta) == 0:
        cost = 1.
    else:
       cost = (n-len(delta))/n
    return cost
    