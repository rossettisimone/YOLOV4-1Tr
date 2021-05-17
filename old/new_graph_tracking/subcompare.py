# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:45:53 2021

@author: fiora
"""
import copy
from new_graph_tracking.unsup_tracking_VIS_new import *
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
         
def addVals(trace, idx, bb2, i2, bboxesn, newL, kk):
    trace.append(idx) 
    if  type(bb2) == list:    
         bbL =  bb2[idx]     
    else:             
      bbL = bb2[idx].astype(np.float)
    bb1 = [bbL]
    i1 = i2.copy()
    if len(bbL)>0:
       newL = make_new_lists(bbL, bboxesn, newL, kk)      
    return idx, bb1, i1, bboxesn, newL, trace 

def make_new_lists(bbL, bboxesn, newL, kk):
    if type(bbL) != list:
       bbLi = np.concatenate([bbL[0: 2], bbL[2:4] - bbL[0:2]] )
    else: bbLi = bbL
    if list(bbLi) in bboxesn[kk]:
       # A = [x for x in bboxesn[kk-1] if x!= list(bbLi) else [-1,-1,-1,-1]]]
       A = [[-1,-1, 0, 0] if x == list(bbLi)  else x for x in bboxesn[kk] ]
       newL.append(A)
       return newL

#bi = Bi.copy(), bboxesn =newList.copy()
#bi = Bir.copy(), bboxesn =newList.copy(), j =ll
def subcompare_frames(n, j, bi, bboxesn, imms, edges, len_edges):
              i1 = imms[j]
              bb1 =  resort_bbs(bi, i1.shape)
              trace =[n]
              probs =[1] 
              bbL = [bb1]
              newL = []
              newL.append(np.array(bboxesn[0]).tolist())
              
              # newL.append(np.array(bboxesn[0])[1:].tolist())
              for kk in range(j+1,len(bboxesn)):
                  # print('kk =', kk)
                  i2 = imms[kk]

                  bb2 =  resort_bbs(bboxesn[kk],i2.shape)
                  if (len(bb2)> 0) and (np.sum(bb2)>= 0):
                      frame_hist = np.zeros((np.int(np.max([len(bb1), len(bb2)])),len_edges*3-9+4))
                      frame_hist1 = gen_hist(bb1, i1, frame_hist, edges, len_edges,1)
                      frame_hist2 = gen_hist(bb2, i2, frame_hist,  edges, len_edges,2)
                      tt, idu = which_faked(bb2)
                      V = compare_frames(frame_hist1, frame_hist2,kk)
                      # print(V)
                      if len(tt) >= 0:
                              Vi = V[0]
                              Vi[idu] = 0
                              V = np.array(Vi)
                              # print('case faked')
                              # print(V)
                      idx    = np.argmax(V)  
                      probs.append([kk, idx, np.max(V)])
                      if np.max(V) >= 0.7:
                          idx, bb1, i1, bboxesn, newL, trace = addVals(trace, idx, bb2, i2, bboxesn, newL, kk)
                          
                          
                      elif np.max(V) < 0.7 and np.max(V) > 0.5:
                          ### resort to bounding boxes
                          # print('resort to DIstANCE list=',n, 'kk =', kk)
                          if resort_to_BBs(bb1,bb2, idx):
                              idx, bb1, i1, bboxesn, newL, trace = addVals(trace, idx, bb2, i2, bboxesn, newL, kk)
                          else:
                             trace.append(-1)
                             newL.append(bboxesn[kk])
                             
                      else:
                             trace.append(-1)
                             newL.append(bboxesn[kk])
                  else:  
                      newL.append([])
                      trace.append(-1)
                      probs.append([kk, -1, -1 ])
                      
              return trace, newL , probs                 


def search_X_extras(newList, idx_objs, INSTp, visited, max_preds, imms, edges, len_edges):
        newListR =  copy.copy(newList) 
        idx_objs_new = copy.copy(idx_objs)  
        tt,  = np.where(INSTp > visited)
        for i, ll in enumerate(tt):
             print('row ll', ll)
             start = ll
             if start < len(newListR) - 1:
                 for n in range(visited, len(newListR[ll])):
                    if n <  len(newListR[ll]): 
                       print(ll, n)
                       Bir = [newListR[ll][n]]
                       if np.sum(Bir)>0 and len(Bir) > 0:
                            tracer, newListA, probs = subcompare_frames(n, ll, Bir, newListR, imms, edges, len_edges)
                            newListR[ll:] = newListA
                            
                            tracer = np.array(tracer)
                            tostack =np.ones((start-1))*-1
                            tracerx = np.concatenate([[n],tostack,tracer])
                            
                            idx_objs_new[:,n] = tracerx
        K = np.sum(idx_objs_new,axis =0) 
        j =np.arange(0,max_preds,1)          
        tt =np.where(K== j)         
        idx_objs_new[1:,tt] = -1  
        return idx_objs_new, newListR                   
    
    
def main_search(bboxes, imms, idx_objs, max_preds, edges, len_edges, video_X):
        newList = copy.copy(bboxes)
        for n in range(0,len(newList[0])):
            Bi = [newList[0][n]]
            if len(Bi) > 0:
                j = 0
                trace, newList, probs = subcompare_frames(n, j, Bi,  newList, imms, edges, len_edges)
                idx_objs[:,n] = trace
        ## begin search
        
        visited =  len(newList[0])
        # to_search =  [list(filter(lambda x: len(x) > visited, L)) for L in newList]
        INSTp = np.array( [X['pred_instances'] for X in video_X])
        if visited < max_preds:  
             print(' searching lost bbs in video',video_X[0]['annotation_id'] )
             idx_objs_new, newListR = search_X_extras(newList, idx_objs, INSTp, visited, max_preds, imms, edges, len_edges)
             return idx_objs_new
        else:
            return idx_objs