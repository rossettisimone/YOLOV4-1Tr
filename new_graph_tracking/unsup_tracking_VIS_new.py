# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:22:25 2021

@author: Fiora
"""


""" Utils"""

import os
import numpy as np
from numpy import matlib
import json

from PIL import Image, ImageOps
from matplotlib import pyplot as plt 
import matplotlib.patches as patches
import tensorflow as tf
import tensorflow_probability as tfp
import itertools as itt
import pandas as pd
from skimage.transform import resize, pyramid_gaussian
#from scipy.stats import linregress
#from  scipy.spatial.distance import correlation
#from skimage.feature import from scipy.signal import correlatedaisy
# from scipy.signal  import convolve2d
from skimage.color import rgb2gray
# from skimage.transform import resize 
from new_graph_tracking.bipartiteG import bipartite_matching
import new_graph_tracking.config_VIS as cfg

""" parameters --> config """
len_edges = cfg.LEN_EDGES
edges = np.linspace(0,255,len_edges)
len_imm = cfg.LEN_IMM

def bb_embeds(bb, cards, shape):
    
    for kk in range(cards-len(bb)):
        a, = np.random.randint(0,shape[1],1)
        b, = np.random.randint(0,shape[0],1)
        bbk = np.array([a,b,a+5, b+5])
        bb.append(bbk)
    return bb


def show_bb_after(img1,img2, n1, n2, ub1, ub2, corr): #,v_name):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img1)
    ax[0].set_title('I('+np.str(n1)+')')
    ax[0].axis('off')
    ax[1].imshow(img2)
    ax[1].set_title('I('+np.str(n2)+')')
    ax[1].axis('off')
  
    wdt1 , hgt1 = img1.shape[0], img1.shape[1]
    # T1 = [wdt1, hgt1, wdt1, hgt1]
    # bbs1 = np.array( [x*T1 for x in ub1])
    bbs1 = ub1.copy()
    wdt2 , hgt2 = img2.shape[0], img2.shape[1]
    # T2 = [wdt2, hgt2, wdt2, hgt2]
    # bbs2 = np.array( [x*T2 for x in ub2])
    bbs2 =  ub2.copy()
    """  end comment """
    for tt, bb in enumerate(bbs1):
        
        x0,y0 = bb[0:2]
        ww, ll = bb[2:4]-bb[0:2]
        
        w = corr[tt,1]
        if w  >= 0:
           rect = patches.Rectangle((x0, y0), ww, ll, linewidth=1, edgecolor='r', facecolor='none')
           ax[0].add_patch(rect)
           ax[0].annotate(str(tt)+':'+str(w), (x0,y0), color='r', weight='bold', fontsize=8)
        else:
           rect = patches.Rectangle((x0, y0), ww, ll, linewidth=1, edgecolor='y', facecolor='none')
           ax[0].add_patch(rect)
           ax[0].annotate(str(tt), (x0,y0), color='y', weight='bold', fontsize=8)
    for tt, bb in enumerate(bbs2):    
        x0,y0 = bb[0:2]
        ww, ll = bb[2:4]-bb[0:2]
        if tt in corr[:,1]:
           rect = patches.Rectangle((x0, y0), ww, ll, linewidth=1, edgecolor='r', facecolor='none')
           ax[1].add_patch(rect)
           ax[1].annotate(str(tt), (x0,y0), color='r', weight='bold', fontsize=8)    
        else:
           rect = patches.Rectangle((x0, y0), ww, ll, linewidth=1, edgecolor='y', facecolor='none')
           ax[1].add_patch(rect)
           ax[1].annotate(str(tt), (x0,y0), color='y', weight='bold', fontsize=8)
    
    # plt.savefig(v_name,  dpi=200, pad_inches=0)
    plt.show()
    # plt.pause(1)
    # plt.close('all')
    # print('saving', v_name)

def show_bb_before(image, bbs,  u, ax):
    # fig, ax = plt.subplots()
    # ax.imshow(image)
    for bbx in bbs:
        
        x0,y0 = bbx[0:2]
        ww, ll = bbx[2:4]-bbx[0:2]
        rect = patches.Rectangle((x0, y0), ww, ll, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.annotate(str(u), (x0,y0), color='r', weight='bold', fontsize=16)
    plt.show()

def  load_info(json_id):
    with open('F:\\AVA\\{}'.format(json_id)) as json_file:
         return json.load(json_file) 


# 
def  info_nums(video_dir_train):
   
    vals_imm = [np.unique(vals[1].get('image_timeStamp'), return_counts =True) for vals in video_dir_train.items()]
    counts_imm = [len(x[0])  for x in vals_imm]
    num_ann = [np.max(x[1]) for x in vals_imm]
    return counts_imm, num_ann


def read_image(filepath):
    return Image.open(filepath) 
        

# def extract_features_from_imm(img,bb):
#     image = np.array(img)
#     features = np.zeros((hgt, wdt, 3))
    
def color_histogram(hh, immr, immg, immb, edges, len_edges):
    weights = np.ones_like(hh)/len(hh)
    hh[:,0] = tfp.stats.histogram(immr, edges)[1:-1]
    hh[:,1] = tfp.stats.histogram(immg, edges)[1:-1]
    hh[:,2] = tfp.stats.histogram(immb, edges)[1:-1]
    hhx = hh*weights
    hhx = hhx.flatten()
    return hhx

def split(bb1, ll):
    k, m = divmod(len(bb1),ll)
    return (bb1[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(ll))

def resort_bbs(bbs,shape):

    B =[]
    for t, bb in enumerate(bbs):
        if bb != [-1, -1, 0, 0] :
            bb_int = np.array([x for x in bb])   ### changed
            bbx = bb_int.copy()             
            bbx = np.concatenate([bbx[0: 2], bbx[2:4] + bbx[0:2]])
            B.append(bbx)
        else: B.append(bb)
    return B

def gen_randomF(n1,n2,n3, edges, len_edges):
        features = np.random.rand(32,32,3)
        hh = np.zeros((len_edges-3, 3))
        ffx = resize(features, (32,32), clip=True,  anti_aliasing= True)
        ff0 = np.ravel(ffx/np.max(ffx))
        frame_histx = np.resize(ff0,(2,355 ))
        ll = np.random.randint(4**6,(4**7)/2)
        ff1 =  np.random.rand(ll)    
        return ff0, ff1

def Gkernel(k,ss): 
    ax = np.linspace(-(k - 1) / 2., (k - 1) / 2., k)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(ss))
    return kernel / np.sum(kernel)

def Dkernel():
   return  np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

def im_grad(imm):
    imm = rgb2gray(imm)
    gr = np.gradient(imm)
    return (gr[0]**2+gr[1]**2)**0.5
    

def  gen_hist(bbs, image, frame_hist, edges, len_edges, k): # 
    len_edges = cfg.LEN_EDGES
    frame_histx = frame_hist.copy()
    rem_imm = np.zeros((len(bbs),cfg.LENH +cfg.LEN_IMM +  4) )
   
    """ Uncomment to control the bb correspondence"""
    # fig, ax = plt.subplots()
    # ax.imshow(image)
    for tt, bb in enumerate(bbs):
        
        if  not any(np.array(bb)>0):
           # print('faked')
           features = np.random.rand(7,7,3).astype(np.float64) 
           bb  = np.random.rand(4)*3
           bb_int = np.array([int(x) for x in bb])
           bbx = bb_int.copy()
        else:
            bb_int = np.array([int(x) for x in bb])
            bbx = bb_int.copy()
            bbx[ bbx <0] = 0
            features = image[bbx[1]:bbx[3], bbx[0]:bbx[2], :].astype(np.float64)
        
       
        ffx = resize(features, (32,32), clip=True,  anti_aliasing= True)
        PF1 = tuple(pyramid_gaussian(ffx, max_layer=2, downscale=2, sigma=3, order=1, mode='reflect', cval=0, multichannel=True))
        U = []
        for  hh in PF1:
           U.append(np.ravel(hh))
        F1 =  np.concatenate([U[0], U[1], U[2]]) 
        hh = np.zeros((len_edges-3, 3))
        hhf = color_histogram(hh, features[:,:,0],features[:,:,1],features[:,:,2], edges, len_edges)
        if type(bb) == list:
            np.array(bb)
        hhX = np.concatenate([hhf,F1, bb])
        rem_imm[tt,:] = hhX # 
        
    return rem_imm

def unique_list(bb_list):
    bb_array = np.array(bb_list)
    unique_bb, counts_bb, idx_bb = np.unique(bb_array,  return_counts = True, return_index = True, axis = 0)
    return unique_bb, counts_bb, idx_bb     
   
 
# frame1 = frame_hist1.copy()
# frame2 =frame_hist2.copy()
def compare_frames(frame1, frame2,p):
    
    m = np.max([frame1.shape[0],frame2.shape[0]])
    list_corr = np.zeros((1,m))
    
    ## Here we know frame1 shape[0]=1
   
    ff_frame = matlib.repmat(frame1,m,1)
    
    corri =  np.array(tfp.stats.correlation(ff_frame,frame2, sample_axis=1, event_axis = None))
    EuDist =  (np.sum(np.dot((ff_frame-frame2), (ff_frame-frame2).T))**0.5) 
    # corri = corri/EuDist
    if any(np.isnan(corri)):
              corri[np.isnan(corri)]  = 0
    list_corr[0,:] = corri
    # for kk in range(m):
    #     hk = frame1[kk,:]
    #     ff_frame = matlib.repmat(hk,m,1)
    #     corri =  np.array(tfp.stats.correlation(ff_frame,frame2, sample_axis=1, event_axis = None))
    #     if any(np.isnan(corri)):
    #          corri[np.isnan(corri)]  = 0
    #     list_corr[kk,:] = corri
    return list_corr     

### Case of measuring bounding boxes

def  resort_to_BBs(A,B, idx):
    # dists = []
    
    x = B[idx]
    if (len(x) > 0) and (type(x)!= list):
        dists = (np.sum(np.dot((A[0]-x), (A[0]-x).T))**0.5)/4
    else: dists= 10**3
    if dists < 80:
        return True
    else: return False
    #        dists.append((np.sum(np.dot((A[0]-x), (A[0]-x).T))**0.5)/4)
    #     else: dists.append(10**3)
    # if (len(dists) == 1 ) and (dists[0]< 100):
    #     return True
    # if (len(dists) > 1 ) and (np.min(dists) < 100):
    #     return True
    # else:
    #    return False     
    
     

            
## This only for AVA because the are more than one timestamp per image
def sort_and_unique(path_videoN, video_name, base_dir):
    unique_TS = [int((x.split('\\')[-1]).split('.png')[0]) for x in path_videoN]
    z = np.array(unique_TS)
    unique_frames, idx_start, count_ann = np.unique(z, return_counts = True, return_index = True)
    imm_to_load = [base_dir+'\\'+video_name+'\\'+str(x) + '.png' for x in unique_frames]
    idx_start = idx_start.astype(np.int)
    count_ann = count_ann.astype(np.int)
    return imm_to_load, idx_start, count_ann
 
def get_memories(step, labels):
    print(step, labels)
    submemx ={} 
    subM = {}
    for ii, ll  in enumerate(labels):
             subM.update({str(ii): ll})
    submemx.update({step: subM})
    return submemx, subM
#
def look_back(submemX):
    hist = []
    for label, idxl in submemX.items():
       for kk, ii in idxl.items():
        if ii == 1:
            hist.append([label, kk, ii])
    return hist[-1]    
#    
def recallH(Hh, bbx1, bb2, kk, imms1, i2, edges, len_edges):
    bbxx = bbx1[Hh[0]]
    bb1 = [(bbxx[int(Hh[2])])]
    ix1 = imms1[Hh[0]]
    bb1 =  resort_bbs(bb1,ix1.shape)
    
    bbn2 = [bb2[kk]]
    frame_hist = np.zeros((np.int(np.max([len(bb1), len(bbn2)])),len_edges*3-9+4))
              
    frame_hist1 = gen_hist(bb1, ix1, frame_hist, edges, len_edges,1)
    frame_hist2 = gen_hist(bbn2, i2, frame_hist,  edges, len_edges,2)
              
    """ V returns the similarity"""
    V = compare_frames(frame_hist1, frame_hist2)
    if V > 0.7: return True
    else: return False

##
def lookup(pos, vals):
    ttx, = np.where((vals[pos,:] == np.max(vals[pos,:])) & (np.max(vals[pos,:]) > 0.7)  )
    if len(ttx)> 0:
        return False
    else: return True
#
def make_idx_objs(instances):
    max_preds = np.max(instances)
    idx_objs = np.zeros((len(instances), max_preds), dtype = np.int)
    idx_objs[0,:] = np.arange(0,max_preds,1)
    
    return idx_objs, max_preds
        
        
def set_pairs(imm_to_load_All, idx_start_All, count_ann_All, compare, cw):
    return imm_to_load_All[cw:cw+compare], idx_start_All[cw:cw+ compare], count_ann_All[cw:cw+compare]
   
def create_VideoN_dataframe(imm_to_load_All):
    df = pd.DataFrame(np.arange(1, len(imm_to_load_All)+1))
    df.index = [imm_to_load_All]
    columnsx = cfg.BOUNDING_BOXES 
    for ii in range(len(columnsx)):
        df[columnsx[ii]] =columnsx[ii]
    return df, columnsx
        

def check_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)



