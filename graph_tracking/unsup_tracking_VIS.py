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
from skimage.transform import resize
from skimage.feature import daisy
from skimage.color import rgb2gray
# from bipartiteG import bipartite_matching
import graph_tracking.config_VIS as cfg

""" parameters --> config """

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
    # plt.suptitle('Correspondence between I(j) and I(j+1).\n In yellow bbs without correspondences ')
    # """ If the bounding boxes are given in image coordinates 
    #     then comment the following part
    #     which is designed for normalized bounding boxes coordinates
    #     and rename
    #     bbs1 = ub1
    #     bbs2 = ub2
    # """
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
    with open('..\..{}'.format(json_id)) as json_file:
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
    for tt, bb in enumerate(bbs):
        """ Uncomment to control the bb correspondence"""
        # show_bb_before(image,[bb], jj, ax)
        bb_int = np.array([int(x) for x in bb])
        bbx = bb_int.copy()
        # features = np.zeros((bb_int[2]- bb_int[0], bb_int[3]-bb_int[1]))
        if bbx[2]>= 0:
           bbx[2] = np.min([bb_int[0] + bb_int[2], shape[1]])
        else:
           bbx[2] = np.max([bb_int[0] + bb_int[2], 0])
        if bbx[3]>= 0:
           bbx[3] = np.min([bb_int[1] + bb_int[3], shape[0]])
        else:
           bbx[3] = np.max([bb_int[1] + bb_int[3], 0]) 
        B.append(bbx)
    return B

def gen_randomF(n1,n2,n3, edges, len_edges):
        features = np.random.rand(n1,n2,n3)
        hh = np.zeros((len_edges-3, 3))
        ffx = resize(features, (n1,n2), clip=True,  anti_aliasing= True)
        ff0 = np.ravel(ffx/np.max(ffx))
        frame_histx = np.resize(ff0,(2,355 ))
        ll = np.random.randint(4**6,(4**7)/2)
        ff1 =  np.random.rand(ll)    
        return ff0, ff1
        
def  gen_hist(bbs, image, frame_hist, edges, len_edges, k): # 
    
    frame_histx = frame_hist.copy()
    rem_imm = np.zeros((len(frame_hist),cfg.LEN_IMM+cfg.FF+frame_histx.shape[1]))
    """ Uncomment to control the bb correspondence"""
    # fig, ax = plt.subplots()
    # ax.imshow(image)
    for tt, bb in enumerate(bbs):

        """ Uncomment to control the bb correspondence"""
        # show_bb_before(image,[bb], jj, ax)
        bb_int = np.array([int(x) for x in bb])
        bbx = bb_int.copy()
        bbx[ bbx <0] = 0
       
        features = image[bbx[1]:bbx[3], bbx[0]:bbx[2], :].astype(np.float64)
        if (features.shape[0]<7) and (features.shape[1]<7):
            # print('faked')
            ff0, ff1 = gen_randomF(32,32,3, edges, len_edges)
            
        else:
            ffx = resize(features, (32,32), clip=True,  anti_aliasing= True)
            ff0 = np.ravel(ffx/np.max(ffx))
            ffxx = np.column_stack([ffx[:,:,0],ffx[:,:,1],ffx[:,:,2]])
            hh = np.zeros((len_edges-3, 3))
            ffx = resize(features, (32,32), clip=True,  anti_aliasing= True)
            ff0 = np.ravel(ffx/np.max(ffx))
            ffxx = np.column_stack([ffx[:,:,0],ffx[:,:,1],ffx[:,:,2]])
            ff1 = np.ravel(daisy(ffxx, step=3, radius=5, rings=2, histograms=4, orientations=4, normalization='l1'))
            hh2 = color_histogram(hh, features[:,:,0],features[:,:,1],features[:,:,2], edges, len_edges)
            frame_histx[tt,0:len(hh2)] = hh2 
            frame_histx[tt,len(hh2) : len(hh2)+5] = bb/(0.5*len(hh))
            
        rem_imm[tt,0:len(frame_histx[tt,:])] = frame_histx[tt,:]
        rem_imm[tt, len(frame_histx[tt,:]): len(frame_histx[tt,:])+len(ff0)] = ff0
        ll = len(frame_histx[tt,:]) + len(ff0)
        rem_imm[tt, ll: ll+len(ff1)] = ff1
    return rem_imm

def unique_list(bb_list):
    bb_array = np.array(bb_list)
    unique_bb, counts_bb, idx_bb = np.unique(bb_array,  return_counts = True, return_index = True, axis = 0)
    return unique_bb, counts_bb, idx_bb     
   
 

def compare_frames(frame1, frame2):

    m = frame1.shape[0]
    list_corr = np.zeros((m, m))
 
  
    
    for kk in range(m):
        hk = frame1[kk,:]
        ff_frame = matlib.repmat(hk,m,1)
        corri =  np.array(tfp.stats.correlation(ff_frame,frame2, sample_axis=1, event_axis = None))
        if any(np.isnan(corri)):
             corri[np.isnan(corri)]  = 0
        list_corr[kk,:] = corri
    return list_corr     
   
        

            
            
## This only for AVA because the are more than one timestamp per image
def sort_and_unique(path_videoN, video_name, base_dir):
    unique_TS = [int((x.split('\\')[-1]).split('.png')[0]) for x in path_videoN]
    z = np.array(unique_TS)
    unique_frames, idx_start, count_ann = np.unique(z, return_counts = True, return_index = True)
    imm_to_load = [base_dir+'\\'+video_name+'\\'+str(x) + '.png' for x in unique_frames]
    idx_start = idx_start.astype(np.int)
    count_ann = count_ann.astype(np.int)
    return imm_to_load, idx_start, count_ann
 
def get_memories(instances):
    memX ={}    
    submemX = {}
    max_preds = np.max(instances)
    idx_objs = np.zeros((len(instances)+1, max_preds), dtype = np.int)
    idx_objs[0,:] = np.arange(0,max_preds,1)
    for k  in range(max_preds):
        submemX.update({k:{'bb': []}})
    return memX, submemX, idx_objs, max_preds
        
        
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

        
