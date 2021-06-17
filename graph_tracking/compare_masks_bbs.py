# -*- coding: utf-8 -*-
"""
Created on Mon May 24 12:13:50 2021

@author: Utente
"""

#1. Comparing Masks

import tensorflow as tf
import numpy as np
import graph_tracking.config_VIS as cfg
from skimage.feature import ORB, match_descriptors, plot_matches, match_template
from skimage.color import  rgb2gray, rgb2lab
from skimage.measure import regionprops
from skimage.util import img_as_ubyte
from scipy import ndimage
import cv2 as cv
from numpy import matlib

from matplotlib import pyplot as plt 
import matplotlib.patches as patches
import tensorflow as tf
import tensorflow_probability as tfp
import itertools as itt
from skimage.transform import resize, pyramid_gaussian


# #### Obtain the masks
# def build_masks(m0):
#         h, w = m0.get('size',[720,1280])
#         rle_arr = m0.get('counts',[0])
#         rle_arr = np.cumsum(rle_arr)
#         indices = []
#         extend = indices.extend
#         list(map(extend, map(lambda s,e: range(s, e), rle_arr[0::2], rle_arr[1::2])));
#         binary_mask = np.zeros(h*w, dtype=np.uint8)
#         binary_mask[indices] = 1     
#         BM = binary_mask.reshape((w, h)).T  
#         return BM
    
    
# def get_all_masks(masks_ps):
#     el_of_masks = list( map(lambda x: map( lambda y: build_masks(y), x) , masks_ps))
#     all_masks = [list(x) for x in el_of_masks]
#     return all_masks

### Comparing Masks and BBS


 #1. Matching features descriptors
def match_descriptors(imma1,imma2):
    if len(imma1) ==0 or len(imma2==0):
        print('empty sets')
    imx1 = img_as_ubyte(rgb2gray(imma1))
    imx1[imx1==0] = 1
    imx2 = img_as_ubyte(rgb2gray(imma2))
    imx2[imx2==0] = 1                    
    descriptor_extractor = ORB(downscale =1.6,n_scales =3, n_keypoints=200,
                               fast_n=9, fast_threshold=0.08, harris_k=0.04)
    empty1 = False 
    empty2 = False
    try:    
           descriptor_extractor.detect_and_extract(imx1)
           keypoints1 = descriptor_extractor.keypoints
           descriptors1 = descriptor_extractor.descriptors
    except (RuntimeError, TypeError, NameError):
           empty1 =True
    
    try:    
           descriptor_extractor.detect_and_extract(imx2)
           keypoints2 = descriptor_extractor.keypoints
           descriptors2 = descriptor_extractor.descriptors
    except (RuntimeError, TypeError, NameError):
           empty2 =True
    
    if empty1 or empty2:
        return [] 
    elif not(len(descriptors1) > 0) and not(len(descriptors2) >0):
          matches12 = match_descriptors(descriptors1, descriptors2, 
                                      metric ='euclidean', max_distance = 30, 
                                      cross_check=True, max_ratio=0.9)
          
          return matches12
    else:  return [] 

##
### match shape
def match_shape( im1, im2,  imm_curr, imm_next,j):
   """ Return values concerning the shape of the masks:
       1. contour matches using poligonal coordinates
       2. centroids distance
       3. distance transform that accounts for shape
          taking the max skeleton
       4. iou area
   """
   """ Metric 1 contour matching"""    
   div_coord = lambda x,z: np.array([x[0]/z[0], x[1]/z[1]]) 
   ix = im1+10
   iy = im2+10
   IOU = (1-np.round((np.sum((im1+1) * (im2+1))/((im1+1+im2+1))))).astype(np.uint8)
  
   ttx, tty = np.where(im1 ==1)
   IOU[ttx,tty] =1
   ttx, tty = np.where(im2 ==1)
   IOU[ttx,tty] = 1
   ttx, tty = np.where(IOU !=1)
   IOU[ttx,tty] = 0
   r1 = regionprops(im1, intensity_image = imm_curr)
   r2 = regionprops(im2, intensity_image = imm_next)
   r3 = regionprops(IOU)
   # bb1 = np.array(r1[0].bbox)
   # bb2 = np.array(r2[0].bbox)
   # bb3 = np.array(r3[0].bbox)
   a1 = r1[0].convex_area
   a2 = r2[0].convex_area
   a3 = r3[0].convex_area
   iou_ratio = 0.5*((a1/a3)+(a2/a3))
   # coord1 = r1[0].coords
   # coord2 = r2[0].coords
   # ret = cv.matchShapes(coord1,coord2,1,0.0)    ### less than 3
   centr1 = r1[0].centroid
   centr2 = r2[0].centroid
   """ metric 2 = centroid distance """
   centroids_distance =  np.linalg.norm(div_coord(centr1,im1.shape)-div_coord(centr2,im2.shape))
   imi1 = r1[0].image
   imi2 = r2[0].image
   imma1I =  r1[0].intensity_image
   imma2I =  r2[0].intensity_image
   imma1 = r1[0].image
   imma2 = r2[0].image
   im1T = ndimage.distance_transform_edt(imi1)
   im2T = ndimage.distance_transform_edt(imi2)
   
   mm =  np.max([np.max(im1T),np.max(im2T)])
   mim = np.min([np.max(im1T),np.max(im2T)])
   """ metric 3 = distance transform val """
   ratioDT =  mim/mm
   #
   wx1 = np.nonzero(im1T)
   mux1 =  0.96*(np.max(im1T)/np.mean(im1T[wx1]))
   wx2 = np.nonzero(im2T) 
   mux2 =  0.96*(np.max(im2T)/np.mean(im2T[wx2]))
    
   ttx1,tty1 = np.where(im1T >=mux1*np.mean(im1T[wx1]))
   ttx2,tty2 = np.where(im2T >=  mux1*np.mean(im2T[wx2]))
   """ max location indicates the shape """
   MaxDT = np.array([[np.linalg.norm(x-y) for x in np.column_stack([ttx1, tty1])] 
                      for y in np.column_stack([ttx2,tty2])])
   # x = tf.reshape(MaxDT, [1, MaxDT.shape[0], MaxDT.shape[1], 1])
   # max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(6, 6),
   #                                 strides=(4,4), padding='valid',  dtype='float64') 
   # xx = max_pool_2d(x).numpy()
   if len(MaxDT) > 0:
       mean_DTd  = np.mean(np.exp(-(np.ravel(MaxDT)-np.mean(MaxDT))/(np.var(MaxDT)+0.1))) #xx[0,:,:,0])
   else: mean_DTd =0
   """ metric 4 = IOU of areas """
   d = im1+im2
   xx  = np.sum([x>0 for x in im1])
   yy  = np.sum([x>0 for x in im2])
   sum_diff = np.sum([x > 1 for x in d])
   iou_area = 0.6*(np.min([xx,yy])/np.max([xx,yy])+ sum_diff/(np.sum(d)-sum_diff))
   return im1T, im2T, imma1I, imma2I, iou_ratio, iou_area, centroids_distance, ratioDT, mean_DTd
   

def return_nil(*A):  
    for x in zip(*A):
        print(x)


def compute_corr(im1,im2, i1, i2, j):
     # print(' compute corr',j)
     im1T, im2T, imma1, imma2 = ([] for i in range(4))
    
     if (np.sum(im1) > 0) and (np.sum(im2)>0):
         # imma1I, imma2I, iou, centroids_distance, ratioDT, min_val_DT
 
        (im1T, im2T, imma1, imma2, iou_ratio, iou_area, 
         centroids_distance, ratioDT, mean_DTd) = match_shape( im1, im2, i1, i2, j)
        # X = match_descriptors(imma1,imma2)
        ll = [0.3, 0.15, 0.3, 0.1, 0.15]
        corri = ll[0]*iou_ratio+ll[1]*iou_area+ll[2]*(1-centroids_distance)+ll[3]*ratioDT+ll[4]* mean_DTd      
     else: 
         corri = .0
         
     return  im1T, im2T, imma1, imma2, corri
#compare_masks(ff_mask, embed2, i1, i2)
def compare_masks(mm1, embed2, i1, i2):
    # mask1 sara' la ripetizione n volte per n =size(mask2,00)
    # della prima maschera
    
    J = np.arange(0,len(embed2), 1)
    Z  = [compute_corr(x,y, i1, i2, j) for  (x,y, j) in zip(mm1,embed2,J)]
    if len(Z) > 0:
        im1t, im2t, imma1, imma2, U = zip(*Z)
    else: 
        im1t, im2t, imma1, imma2 = ([] for i in range(4))
        U = .0
    return im1t, im2t, imma1, imma2, U
##
"""  BBS COMAPRISON """

#2.  Comparing bbs
def color_histogram(hh, immr, immg, immb, edges, len_edges):
    # weights = np.ones_like(hh)/len(hh)
    hh[:,0] = tfp.stats.histogram(immr, edges)[1:-1]
    hh[:,1] = tfp.stats.histogram(immg, edges)[1:-1]
    hh[:,2] = tfp.stats.histogram(immb, edges)[1:-1]
    # hhx = hh*weights
    hhx = hh.flatten()
    hh =  np.zeros((len_edges-3, 3))
    return hhx, hh

def Ihistogram(hh, imm, edges, len_edges):
    imm1 = imm.ravel()
    # weights = 3*np.ones_like(hh[:,0])/len(hh[:,0])
    hist = tfp.stats.histogram(imm, edges)[1:-1]
    # hhx = np.array(hist*weights)
    # hist = hist[0:120]
    
    return np.array(hist[0:70])


    
def  gen_hist(bbs, imask, tmask, image, edges, len_edges, k): # 
    len_edges = cfg.LEN_EDGES
    rem_imm = np.zeros((len(bbs),3*cfg.LENH +cfg.FF +  74) ) 
   
    """ Uncomment to control the bb correspondence"""
    # fig, ax = plt.subplots()
    # ax.imshow(image)
    imask = [x for x in imask if len(x)>0]
    tmask = [x for x in tmask if len(x)>0]
    for tt, (bb, mk, mt) in enumerate(zip(bbs, imask, tmask)):
        # print(tt, bb,  type(mk))
        if  not any(np.array(bb)>0):
           hhX =np.random.rand(1, 3*cfg.LENH +cfg.FF +  4)
        
        else:
            bb_int = np.array([int(x) for x in bb])
            bbx = bb_int.copy()
            bbx[ bbx <0] = 0
            features = image[bbx[1]:bbx[3], bbx[0]:bbx[2], :]*1.
            featuresN = features/255.
            labFeat = rgb2lab(featuresN, illuminant='D50')
            labFeat[:,:,0] =  2* labFeat[:,:,0]/ 100 -1
            labFeat[:,:,1:] = labFeat[:,:,1:] / 127
            lF = labFeat + np.abs(np.min(labFeat))
            lF = ((lF/np.max(lF))*255.)
            
            # plt.imshow(np.uint8(lF))
             
        ##
            hh = np.zeros((len_edges-3, 3))
            ffx = resize(features, (64,64), clip=True,  anti_aliasing= True)
            PF1 = tuple(pyramid_gaussian(ffx, max_layer=2, downscale=3, sigma=3, order=1, mode='reflect', cval=0, multichannel=True))
            U =[]
            for J in PF1:
                  ch, hh = color_histogram(hh, J[:,:,0], J[:,:,1], J[:,:,2], edges, len_edges)
                  U.append(ch)
            F1 =  np.concatenate([U[0], U[1], U[2]])
           
            hhL, hh = color_histogram(hh, lF[:,:,0], lF[:,:,1], lF[:,:,2], edges, len_edges)
            hhf, hh = color_histogram(hh, features[:,:,0],features[:,:,1],features[:,:,2], edges, len_edges)
           
            mk = mk*1.0
            hhg, hh = color_histogram(hh, mk[:,:,0], mk[:,:,1], [mk[:,:,2]], edges, len_edges)
            hht = Ihistogram(hh, mt, edges, len_edges)
            
            ### merge 
            if type(bb) == list:
               np.array(bb)
            A = [hhf, hhg, hht, hhL,  F1, bb]  
            n = sum(map(len, A))
            Ni = [x.shape[0]/n for x in A]
            B = [y*z for (y,z) in zip(A,Ni)]
            hhX = np.concatenate(B)
        rem_imm[tt,:] = hhX # 
        
    return rem_imm

####
def embeddingx(frame, w):
    embed = np.zeros((max(w)+1, frame.shape[1]))
    i = 0
    for x in w:
        embed[x,:] = frame[i,:]
        i = i+1
        
    U = np.sum(embed, axis = 1)
    dd = np.where(U==0) 
    div = np.random.randint(30,200,1)
    embed[dd,:] = np.random.rand(1, frame.shape[1])/div
    return embed
#
# frame1 = frame_hist1.copy()
# frame2 =frame_hist2.copy()    
def compare_frames_new(frame1, frame2, lev1, lev2):
    # fR =lev1
    # tO =lev2
    # m =  frame1.shape[0]
    # d =  frame2.shape[0]
    corri =[]
    # list_corr1 = np.zeros((m,d))
    
    if len(lev1)> 0:
       embed1 = embeddingx(frame1, lev1)
    else:
       embed1 = frame1
    if len(lev2) > 0:
       embed2 = embeddingx(frame2, lev2)
    else:
        embed2 = frame2
    #    
    m1 =  embed1.shape[0]
    d1 =  embed2.shape[0]
    list_corr = np.zeros((m1, d1))
    # for u, ff in enumerate(frame1)):
    for u, ff in enumerate(embed1):
        # print(u, ff[0], f1, t1)
        ff_frame = matlib.repmat(ff,d1,1)
        
       # corri =  np.array(tfp.stats.correlation(ff_frame,frame2, sample_axis=1, event_axis = None))
        corri =  np.array(tfp.stats.correlation(ff_frame,embed2, sample_axis=1, event_axis = None))
        if any(np.isnan(corri)):
              corri[np.isnan(corri)]  = 0
        # list_corr1[u,:]= corri
        list_corr[u,:] = corri
   
    return list_corr     

########### embedding for the mask
def embedding_masks(mask, w, shape):
    #qui l'embedding e' una lista
    embed = [np.zeros_like(mask[0]) for _ in range(max(w)+1)]   #np.zeros((max(w)+1, mask.shape[1]))
    i = 0
    for x in w:
        embed[x] = mask[i]
        i = i+1
    return embed
     
#
# mask1 = mms1.copy()
# mask2 = mms2.copy()    
def compare_masks_new(mask1, mask2, i1,i2, lev1, lev2):
    cmmp =[]
    if len(lev1)> 0:
       sh = mask1[0].shape
       embed1 = embedding_masks(mask1, lev1, sh)
    else:
       embed1 = mask1
    if len(lev2) >0:
       sh = mask2[0].shape
       embed2 = embedding_masks(mask2, lev2, sh)
    else:
        embed2 = mask2
    #    
    m1 =  len(embed1)
    d1 =  len(embed2)
    list_corr = np.zeros((m1, d1))
    mT1, mT2, mI1, mI2, im_mask2 = ([] for i in range(5))
    
    saved = False
    for u, mm in enumerate(embed1):
        # print(u)
        ff_mask = [mm]*len(embed2)
        #compare_masks(mm1, masks2, i1, i2)
        im1T, im2T, im_mask1, im_mask2, cmmp = compare_masks(ff_mask, embed2, i1, i2)
        tt1 = [i for i, x in enumerate(im_mask1) if len(x)>0]
        if len(tt1) > 0:
           mI1.append(im_mask1[tt1[0]])
           mT1. append(im1T[tt1[0]])
        # if any(np.isnan(cmmp)):
        #       cmmp[np.isnan(cmmp)]  = 0
        list_corr[u,:] = np.array(cmmp)
        
        tt2 = [i for i, x in enumerate(im_mask2) if len(x)>0]
        if len(tt2) >= len(mask2) and (not saved):
            mI2 = [x for x in im_mask2 if len(x) > 0]
            mT2 = [x for x in im2T if len(x) > 0]
            saved = True
    return mT1, mT2, mI1, mI2, list_corr     

### general metrics
### Dice
def dice_sorens(mask1,mask2):
    m1C = 1-mask1
    m2C = 1-mask2
    m1C = 1-mask1
    m2C = 1-mask2
    TP = np.count_nonzero(mask1*mask2)
    TN = np.count_nonzero(m1C*m2C)
    FP= np.count_nonzero(m1C*m2C)
    FN= np.count_nonzero(mask1*m2C)   
    
    ## metrics
    dice = 2*TP/(FP + 2*TP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    diff_area = ((TP + FP) - (TP + FN)) 
    return dice, recall, precision, diff_area