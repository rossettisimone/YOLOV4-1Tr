#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:02:49 2020

@author: fiorapirri
"""


from os import listdir
from os.path import isfile, join
import argparse
#import cv2
import numpy as np
import sys
import os
import shutil
import random 
import math


def IOU(x,centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities) 

def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        #note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum+= max(IOU(X[i],centroids)) 
    return sum/n

def write_anchors_to_file(centroids,X,anchor_file):
    f = open(anchor_file,'w')
    
    anchors = centroids.copy()
    print(anchors.shape)

    for i in range(anchors.shape[0]):
        anchors[i][0]*=width_in_cfg_file/32.
        anchors[i][1]*=height_in_cfg_file/32.
         

    widths = anchors[:,0]
    sorted_indices = np.argsort(widths)

    print('Anchors = ', anchors[sorted_indices])
    
    a = np.reshape(anchors[sorted_indices], (anchors_per_level,levels,2))
    b = np.array([a[i]*s for i,s in enumerate(scales)],dtype=np.int32).flatten()
    
    print('Anchors FPN = ', b)
    
    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, '%(anchors[i,0],anchors[i,1]))

    #there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n'%(anchors[sorted_indices[-1:],0],anchors[sorted_indices[-1:],1]))
    
    for i in range(0,len(b)-3,2):
        f.write('%d,%d, '%(b[i],b[i+1]))
    
    #there should not be comma after last anchor, that's why
    f.write('%d, %d\n'%(b[-2],b[-1]))
    
    f.write('%f\n'%(avg_IOU(X,centroids)))
    print()

def kmeans(X,centroids,eps,anchor_file):
    
    N = X.shape[0]
    iterations = 0
    k,dim = centroids.shape
    prev_assignments = np.ones(N)*(-1)    
    iter = 0
    old_D = np.zeros((N,k))

    while True:
        D = [] 
        iter+=1           
        for i in range(N):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D) # D.shape = (N,k)
        
        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D))))
            
        #assign samples to centroids 
        assignments = np.argmin(D,axis=1)
        
        if (assignments == prev_assignments).all() :
            print("Centroids = ",centroids)
            write_anchors_to_file(centroids,X,anchor_file)
            return

        #calculate new centroids
        centroid_sums=np.zeros((k,dim),np.float)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]        
        for j in range(k):            
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))
        
        prev_assignments = assignments.copy()     
        old_D = D.copy()  

def main():
    
    global width_in_cfg_file
    global height_in_cfg_file
    global levels
    global scales
    global output_dir
    global num_clusters
    global anchors_per_level
    
    width_in_cfg_file = 416.
    height_in_cfg_file = 416.
    levels = 4
    anchors_per_level = 4
    scales = [4,8,16,32]
    output_dir = './info'
    num_clusters = levels * anchors_per_level
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    from utils import read_image
    from PIL import Image
    from utils import file_reader
    import config as cfg
    lines = []
    PATHS = ["kinetics_frames_masks_train_v1.0.json", "ava_frames_masks_train_v2.2.json"]
    for file_path in PATHS:
        lines += file_reader(file_path)
        print('Train Dataset {} loaded'.format(file_path))
    
    annotation_dims = []

    for line in lines:
        v_id =  line['v_id']
        frame_name = line['f_l'][30]
        w, h = int(line['w']), int(line['h'])
        ih, iw = (416, 416)
        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        for l in line['p_l']:
            box = np.array(l['bb_l'][30])
            box = np.clip(box,0,1)
            xx, yy = box[::2]*w/nw + dw, box[1::2]*h/nh + dh
            box=np.array([np.min(xx),np.min(yy),np.max(xx),np.max(yy)])
            box = np.clip(box,0,1)
            W,H = (box[2]-box[0]), (box[3]-box[1])       
            if W>1e-3 and H>1e-3 and W/H > 0.2 and H/W > 0.2:
                annotation_dims.append(tuple(map(float,(W,H))))
    annotation_dims = np.array(annotation_dims)
    print(annotation_dims.shape)
    eps = 0.005
    
    if num_clusters == 0:
        for num_clusters in range(1,11): #we make 1 through 10 clusters 
            anchor_file = join( output_dir,'anchors%d.txt'%(num_clusters))

            indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims,centroids,eps,anchor_file)
            print('centroids.shape', centroids.shape)
    else:
        anchor_file = join( output_dir,'anchors%d.txt'%(num_clusters))
        indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
        centroids = annotation_dims[indices]
        kmeans(annotation_dims,centroids,eps,anchor_file)
        print('centroids.shape', centroids.shape)

if __name__=="__main__":
    main()