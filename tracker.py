#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:46:43 2021

@author: fiorapirri
"""



import env


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import config as cfg
import tensorflow as tf
from loader_ytvos import DataLoader 
from model import get_model
import numpy as np
from PIL import Image
import json
from utils import file_reader, prediction_preprocess, prediction_postprocess, unmold_mask_proposals, rle_encoding
from tqdm import tqdm
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model = get_model(infer=True)

#fine_tuning(model)

model.load_weights('/home/fiorapirri/tracker/weights/model.60--6.610.h5')

model.trainable = False
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DATASET = file_reader(cfg.YT_TEST_ANNOTATION_PATH)

IMG_PATH = os.path.join(cfg.YT_TEST_FRAMES_PATH)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VIDEOS = DATASET['videos']

CLASSES = {}

for obj in DATASET['categories']:
    CLASSES[int(obj['id'])] = str(obj['name'])
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#prediction{
#            "video_id" : int, 
#            "category_id" : int, 
#            "segmentations" : [RLE or [polygon] or None], 
#            "score" : float, 
#        }
        
NEW_DATASET = []
new_annotation_count = 0
CONF_THRESH = 0.6
for video_index in tqdm(range(len(VIDEOS)),ascii=True, desc='predicting'):
    video = VIDEOS[video_index]
    video_id = video['id']
    length = video['length']
    file_names = video['file_names']
    w = video['width']
    h = video['height']
    for i in range(length):
        # ground truth
        new_annotation_count += 1
        file_name = file_names[i]
        sub_path = file_name.split('.')[0]
        num_instances = 0
        new_annotation = dict()
        new_annotation['annotation_id'] = new_annotation_count
        new_annotation['video_id'] = video_id
        new_annotation['len_sequence'] = length
        new_annotation['num_sequence'] = i+1
        new_annotation['file_name'] = os.path.join('JPEGImages', file_name)
        new_annotation['height'] = h
        new_annotation['width'] = w
        # predictions
        img = np.array(Image.open(os.path.join(IMG_PATH, file_name)))
        img, hh, ww = prediction_preprocess(img)
        box, conf, class_id, mask = model.infer(tf.constant(img)[tf.newaxis])
        pred_instances = int(tf.reduce_sum(tf.cast(conf>=CONF_THRESH,tf.int32)).numpy())
        pred_confs = []
        pred_categories = []
        pred_bboxes = []
        pred_segmentations = []
        if pred_instances>0:
            conf = conf[0,:pred_instances].numpy()
            box = box[0,:pred_instances,:].numpy()
            class_id = class_id[0,:pred_instances].numpy()
            mask = mask[0,:pred_instances,:,:].numpy()
            mask = unmold_mask_proposals((mask, box)).numpy()
            bboxes, masks = prediction_postprocess(box, mask, hh, ww)
            pred_confs.extend([round(c,3) for c in conf.tolist()])
            pred_categories.extend([int(c) for c in class_id.tolist()])
            pred_bboxes.extend(bboxes.tolist())
            pred_segmentations.extend([rle_encoding(mask) for mask in masks]) 
        new_annotation['pred_instances'] = pred_instances
        new_annotation['pred_categories'] = pred_categories
        new_annotation['pred_confs'] = pred_confs
        new_annotation['pred_bboxes'] = pred_bboxes
        new_annotation['pred_segmentations'] = pred_segmentations
        
        NEW_DATASET.append(new_annotation)
#    print(time.time()-start)
#    break
NEW_INSTANCES = dict()
NEW_INSTANCES['categories'] = cfg.CLASS_YTVIS21
NEW_INSTANCES['annotations'] = NEW_DATASET

with open('pred_test_instances.json', 'w') as f:
    f.write(json.dumps(NEW_INSTANCES))
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from graph_tracking.mainVIS_Track import *

DATASET_ = file_reader('pred_test_instances.json')

ANNOTATIONS = DATASET_['annotations']

CATEGORIES = cfg.CLASS_YTVIS21

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#import numpy as np
#import os
#PATHS = []
#for root, dirs, files in os.walk(IMG_PATH):
#    for f in files:
#        PATHS.append(os.path.relpath(os.path.join(root, f), "."))
#PATHS = sorted(PATHS)
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#from PIL import Image
#images=[]
#w,h = 480,480
#for img in PATHS[:4000]:
#    im = Image.open(img)
#    W,H = im.size
#    scale = max(W/w,H/h)
#    im = im.resize((int(W/scale),int(H/scale)),2)
#    W,H = im.size
#    ww = 0 if W%2 == 0 else 1
#    hh = 0 if H%2 == 0 else 1
#    imm = np.zeros((h,w,3),dtype=np.uint8)
#    imm[240-H//2:240+H//2+hh,240-W//2:240+W//2+ww,:]=np.array(im,dtype=np.uint8)
#    images.append(imm)
#    
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#from moviepy.editor import ImageSequenceClip
#clip = ImageSequenceClip(images, fps=30)
#name = "test_set.mp4"
#clip.write_videofile(name) 
#    