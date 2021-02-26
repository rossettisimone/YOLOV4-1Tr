#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:08:40 2021

@author: Simone Rossetti
"""    
#%%%%%%%%%%%%%%%%%%%%%%%%%%% BUILD ENV %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import env

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import config as cfg
import tensorflow as tf
from loader import DataLoader 
from model import get_model

#%%%%%%%%%%%%%%%%%%%%%%%%%%% CHECKPOINT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model = get_model()

model.summary()

model.load_weights('/home/fiorapirri/tracker/weights/model.09--10.160.h5')

model.trainable = False

#%%%%%%%%%%%%%%%%%%%%%%%%%%% FPS TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import timeit

input_data = tf.random.uniform((1, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3))
#model.predict(input_layer); # Warm Up

trials = 100

model.infer(input_data);

print("Fps:", trials/timeit.timeit(lambda: model.infer(input_data), number=trials))

#%%%%%%%%%%%%%%%%%%%%%%%%%%% DATASET ENCODING TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from loader import DataLoader
from utils import show_infer, show_mAP, draw_bbox, filter_inputs
import matplotlib.pyplot as plt

ds = DataLoader(shuffle=True, augment=False)
iterator = ds.train_ds.filter(filter_inputs).repeat().batch(1).__iter__()
data = iterator.next()
image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes = data
draw_bbox(image[0].numpy(), bboxs = gt_bboxes[0].numpy(), masks=tf.transpose(gt_masks[0],(1,2,0)).numpy(), conf_id = None, mode= 'PIL')
plt.imshow(tf.reduce_sum(tf.reduce_sum(label_2[0],axis=0),axis=-1))
plt.show()
plt.imshow(tf.reduce_sum(tf.reduce_sum(label_3[0],axis=0),axis=-1))
plt.show()
plt.imshow(tf.reduce_sum(tf.reduce_sum(label_4[0],axis=0),axis=-1))
plt.show()
plt.imshow(tf.reduce_sum(tf.reduce_sum(label_5[0],axis=0),axis=-1))
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%% DATASET DECODING TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from utils import decode_labels 

p = [label_2,label_3,label_4,label_5]
proposals = decode_labels(p)
draw_bbox(image[0].numpy(), bboxs = proposals[0,:,:4].numpy()*cfg.TRAIN_SIZE, mode= 'PIL')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PREDICTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import matplotlib.pyplot as plt
from loader import DataLoader 
import time
from utils import show_infer, draw_bbox, show_mAP, encode_labels

def _round(vec):
    threshold = 0.3
    return tf.where(vec>threshold,1,0)
i = 0
sec = 0
AP = 0
ds = DataLoader(shuffle=True, augment=False)
iterator = ds.train_ds.unbatch().batch(1)
_ = model.infer(iterator.__iter__().next()[0])

for data in iterator.take(10):
    image, gt_mask, gt_masks, gt_bboxes = data
    start = time.perf_counter()
    predictions = model.infer(image)
    preds, embs, proposals, pred_class_logits, pred_class, pred_bbox, pred_mask = predictions
    end = time.perf_counter()-start
    i+=1
    sec += end
    print(i/sec)
    label_2, label_3, label_4, label_5 = tf.map_fn(encode_labels, (gt_bboxes, gt_mask), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
    data_ = image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes
    show_infer(data_, predictions)
    AP += show_mAP(data_, predictions)
    mAP = AP/i    
    print(mAP)
   draw_bbox(image[0].numpy(), bboxs = gt_bboxes[0].numpy(), masks=tf.transpose(gt_masks[0],(1,2,0)).numpy(), conf_id = None, mode= 'PIL')
    plt.imshow(_round(pred_mask[0,0,:,:,1]))
    plt.show()
