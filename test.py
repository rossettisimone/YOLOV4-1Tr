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

model.load_weights('/home/fiorapirri/tracker/weights/model.07--10.531.h5')

model.trainable = False

#%%%%%%%%%%%%%%%%%%%%%%%%%%% FPS TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import timeit

input_data = tf.random.uniform((1, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3))
#model.predict(input_layer); # Warm Up

trials = 100

model.infer(input_data);

print("Fps:", trials/timeit.timeit(lambda: model.infer(input_data), number=trials))

#%%

import timeit

from loader import DataLoader 

ds = DataLoader(shuffle=True, augment=True)

iterator = ds.train_ds.__iter__()

trials = 100

print("Time:", timeit.timeit(lambda: iterator.next(), number=trials)/trials)

#%%%%%%%%%%%%%%%%%%%%%%%%%%% DATASET ENCODING TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from loader import DataLoader
from utils import show_infer, show_mAP, draw_bbox, filter_inputs, encode_labels
import matplotlib.pyplot as plt

ds = DataLoader(shuffle=True, augment=False)
iterator = ds.train_ds.unbatch().batch(1).__iter__()
data = iterator.next()
image, gt_mask, gt_masks, gt_bboxes = data
label_2, label_3, label_4, label_5 = tf.map_fn(encode_labels, (gt_bboxes, gt_mask), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
data = image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes 
draw_bbox(image[0].numpy(), bboxs = gt_bboxes[0].numpy(), masks=tf.transpose(gt_masks[0],(1,2,0)).numpy(), conf_id = None, mode= 'PIL')
plt.imshow(tf.reduce_sum(tf.reduce_sum(label_2[0],axis=0),axis=-1))
plt.show()
plt.imshow(tf.reduce_sum(tf.reduce_sum(label_3[0],axis=0),axis=-1))
plt.show()
plt.imshow(tf.reduce_sum(tf.reduce_sum(label_4[0],axis=0),axis=-1))
plt.show()
plt.imshow(tf.reduce_sum(tf.reduce_sum(label_5[0],axis=0),axis=-1))
plt.show()

#%%
from layers import fine_tuning

model = get_model()

fine_tuning(model)

model.summary()

model.load_weights('/home/fiorapirri/tracker/weights/model.18--7.873.h5')

model.trainable = False

#%%

from loader import DataLoader
from utils import draw_bbox, encode_labels
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
ds = DataLoader(shuffle=True, augment=False)
iterator = ds.train_ds.unbatch().batch(1).__iter__()
data = iterator.next()
image, gt_mask, gt_masks, gt_bboxes = data

draw_bbox(image = image[0].numpy(), mode= 'PIL')

output = model(image)

path = os.path.join('experiments')
os.makedirs(path,exist_ok=True)
path = os.path.join(path,'new_labelling')
os.makedirs(path,exist_ok=True)
Image.fromarray(np.array(image[0].numpy()*255,np.uint8)).save(os.path.join(path,'image.png'))
for i in range(0,4):
    if i==0:
        subpath = os.path.join(path,'backbone')
        os.makedirs(subpath,exist_ok=True)
        for j,o in enumerate(output[i]):
            subsubpath = os.path.join(subpath,'b_{}'.format(str(j+2)))
            os.makedirs(subsubpath,exist_ok=True)
            o=o[0]
            for c in range(o.shape[-1]):
                img = o[:,:,c].numpy()*255
                name = os.path.join(subsubpath,'channel_{}.png'.format(str(c)))
                img = Image.fromarray(img)
                w,h = img.size
                img.resize((w*5,h*5)).convert("L").save(name)
            print(os.path.join(subpath,'b_{}'.format(str(j+2))))
    elif i==1:
        subpath = os.path.join(path,'neck')
        os.makedirs(subpath,exist_ok=True)
        for j,o in enumerate(output[i]):
            subsubpath = os.path.join(subpath,'n_{}'.format(str(j+2)))
            os.makedirs(subsubpath,exist_ok=True)
            o=o[0]
            for c in range(o.shape[-1]):
                img = o[:,:,c].numpy()*255
                name = os.path.join(subsubpath,'channel_{}.png'.format(str(c)))
                img = Image.fromarray(img)
                w,h = img.size
                img.resize((w*5,h*5)).convert("L").save(name)
            print(os.path.join(subpath,'n_{}'.format(str(j+2))))
    elif i==2:
        subpath = os.path.join(path,'head_predictions')
        os.makedirs(subpath,exist_ok=True)
        for j,o in enumerate(output[i]):
            subsubpath = os.path.join(subpath,'p_{}'.format(str(j+2)))
            os.makedirs(subsubpath,exist_ok=True)
            o=o[0]
            for a in range(o.shape[0]):
                subsubsubpath = os.path.join(subsubpath,'a_{}'.format(str(a+1)))
                os.makedirs(subsubsubpath,exist_ok=True)
                for c in range(o.shape[-1]):
                    img = o[a,:,:,c].numpy()*255
                    name = os.path.join(subsubsubpath,'channel_{}.png'.format(str(c)))
                    img = Image.fromarray(img)
                    w,h = img.size
                    img.resize((w*5,h*5)).convert("L").save(name)
            print(os.path.join(subpath,'p_{}'.format(str(j+2))))
    elif i==3:
        subpath = os.path.join(path,'head_embeddings')
        os.makedirs(subpath,exist_ok=True)
        for j,o in enumerate(output[i]):
            subsubpath = os.path.join(subpath,'e_{}'.format(str(j+2)))
            os.makedirs(subsubpath,exist_ok=True)
            o=o[0]
            for c in range(o.shape[-1]):
                img = o[:,:,c].numpy()*255
                name = os.path.join(subsubpath,'channel_{}.png'.format(str(c)))
                img = Image.fromarray(img)
                w,h = img.size
                img.resize((w*5,h*5)).convert("L").save(name)
            print(os.path.join(subpath,'e_{}'.format(str(j+2))))
        

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
