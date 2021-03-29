#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:08:40 2021

@author: Simone Rossetti
"""    
#%%%%%%%%%%%%%%%%%%%%%%%%%%% BUILD ENV %%%%%%%%%%%%%%%%%%%%%%%%%%%%§%%%%%%
import env


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import config as cfg
import tensorflow as tf
from loader_ytvos import DataLoader 
from model import get_model

#%%%%%%%%%%%%%%%%%%%%%%%%%%% CHECKPOINT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import tensorflow_addons as tfa

model = get_model()

#model.summary()

optimizer = tfa.optimizers.SGDW( weight_decay = cfg.WD, \
                                learning_rate = cfg.LR, momentum = cfg.MOM, \
                                nesterov = False, clipnorm = cfg.GRADIENT_CLIP)
model.compile(optimizer)

#from loader_avakin import DataLoader
from utils import encode_labels, preprocess_mrcnn
from layers import yolov4_plus1_proposal_graph
from model import compute_loss
#
#ds = DataLoader(shuffle=True, augment=False)
#iterator = ds.train_ds.unbatch().batch(1).__iter__()
image, gt_masks, gt_bboxes = iterator.next()
gt_bboxes
label_2, label_3, label_4, label_5 = tf.map_fn(encode_labels, gt_bboxes, fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))


training = True
labels = [label_2, label_3, label_4, label_5]
#with tf.GradientTape() as tape:
preds, proposals, pred_masks = model(image, training=training)
#preds = labels

#proposals = yolov4_plus1_proposal_graph(preds)
class_ids = proposals[...,5]
conf = proposals[...,4]
proposal = proposals[...,:4]

target_class_ids, target_masks = preprocess_mrcnn(proposal, gt_bboxes, gt_masks) # preprocess and tile labels according to IOU
#pred_masks = target_masks[...,None]

alb_total_loss, *loss_list = compute_loss(model, labels, preds, proposal, target_class_ids, target_masks, pred_masks, training)
#gradients = tape.gradient(alb_total_loss, model.trainable_variables)
#optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, model.trainable_variables))
#
#for (grad, var) in zip(gradients, model.trainable_variables):
#    print(var.name)
#model.load_weights('/home/fiorapirri/tracker/weights/model.07--10.531.h5')
#
#model.trainable = False

#%%%%%%%%%%%%%%%%%%%%%%%%%%% FPS TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import timeit

input_data = tf.random.uniform((1, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3))
#model.predict(input_layer); # Warm Up

trials = 100

model.infer(input_data);

print("Fps:", trials/timeit.timeit(lambda: model.infer(input_data), number=trials))

#%%

import timeit

from loader_ytvos import DataLoader 

ds = DataLoader(shuffle=True, augment=True)
#rle_arr = ds.annotation[100]['segmentations'][0]
#%%
iterator = ds.train_ds.unbatch().batch(1).__iter__()

trials = 100

print("Time:", timeit.timeit(lambda: iterator.next(), number=trials)/trials)

#%%%%%%%%%%%%%%%%%%%%%%%%%%% DATASET ENCODING TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from loader_ytvos import DataLoader
from utils import show_infer, show_mAP, draw_bbox, filter_inputs, encode_labels, xyxy2xywh, crop_and_resize,decode_ground_truth, decode_mask
import matplotlib.pyplot as plt

ds = DataLoader(shuffle=True, augment=True)
iterator = ds.train_ds.unbatch().batch(1).__iter__()

#%%
from utils import decode_labels
for i in range(1):
    data = iterator.next()
    image, gt_masks, gt_bboxes = data
    label_2, label_3, label_4, label_5 = tf.map_fn(encode_labels, gt_bboxes, fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
#    data = image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes 
    gt_masks = tf.map_fn(crop_and_resize, (xyxy2xywh(gt_bboxes)/cfg.TRAIN_SIZE, tf.cast(tf.greater(gt_bboxes[...,4],-1.0),tf.float32), gt_masks), fn_output_signature=tf.float32)
    gt_bbox, gt_class_id, gt_mask = decode_ground_truth(gt_masks[0], gt_bboxes[0])
    draw_bbox(image[0].numpy(), box = gt_bbox, mask=gt_mask, class_id = gt_class_id, class_dict = ds.class_dict, mode= 'PIL')
    plt.imshow(tf.reduce_sum(tf.reduce_sum(label_2[0],axis=0),axis=-1))
    plt.show()
    plt.imshow(tf.reduce_sum(tf.reduce_sum(label_3[0],axis=0),axis=-1))
    plt.show()
    plt.imshow(tf.reduce_sum(tf.reduce_sum(label_4[0],axis=0),axis=-1))
    plt.show()
    plt.imshow(tf.reduce_sum(tf.reduce_sum(label_5[0],axis=0),axis=-1))
    plt.show()
    predictions = [label_2,label_3,label_4,label_5]
    proposals = decode_labels(predictions)
    bbox, conf, classs = tf.split(proposals[0], (4,1,1),axis=-1)
    thr = conf[...,0]>cfg.CONF_THRESH
    bbox, conf, classs = bbox[thr], conf[thr], classs[thr]
    bbox= tf.round(bbox*cfg.TRAIN_SIZE)
    draw_bbox(image[0].numpy(), box = bbox, class_id = classs, class_dict = ds.class_dict, mode= 'PIL')

#%%
from model import get_model

model = get_model(infer=True)

#fine_tuning(model)

model.load_weights('/home/fiorapirri/tracker/weights/model.03--5.297.h5')

model.trainable = False

#model.summary()

#model.outputs


#%%

from loader_avakin import DataLoader
from utils import draw_bbox, encode_labels
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
ds = DataLoader(shuffle=True, augment=False)
iterator = ds.train_ds.unbatch().batch(1).__iter__()
data = iterator.next()
image, gt_masks, gt_bboxes = data

draw_bbox(image = image[0].numpy(), mode= 'PIL')

output = model.infer(image)

path = os.path.join('experiments')
os.makedirs(path,exist_ok=True)
path = os.path.join(path,'featires1')
os.makedirs(path,exist_ok=True)
Image.fromarray(np.array(image[0].numpy()*255,np.uint8)).save(os.path.join(path,'image.png'))
for i in range(0,7):
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
                img.convert("L").save(name)
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
                img.convert("L").save(name)
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
                    img.convert("L").save(name)
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
                img.convert("L").save(name)
            print(os.path.join(subpath,'e_{}'.format(str(j+2))))
    elif i==5:
        subpath = os.path.join(path,'rois_box_classification')
        os.makedirs(subpath,exist_ok=True)
        for j,o in enumerate(output[i]):
            o=o[0]
            for s in range(o.shape[0]):
                subsubpath = os.path.join(subpath,'roi_{}'.format(str(s)))
                os.makedirs(subsubpath,exist_ok=True)
                for c in range(o.shape[-1]):
                    img = o[s,:,:,c].numpy()*255
                    name = os.path.join(subsubpath,'channel_{}.png'.format(str(c)))
                    img = Image.fromarray(img)
                    w,h = img.size
                    img.convert("L").save(name)
            print(os.path.join(subpath,'roi_class_{}'.format(str(j))))
    elif i==6:
        subpath = os.path.join(path,'rois_mask')
        os.makedirs(subpath,exist_ok=True)
        for j,o in enumerate(output[i]):
            o=o[0]
            for s in range(o.shape[0]):
                subsubpath = os.path.join(subpath,'roi_{}'.format(str(s)))
                os.makedirs(subsubpath,exist_ok=True)
                for c in range(o.shape[-1]):
                    img = o[s,:,:,c].numpy()*255
                    name = os.path.join(subsubpath,'channel_{}.png'.format(str(c)))
                    img = Image.fromarray(img)
                    w,h = img.size
                    img.convert("L").save(name)
            print(os.path.join(subpath,'roi_mask_{}'.format(str(j))))

#%%%%%%%%%%%%%%%%%%%%%%%%%%% DATASET DECODING TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from utils import decode_labels 

p = [label_2,label_3,label_4,label_5]
proposals = decode_labels(p)
draw_bbox(image[0].numpy(), bboxs = proposals[0,:,:4].numpy()*cfg.TRAIN_SIZE, mode= 'PIL')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PREDICTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import matplotlib.pyplot as plt
from loader_ytvos import DataLoader 
#import time
from utils import show_infer, draw_bbox, show_mAP, encode_labels, crop_and_resize,xyxy2xywh, decode_ground_truth


ds = DataLoader(shuffle=True, augment=False)
iterator = ds.train_ds.unbatch().batch(1).__iter__()
_ = model.infer(iterator.next()[0])
#%%
AP = 0
#for i in range(1):
data = iterator.next()
image, gt_masks, gt_bboxes = data
gt_masks = tf.map_fn(crop_and_resize, (xyxy2xywh(gt_bboxes)/cfg.TRAIN_SIZE, tf.cast(tf.greater(gt_bboxes[...,4],-1.0),tf.float32), gt_masks), fn_output_signature=tf.float32)
predictions = model.infer(image)
box, conf, class_id, mask = predictions
label_2, label_3, label_4, label_5 = tf.map_fn(encode_labels, gt_bboxes, fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
data_ = image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes
show_infer(data_, predictions, ds.class_dict)
AP = show_mAP(data_, predictions)
print(AP)
gt_bbox, gt_class_id, gt_mask = decode_ground_truth(gt_masks[0], gt_bboxes[0])
draw_bbox(image[0].numpy(), box = gt_bbox, mask=gt_mask, class_id = gt_class_id, class_dict = ds.class_dict, mode= 'PIL')
