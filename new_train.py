#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:08:40 2021

@author: Simone Rossetti
"""    
#%%%%%%%%%%%%%%%%%%%%%%%%%%% BUILD ENV %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pre_run
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%% FPS TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import tensorflow as tf
from loader import DataLoader 
import tensorflow_addons as tfa
from datetime import datetime
from new_utils import distributed_fit, fit
from new_model import build_model
import config as cfg

strategy = tf.distribute.MirroredStrategy()

folder = "{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

writer = tf.summary.create_file_writer("./{}/logdir".format(folder))

optimizer = tfa.optimizers.SGDW( weight_decay = cfg.WD, learning_rate = cfg.LR, momentum = cfg.MOM, nesterov = False)# clipnorm = cfg.GRADIENT_CLIP

with strategy.scope():
    
    dataset = DataLoader(shuffle=True, data_aug=True)
    
    model = build_model()

model.summary()

distributed_fit(strategy, model, optimizer, dataset, writer, folder, freeze_bkbn_epochs = 2, freeze_bn = False)

#%%%%%%%%%%%%%%%%%%%%%%%%%%% FPS TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#import timeit
#from new_utils import infer
#input_layer = tf.random.uniform((1, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3))
#infer(model, input_layer); # Warm Up
#
#trials = 100
#print("Fps:", trials/timeit.timeit(lambda: infer(model, input_layer), number=trials))

#%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#from loader import DataLoader
#from utils import show_infer, show_mAP, draw_bbox, filter_inputs
#import matplotlib.pyplot as plt
#
#ds = DataLoader(shuffle=True, data_aug=False)
#iterator = ds.train_ds.filter(filter_inputs).repeat().batch(1).__iter__()
#data = iterator.next()
#image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes = data
#draw_bbox(image[0].numpy(), bboxs = gt_bboxes[0].numpy(), masks=tf.transpose(gt_masks[0],(1,2,0)).numpy(), conf_id = None, mode= 'PIL')
#plt.imshow(tf.reduce_sum(tf.reduce_sum(label_2[0],axis=0),axis=-1))
#plt.show()
#plt.imshow(tf.reduce_sum(tf.reduce_sum(label_3[0],axis=0),axis=-1))
#plt.show()
#plt.imshow(tf.reduce_sum(tf.reduce_sum(label_4[0],axis=0),axis=-1))
#plt.show()
#plt.imshow(tf.reduce_sum(tf.reduce_sum(label_5[0],axis=0),axis=-1))
#plt.show()
#
#from utils import decode_labels 
#p = [label_2,label_3,label_4,label_5]
#proposals = decode_labels(p)
#draw_bbox(image[0].numpy(), bboxs = proposals[0,:,:4].numpy()*cfg.TRAIN_SIZE, mode= 'PIL')
#from utils import *

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%