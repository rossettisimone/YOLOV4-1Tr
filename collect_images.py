#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 13:10:27 2021

@author: fiorapirri
"""
import os
import config as cfg
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPU

import tensorflow as tf
#tf.get_logger().setLevel('WARNING')
# tf.compat.v1.reset_default_graph()
# tf.debugging.enable_check_numerics()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_devices = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_devices), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)
else: 
    print('No GPU found')

#mirrored_strategy = tf.distribute.MirroredStrategy(devices=[device.name for device in logical_devices])
#print ('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
#with mirrored_strategy.scope():

#from models import MSDS
#from loader import DataLoader 

# tensorboard --logdir /media/data4/Models/simenv/tracker/logdir --port 6006
# scp /home/fiorapirri/Documents/workspace/tracker4/weights/yolov4.weights alcor@Alcor:/media/data4/Models/simenv/tracker/weights/yolov4.weights

#ds = DataLoader(shuffle=True, data_aug=False)
#model = MSDS(data_loader = ds, emb = False, mask = True)
#model.custom_build()
##model.plot()
##model.bkbn.model.summary() 
##model.neck.summary()
##model.head.summary()
#model.summary()
#model.load('./weights/MSDS_noemb_mask_14_-6.43556_2021-02-14-02-21-56.tf')
#model.trainable = False #

import matplotlib.pyplot as plt
import numpy as np
from utils import draw_bbox, xyxy2xywh, decode_delta, xywh2xyxy, decode_delta_map, check_proposals,nms_proposals, preprocess_mrcnn
from loader import DataLoader 
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from moviepy.editor import ImageSequenceClip
import gc
from PIL import Image
from utils import data_labels
ds = DataLoader(shuffle=True, augment=False)
iterator = ds.train_ds.unbatch().batch(1).take(10)
video = []
for i,data in enumerate(iterator):   
    image, gt_mask, gt_masks, gt_bboxes = data

    label_2, label_3, label_4, label_5 = tf.map_fn(data_labels, (gt_bboxes, gt_mask), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))

    threshold = 0.5
    start = time.time()
    preds, embs, proposals, logits, probs, bboxes, masks = model.infer(image)
    print(time.time()-start)
    v = tf.reduce_sum(tf.cast(tf.reduce_sum(tf.cast(proposals[0,:,4]>threshold, tf.int32)),tf.int32))
    pro = proposals[0,:v,:4]
    pr = xyxy2xywh(pro)
    bb = bboxes[0,:v,1,:4]
    bb = decode_delta(bb, pr)
    bb = xywh2xyxy(bb)
    bb = tf.clip_by_value(bb,0.0,1.0)
    bb = tf.round(bb*cfg.TRAIN_SIZE)
    mas = masks[0,:v,:,:,1]
    vv = tf.reduce_sum(tf.cast(tf.reduce_sum(gt_bboxes[0],axis=-1)>0,tf.int32))
    
    fig, axs = plt.subplots(3,5, figsize=(20,20))
    ((ax1,ax2,ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10),(ax11,ax12,ax13,ax14,ax15))=axs
    ax1.imshow(draw_bbox(image[0].numpy(), masks = tf.transpose(gt_masks[0,:vv],(1,2,0)).numpy(), bboxs = gt_bboxes[0,:vv].numpy()))
    ax2.imshow(tf.reduce_sum(tf.reduce_sum(label_2[0],axis=0),axis=-1))
    ax3.imshow(tf.reduce_sum(tf.reduce_sum(label_3[0],axis=0),axis=-1))
    ax4.imshow(tf.reduce_sum(tf.reduce_sum(label_4[0],axis=0),axis=-1))
    ax5.imshow(tf.reduce_sum(tf.reduce_sum(label_5[0],axis=0),axis=-1))

    ax6.imshow(draw_bbox(image[0].numpy(),pro.numpy()*cfg.TRAIN_SIZE,conf_id = proposals[0,:v,4].numpy()))
    ax7.imshow(tf.reduce_sum(tf.reduce_sum(preds[0][0],axis=0),axis=-1))
    ax8.imshow(tf.reduce_sum(tf.reduce_sum(preds[1][0],axis=0),axis=-1))
    ax9.imshow(tf.reduce_sum(tf.reduce_sum(preds[2][0],axis=0),axis=-1))
    ax10.imshow(tf.reduce_sum(tf.reduce_sum(preds[3][0],axis=0),axis=-1))
    
    ax11.imshow(draw_bbox(image[0].numpy(),bboxs=bb, masks=tf.transpose(mas,(1,2,0)).numpy(), conf_id=proposals[0,:v,4].numpy()))
    ax12.imshow(tf.reduce_sum(tf.reduce_sum(embs[0],axis=0),axis=-1))
    ax13.imshow(tf.reduce_sum(tf.reduce_sum(embs[1],axis=0),axis=-1))
    ax14.imshow(tf.reduce_sum(tf.reduce_sum(embs[2],axis=0),axis=-1))
    ax15.imshow(tf.reduce_sum(tf.reduce_sum(embs[3],axis=0),axis=-1))
    plt.show()
#
#    canvas = FigureCanvas(fig)
#    canvas.draw()
#    s, (width, height) = canvas.print_to_buffer()
#    im = np.fromstring(s, np.uint8).reshape((height, width, 4)) 
#    plt.close()
#    Image.fromarray(im).save('./keyframes/{}.png'.format(i))
#    gc.collect()