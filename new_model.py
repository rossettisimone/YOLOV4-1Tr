#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:08:40 2021

@author: 
"""
import os
import config as cfg

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPU

import gc 
gc.collect()

import tensorflow as tf

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
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import tensorflow as tf
from backbone import cspdarknet53_graph, load_weights_cspdarknet53, freeze_weights_cspdarknet53
from layers import yolov4_plus1_graph, yolov4_plus1_decode_graph, yolov4_plus1_proposal_graph,\
     fpn_classifier_graph_AFP, build_fpn_mask_graph_AFP
import config as cfg
from new_utils import distributed_fit, infer

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
################################ MODEL ###################################
##########################################################################
from loader import DataLoader 
import tensorflow_addons as tfa
from datetime import datetime

central_storage_strategy = tf.distribute.experimental.CentralStorageStrategy()

with central_storage_strategy.scope():

    dataset = DataLoader(shuffle=True, data_aug=True)

    input_layer = tf.keras.layers.Input((cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3))
    
    backbone = cspdarknet53_graph(input_layer) # may try a smaller backbone? backbone = cspdarknet53_tiny(input_layer)
    
    neck = yolov4_plus1_graph(backbone)
    
    rpn_predictions, rpn_embeddings = yolov4_plus1_decode_graph(neck)
    rpn_proposals = yolov4_plus1_proposal_graph(rpn_predictions)
    
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph_AFP([rpn_proposals[...,:4],rpn_embeddings])
    mrcnn_mask = build_fpn_mask_graph_AFP([rpn_proposals[...,:4],rpn_embeddings])

    model = tf.keras.Model(inputs=input_layer, outputs=[rpn_predictions, rpn_embeddings, \
                                                    rpn_proposals, mrcnn_class_logits, \
                                                    mrcnn_class, mrcnn_bbox, mrcnn_mask])

    load_weights_cspdarknet53(model, cfg.CSP_DARKNET53) # load backbone weights and set to non trainable
    
    freeze_weights_cspdarknet53(model)
    
    model.s_c = tf.Variable(initial_value=0.0, trainable=True)
    model.s_r = tf.Variable(initial_value=0.0, trainable=True)
    model.s_mc = tf.Variable(initial_value=0.0, trainable=True)
    model.s_mr = tf.Variable(initial_value=0.0, trainable=True)
    model.s_mm = tf.Variable(initial_value=0.0, trainable=True)
    
    model.summary()
    
    optimizer = tfa.optimizers.SGDW( weight_decay = cfg.WD, learning_rate = cfg.LR, momentum = cfg.MOM, nesterov = False)# clipnorm = cfg.GRADIENT_CLIP

    folder = "{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    writer = tf.summary.create_file_writer("./{}/logdir".format(folder))

    distributed_fit(central_storage_strategy, model,  optimizer, dataset, writer, folder, epoch = 1, epochs = cfg.EPOCHS, batch_size = cfg.BATCH,\
        steps_train = cfg.STEPS_PER_EPOCH_TRAIN, steps_val = cfg.STEPS_PER_EPOCH_VAL,\
        freeze_bkbn_epochs = 2, freeze_bn = False)

#%%%%%%%%%%%%%%%%%%%%%%%%%%% FPS TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import timeit

input_layer = tf.random.uniform((1, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3))
infer(input_layer); # Warm Up

trials = 100
print("Fps:", trials/timeit.timeit(lambda: infer(model, input_layer), number=trials))
