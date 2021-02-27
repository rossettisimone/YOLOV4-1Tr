#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:08:40 2021

@author: Simone Rossetti
"""    
#%%%%%%%%%%%%%%%%%%%%%%%%%%% BUILD ENV %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import env

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import config as cfg
import tensorflow as tf
import tensorflow_addons as tfa
from loader import DataLoader 
from model import get_model
from layers import  FreezeBackbone, EarlyStoppingRPN
from utils import folders

#%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logdir, filepath = folders()

writer = tf.summary.create_file_writer(logdir)
writer.set_as_default()

callbacks = tf.keras.callbacks.TensorBoard(
                                log_dir=logdir, histogram_freq=1, write_graph=True,
                                write_images=True, update_freq='batch', profile_batch='2, 4',
                                embeddings_freq=1, embeddings_metadata=None)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, save_weights_only=True,
                                                monitor='val_alb_total_loss', mode='min',
                                                save_best_only=False, verbose = 1)

early = EarlyStoppingRPN(patience1 = 5, patience2 = 10)

freeze = FreezeBackbone(n_epochs = 2)

optimizer = tfa.optimizers.SGDW( weight_decay = cfg.WD, \
                                learning_rate = cfg.LR, momentum = cfg.MOM, \
                                nesterov = False, clipnorm = cfg.GRADIENT_CLIP)

GPUs = ["GPU:"+i for i in cfg.GPU.split(',')]
strategy = tf.distribute.MirroredStrategy(GPUs)
GLOBAL_BATCH = cfg.BATCH * strategy.num_replicas_in_sync

with strategy.scope(): 
    dataset = DataLoader(batch_size = GLOBAL_BATCH, shuffle=True, augment=True)
    train_dataset = dataset.train_ds
    val_dataset = dataset.val_ds
    model = get_model(pretrained_backbone = True)
    model.compile(optimizer)

model.fit(dataset.train_ds, epochs = cfg.EPOCHS, steps_per_epoch = cfg.STEPS_PER_EPOCH_TRAIN, \
          validation_data = dataset.val_ds, validation_steps = cfg.STEPS_PER_EPOCH_VAL,\
          validation_freq = 1, max_queue_size = GLOBAL_BATCH * 10,
          callbacks = [callbacks, checkpoint, early, freeze], use_multiprocessing = True, workers = 48)

model.evaluate(val_dataset, batch_size = GLOBAL_BATCH, callbacks = [callbacks], steps = cfg.STEPS_PER_EPOCH_VAL)
