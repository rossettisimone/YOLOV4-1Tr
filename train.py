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
import tensorflow_addons as tfa
from loader import DataLoader 
from datetime import datetime
from model import get_model
from layers import  FreezeBackbone, EarlyStoppingAtMinLoss, EarlyStoppingRPN

#%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

folder = "{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
os.mkdir(folder)
logdir = os.path.join(folder, 'logdir')
os.mkdir(logdir)

# save config file
with open('config.py', mode='r') as in_file, open('{}/config.txt'.format(folder), mode='w') as out_file:
    out_file.write(in_file.read())

writer = tf.summary.create_file_writer(logdir)
writer.set_as_default()
        
optimizer = tfa.optimizers.SGDW( weight_decay = cfg.WD, \
                                learning_rate = cfg.LR, momentum = cfg.MOM, \
                                nesterov = False, clipnorm = cfg.GRADIENT_CLIP)

callbacks = tf.keras.callbacks.TensorBoard(
                                log_dir=logdir, histogram_freq=1, write_graph=True,
                                write_images=True, update_freq='batch', profile_batch='2, 4',
                                embeddings_freq=1, embeddings_metadata=None)

filepath = os.path.join(folder, 'weights')
os.mkdir(filepath)
filepath = os.path.join(filepath,'model.{epoch:02d}-{val_alb_total_loss:.3f}.h5')
             
# Create a callback that saves the model's weights
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                save_weights_only=True,
                                                monitor='val_alb_total_loss',
                                                mode='min',
                                                save_best_only=False,
                                                verbose = 1)

GPUs = ["GPU:"+i for i in cfg.GPU.split(',')]
strategy = tf.distribute.MirroredStrategy(GPUs)
GLOBAL_BATCH = cfg.BATCH * strategy.num_replicas_in_sync

with strategy.scope(): 
    dataset = DataLoader(batch_size = GLOBAL_BATCH, shuffle=True, augment=True)
    train_dataset = dataset.train_ds
    val_dataset = dataset.val_ds
    model = get_model()
    model.compile(optimizer)

early = EarlyStoppingRPN(patience1 = 3, patience2 = 5)
freeze = FreezeBackbone(n_epochs = 2)
model.fit(dataset.train_ds, epochs = cfg.EPOCHS, steps_per_epoch = cfg.STEPS_PER_EPOCH_TRAIN, \
          validation_data = dataset.val_ds, validation_steps = cfg.STEPS_PER_EPOCH_VAL,\
          validation_freq = 1, max_queue_size = GLOBAL_BATCH * 10,
          callbacks = [callbacks, checkpoint, freeze, early], use_multiprocessing = True, workers = 48)

model.evaluate(val_dataset, batch_size = GLOBAL_BATCH, callbacks = [callbacks], steps = cfg.STEPS_PER_EPOCH_VAL)
