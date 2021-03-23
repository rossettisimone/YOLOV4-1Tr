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
from loader_ytvos import DataLoader 
from model import get_model
from utils import  FreezeBackbone, EarlyStoppingRPN, fine_tuning, folders
# tf.config.optimizer.set_jit(True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

logdir, filepath = folders()

writer = tf.summary.create_file_writer(logdir)
writer.set_as_default()

callbacks = tf.keras.callbacks.TensorBoard(
                                log_dir=logdir, histogram_freq=1, write_graph=True,
                                write_images=True, update_freq='batch', profile_batch='2, 4',
                                embeddings_freq=1, embeddings_metadata=None)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, save_weights_only=True,
                                                monitor='val_alb_loss', mode='min',
                                                save_best_only=False, verbose = 1)

optimizer = tfa.optimizers.SGDW( weight_decay = cfg.WD, \
                                learning_rate = cfg.LR, momentum = cfg.MOM, \
                                nesterov = False, clipnorm = cfg.GRADIENT_CLIP)

GPUs = ["GPU:"+str(i) for i in range(len(cfg.GPU.split(',')))]
strategy = tf.distribute.MirroredStrategy(GPUs)
GLOBAL_BATCH = cfg.BATCH * strategy.num_replicas_in_sync

with strategy.scope(): 
    dataset = DataLoader(batch_size = GLOBAL_BATCH, shuffle=cfg.SHUFFLE, augment=cfg.DATA_AUGMENT)
    train_dataset = dataset.train_ds
    val_dataset = dataset.val_ds
    model = get_model()
    model.compile(optimizer)

train_history = model.fit(train_dataset, epochs = cfg.FINE_TUNING, steps_per_epoch = cfg.STEPS_PER_EPOCH_TRAIN, \
                      validation_data = val_dataset, validation_steps = cfg.STEPS_PER_EPOCH_VAL,\
                      validation_freq = 1, max_queue_size = GLOBAL_BATCH * 10,
                      callbacks = [callbacks, checkpoint], use_multiprocessing = True, workers = 48)

model.evaluate(val_dataset, batch_size = GLOBAL_BATCH, callbacks = [callbacks], steps = cfg.STEPS_PER_EPOCH_VAL)

with strategy.scope():
    model.load_weights(filepath.format(epoch = cfg.FINE_TUNING,\
                                       val_alb_loss = train_history.history['val_alb_loss'][cfg.FINE_TUNING-1]))
    model.compile(optimizer)
    fine_tuning(model)
    dataset.train_ds = dataset.initilize_train_ds() # shuffle

model.fit(train_dataset, initial_epoch = cfg.FINE_TUNING, epochs = cfg.EPOCHS, steps_per_epoch = cfg.STEPS_PER_EPOCH_TRAIN, \
          validation_data = val_dataset, validation_steps = cfg.STEPS_PER_EPOCH_VAL,\
          validation_freq = 1, max_queue_size = GLOBAL_BATCH * 10,
          callbacks = [callbacks, checkpoint], use_multiprocessing = True, workers = 48)

model.evaluate(val_dataset, batch_size = GLOBAL_BATCH, callbacks = [callbacks], steps = cfg.STEPS_PER_EPOCH_VAL)
