#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:08:40 2021

@author: Simone Rossetti
"""    
#%%%%%%%%%%%%%%%%%%%%%%%%%%% BUILD ENV %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pre_run

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import tensorflow as tf
from loader import DataLoader 
import tensorflow_addons as tfa
from datetime import datetime
import config as cfg
import os
from new_model import Model, FreezeBackbone
from utils import filter_inputs

#%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

folder = "{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

logdir = os.path.join(folder, 'logdir')

writer = tf.summary.create_file_writer(logdir)

writer.set_as_default()
        
optimizer = tfa.optimizers.SGDW( weight_decay = cfg.WD, \
                                learning_rate = cfg.LR, momentum = cfg.MOM, \
                                nesterov = False, clipnorm = cfg.GRADIENT_CLIP)

callbacks = tf.keras.callbacks.TensorBoard(
                                log_dir=logdir, histogram_freq=1, write_graph=True,
                                write_images=True, update_freq='batch', profile_batch='2, 4',
                                embeddings_freq=1, embeddings_metadata=None)

filepath = os.path.join(folder, 'weights', "cp-{epoch:04d}.ckpt")

# Create a callback that saves the model's weights
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = filepath, \
                               monitor='val_alb_total_loss', verbose=1, save_best_only=False,
                               save_weights_only=True, save_freq='epoch')

GPUs = ["GPU:"+i for i in cfg.GPU.split(',')]

strategy = tf.distribute.MirroredStrategy(GPUs)

GLOBAL_BATCH = cfg.BATCH * strategy.num_replicas_in_sync

with strategy.scope():
    
    dataset = DataLoader(batch_size = GLOBAL_BATCH, shuffle=True, data_aug=True)
    
    train_dataset = dataset.train_ds.repeat()
    
    val_dataset = dataset.val_ds
    
    model = Model()
    
    model.compile(optimizer)

freeze = FreezeBackbone(n_epochs = 2, model = model)

model.fit(dataset.train_ds, epochs = cfg.EPOCHS, steps_per_epoch = cfg.STEPS_PER_EPOCH_TRAIN, \
          validation_data = dataset.val_ds, validation_steps = cfg.STEPS_PER_EPOCH_VAL,\
          validation_freq = 1, max_queue_size = GLOBAL_BATCH * 10,
          callbacks = [callbacks, checkpoint, freeze])#use_multiprocessing = True, workers = 24

model.evaluate(dataset.val_ds, batch_size = GLOBAL_BATCH, callbacks = [callbacks])

model.load_weights(filepath.format(epoch = 3))

#%%%%%%%%%%%%%%%%%%%%%%%%%%% CHECKPOINT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model = Model()

model.model.summary()

checkpoint = tf.train.Checkpoint(model.model)

checkpoint.restore('./weights/cp-0005.ckpt')

model.model.trainable = False

#%%%%%%%%%%%%%%%%%%%%%%%%%%% FPS TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import timeit

input_layer = tf.random.uniform((1, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3))
#model.predict(input_layer); # Warm Up

trials = 100

@tf.function
def infer(model, input_data):
    return model(input_data)

infer(model.model, input_layer);

print("Fps:", trials/timeit.timeit(lambda: infer(model.model, input_layer), number=trials))

#%%%%%%%%%%%%%%%%%%%%%%%%%%% DATASET TEST %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PREDICTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import matplotlib.pyplot as plt
from loader import DataLoader 
import time
from utils import show_infer, draw_bbox, show_mAP, data_labels

i = 0
sec = 0
AP = 0
ds = DataLoader(shuffle=True, data_aug=False)
iterator = ds.val_ds.unbatch().batch(1)
_ = infer(model.model, iterator.__iter__().next()[0])
for data in iterator.take(10):
    image, gt_masks, gt_bboxes = data
    start = time.perf_counter()
    predictions = infer(model.model, image)
    end = time.perf_counter()-start
    i+=1
    sec += end
    print(i/sec)
    label_2, label_3, label_4, label_5 = tf.map_fn(data_labels, (gt_bboxes, gt_masks), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
    data = image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes
    show_infer(data, predictions)
    AP += show_mAP(data, predictions)
    mAP = AP/i    
    print(mAP)
#    draw_bbox(image[0].numpy(), bboxs = gt_bboxes[0].numpy(), masks=tf.transpose(gt_masks[0],(1,2,0)).numpy(), conf_id = None, mode= 'PIL')
