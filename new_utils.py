#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:30:15 2021

@author: 
"""

import tensorflow as tf
from utils import entry_stop_gradients, preprocess_mrcnn, mrcnn_class_loss_graph, mrcnn_bbox_loss_graph,\
     mrcnn_mask_loss_graph, smooth_l1_loss, filter_inputs, show_mAP
import gc
import config as cfg
from datetime import datetime
     
def compute_loss(model, labels, preds, embs, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_masks, training):
    # rpn loss 
    alb_total_loss, *rpn_loss_list = compute_loss_rpn(model, labels, preds, embs, training)
    # mrcnn loss
    alb_loss, *mrcnn_loss_list = compute_loss_mrcnn(model, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_masks)
    alb_total_loss += alb_loss
    alb_total_loss *= 0.5

    return [alb_total_loss] + rpn_loss_list + mrcnn_loss_list

def compute_loss_rpn(model, labels, preds, embs, training):
    rpn_box_loss = []
    rpn_class_loss = []
    for label, pred, emb in zip(labels, preds, embs):
        lbox, lconf = compute_loss_rpn_level(label, pred, emb)
        rpn_box_loss.append(lbox)
        rpn_class_loss.append(lconf)
    rpn_box_loss, rpn_class_loss = tf.reduce_mean(rpn_box_loss,axis=0), tf.reduce_mean(rpn_class_loss,axis=0)
    alb_loss = tf.math.exp(-model.s_r)*rpn_box_loss + tf.math.exp(-model.s_c)*rpn_class_loss \
                + (model.s_r + model.s_c) #Automatic Loss Balancing        
    return alb_loss, rpn_box_loss, rpn_class_loss

def compute_loss_rpn_level(label, pred, emb):
    pbox = pred[..., :4]
    pconf = pred[..., 4:6]
    tbox = label[...,:4]
    tconf = label[...,4]
    mask = tf.greater(tconf,0.0)
    if tf.greater(tf.reduce_sum(tf.cast(mask,tf.float32)),0.0):
        lbox = tf.reduce_mean(smooth_l1_loss(y_true = tf.boolean_mask(tbox,mask),y_pred = tf.boolean_mask(pbox,mask)))
    else:
        lbox = tf.constant(0.0)
    non_negative_entry = tf.cast(tf.greater_equal(tconf[...,tf.newaxis],0.0),tf.float32)
    pconf = entry_stop_gradients(pconf, non_negative_entry) # stop gradient for regions labeled -1 below CONF threshold, look dataloader
    tconf = tf.cast(tf.where(tf.less(tconf, 0.0), 0.0, tconf),tf.int32)
    lconf =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tconf,logits=pconf)) # apply softmax and do non negative log likelihood loss 
    return lbox, lconf

def compute_loss_mrcnn(model, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_masks):
    # prepare the ground truth
    mrcnn_class_loss = mrcnn_class_loss_graph(target_class_ids, pred_class_logits)
    mrcnn_box_loss = mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox)
    mrcnn_mask_loss = mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks)
    if tf.greater(tf.reduce_sum(proposals),0.0) :
        alb_loss = tf.math.exp(-model.s_mc)*mrcnn_class_loss + tf.math.exp(-model.s_mr)*mrcnn_box_loss \
                + tf.math.exp(-model.s_mm)*mrcnn_mask_loss + (model.s_mr + model.s_mc + model.s_mm) #Automatic Loss Balancing        
    else:
        alb_loss = mrcnn_class_loss + mrcnn_box_loss + mrcnn_mask_loss
    return alb_loss, mrcnn_class_loss, mrcnn_box_loss, mrcnn_mask_loss

@tf.function
def distributed_train_step(central_storage_strategy, model, dist_data,  optimizer):
    per_replica_losses = central_storage_strategy.run(train_step, args=(model, dist_data, optimizer,))
    return central_storage_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
  
@tf.function
def distributed_val_step(central_storage_strategy, model, dist_data):
    per_replica_losses = central_storage_strategy.run(val_step, args=(model, dist_data,))
    return central_storage_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

def train_step(model, data, optimizer):
    training = True
    image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
    labels = [label_2, labe_3, label_4, label_5]
    with tf.GradientTape() as tape:
        preds, embs, proposals, pred_class_logits, pred_class, pred_bbox, pred_mask = model(image, training=training)
        proposals = proposals[...,:4]
        target_class_ids, target_bbox, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks) # preprocess and tile labels according to IOU
        alb_total_loss, *loss_list = compute_loss(model, labels, preds, embs, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_mask, training)
        
        gradients = tape.gradient(alb_total_loss, model.trainable_variables)
        optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, model.trainable_variables) if grad is not None)
    return [alb_total_loss] + loss_list

def val_step(model, data):
    training = False
    image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
    labels = [label_2, labe_3, label_4, label_5]
    predictions = model(image, training=training)
    preds, embs, proposals, pred_class_logits, pred_class, pred_bbox, pred_mask = predictions
    proposals = proposals[...,:4]
    target_class_ids, target_bbox, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks) # preprocess and tile labels according to IOU
    alb_total_loss, *loss_list = compute_loss(model, labels, preds, embs, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_mask, training)
    
    return [alb_total_loss] + loss_list, predictions

def distributed_fit(central_storage_strategy, model, optimizer, dataset, writer, folder, epoch = 1, epochs = cfg.EPOCHS, batch_size = cfg.BATCH,\
        steps_train = cfg.STEPS_PER_EPOCH_TRAIN, steps_val = cfg.STEPS_PER_EPOCH_VAL,\
        freeze_bkbn_epochs = 2, freeze_bn = False):
    train_generator = dataset.train_ds.repeat().filter(filter_inputs).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_generator = dataset.val_ds.repeat().filter(filter_inputs).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    dist_dataset_train = central_storage_strategy.experimental_distribute_dataset(train_generator).__iter__()
    dist_dataset_val = central_storage_strategy.experimental_distribute_dataset(val_generator)
    step_train = 1
    step_val = 1
    while epoch < epochs:
        freeze_model(model,  trainable = True)
        if freeze_bn: # small batch size 
            freeze_batch_norm(model, trainable = False) 
        if epoch < freeze_bkbn_epochs:
            freeze_backbone(model, trainable = False)
        while step_train < epoch * steps_train :
            data = dist_dataset_train.next()
            losses = distributed_train_step(central_storage_strategy, model, data, optimizer)
            loss_summary(writer,  model, optimizer, step_train, losses, training=True)
            print_loss(epoch, step_train, steps_train, losses, training=True)
            denses_summary(writer, model, step_train, training = True)
            step_train += 1
        freeze_model(model,  trainable = False)
        gc.collect()
        val = dist_dataset_val.__iter__() # use always same batchs
        mean_AP = []
        while step_val < epoch * steps_val :
            data = val.next()
            losses, predictions = distributed_val_step(central_storage_strategy, model, data) 
            mean_AP.append(show_mAP(data, predictions))
            loss_summary(writer, model, optimizer, step_val, losses, training=False)
            print_loss(epoch, step_val, steps_val, losses, training=False)
            denses_summary(writer, model, step_val, training = False)
            mAP_summary(writer, step_val, tf.reduce_mean(mean_AP))
            step_val += 1
        path = "./{}/weights/model_lr{:0.5f}_ep{}_mAP{:0.5f}_date{}.tf".format(folder, optimizer.lr.numpy(), epoch, tf.reduce_mean(mean_AP),datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        model.save(path)
        gc.collect()
        epoch += 1
        
def fit(model,  optimizer, dataset, writer, folder, epoch = 1, epochs = cfg.EPOCHS, batch_size = cfg.BATCH,\
        steps_train = cfg.STEPS_PER_EPOCH_TRAIN, steps_val = cfg.STEPS_PER_EPOCH_VAL,\
        freeze_bkbn_epochs = 2, freeze_bn = False):
    train_generator = dataset.train_ds.repeat().filter(filter_inputs).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).__iter__()
    val_generator = dataset.val_ds.repeat().filter(filter_inputs).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    step_train = 1
    step_val = 1
    while epoch < epochs:
        freeze_model(model,  trainable = True)
        if freeze_bn: # small batch size 
            freeze_batch_norm(model, trainable = False) 
        if epoch < freeze_bkbn_epochs:
            freeze_backbone(model, trainable = False)
        while step_train < epoch * steps_train :
            data = train_generator.next()
            losses = train_step(model, data,  optimizer)
            loss_summary(writer, model, optimizer, step_train, losses, training=True)
            print_loss(epoch, step_train, steps_train, losses, training=True)
            denses_summary(writer, model, step_train, training = True)
            step_train += 1
        freeze_model(model,  trainable = False)
        gc.collect()
        val = val_generator.__iter__() # use always same batchs
        mean_AP = []
        while step_val < epoch * steps_val :
            data = val.next()
            losses, predictions = val_step(model, data) 
            mean_AP.append(show_mAP(data, predictions))
            loss_summary(writer, model, optimizer, step_val, losses, training=False)
            print_loss(epoch, step_val, steps_val, losses, training=False)
            denses_summary(writer, model, step_val, training = False)
            mAP_summary(writer, step_val, tf.reduce_mean(mean_AP))
            step_val += 1
        path = "./{}/weights/model_lr{:0.5f}_ep{}_mAP{:0.5f}_date{}.tf".format(folder, optimizer.lr.numpy(), epoch, tf.reduce_mean(mean_AP),datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        model.save(path)
        gc.collect()
        epoch += 1

@tf.function
def infer(model, image):
    preds, embs, proposals, logits, probs, bboxes, masks = model(image, training = False)
    return preds, embs, proposals, logits, probs, bboxes, masks

def print_loss(epoch, step, steps, losses, training=True):
    if training:
        res = "=> EPOCH {}  TRAIN STEP {}/{}  auto_loss_bal: "\
            "{:0.5f}  rpn_box_loss: {:0.5f}  rpn_class_loss: {:0.5f}  "\
            "".format(epoch, step % steps, steps,
                      losses[0], losses[1], losses[2])
        res += "mrcnn_class_loss: {:0.5f}  mrcnn_box_loss: {:0.5f}"\
            "  mrcnn_mask_loss: {:0.5f}".format(losses[-3], losses[-2], losses[-1])
        tf.print(res)
    else:
        res = "=> EPOCH {}  VAL STEP {}/{}  auto_loss_bal: {:0.5f}  rpn_box_loss: "\
            "{:0.5f}   rpn_class_loss: {:0.5f} ".format(epoch, step % steps, steps,
             losses[0], losses[1], losses[2])
        res += "mrcnn_class_loss: {:0.5f}  mrcnn_box_loss: "\
                "{:0.5f}   mrcnn_mask_loss: {:0.5f}".format(losses[-3], losses[-2], losses[-1])
        tf.print(res)

def loss_summary(writer, model, optimizer, step, losses, training=True):
    with writer.as_default():
        scope = 'train' if training else 'val'
        with tf.name_scope(scope):
            with tf.name_scope('loss'):
                tf.summary.scalar("auto_loss_bal", tf.squeeze(losses[0]), step=step)
                tf.summary.scalar("lr", optimizer.lr, step=step)
                with tf.name_scope('rpn'):
                    tf.summary.scalar("s_c", model.s_c, step=step)
                    tf.summary.scalar("s_r", model.s_r, step=step)
                    tf.summary.scalar("rpn_box_loss", tf.squeeze(losses[1]), step=step)
                    tf.summary.scalar("rpn_class_loss", tf.squeeze(losses[2]), step=step)
                with tf.name_scope('mrcnn'): 
                    tf.summary.scalar("s_mc", model.s_mc, step=step)
                    tf.summary.scalar("s_mr", model.s_mr, step=step)
                    tf.summary.scalar("s_mm", model.s_mm, step=step)
                    tf.summary.scalar("mrcnn_class_loss", tf.squeeze(losses[-3]), step=step)
                    tf.summary.scalar("mrcnn_box_loss", tf.squeeze(losses[-2]), step=step)
                    tf.summary.scalar("mrcnn_mask_loss", tf.squeeze(losses[-1]), step=step)
    writer.flush()

def denses_summary(writer, model, step, training):
    with writer.as_default():
        scope = 'train' if training else 'val'
        with tf.name_scope(scope):
            for layer in model.layers:
                if layer.name == 'mrcnn_class_logits' or layer.name == 'mrcnn_bbox_fc' or layer.name == 'mrcnn_mask_fc':
                    with tf.name_scope(layer.name):
                        with tf.name_scope('weights'):
                            variable_summaries(layer.layer.kernel, step)
                        with tf.name_scope('biases'):
                            variable_summaries(layer.layer.bias, step)
    writer.flush()

def mAP_summary(writer, step, mean_AP):
    with writer.as_default():
        with tf.name_scope('val'):
            with tf.name_scope('mAP_0.5:0.05:0.95'):
                tf.summary.scalar("mean_AP", tf.squeeze(mean_AP), step=step)

def freeze_model(model,  trainable = True):
    model.trainable = trainable
    model.s_c._trainable = trainable
    model.s_r._trainable = trainable
    model.s_mc._trainable = trainable
    model.s_mr._trainable = trainable
    model.s_mm._trainable = trainable

def freeze_batch_norm(model, trainable = False):  
    for layer in model.layers:
        try:
            bn_layer_name = layer.name
            if bn_layer_name[:10] == 'batch_norm':
                bn_layer = model.get_layer(bn_layer_name)
                bn_layer._trainable = trainable
        except:
            continue
        try:
            bn_layer_name = layer.layer.name # TimeDistributed hides the name
            if bn_layer_name[:10] == 'batch_norm':
                bn_layer = model.get_layer(bn_layer_name)
                bn_layer._trainable = trainable
        except:
            continue

def freeze_backbone(model, trainable = False):
    cutoff = 78 # 77 convolutions and batch normalizations
    conv_0 = int(model.layers[1].name.split('_')[-1]) if not model.layers[1].name == 'conv2d' else 0
    batch_0 = int (model.layers[2].name.split('_')[-1]) if not model.layers[2].name == 'batch_normalization' else 0
    for i in range(0,cutoff):
        k = i + conv_0
        j = i + batch_0
        conv_layer_name = 'conv2d_%d' %k if k > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'
        conv_layer = model.get_layer(conv_layer_name)
        bn_layer = model.get_layer(bn_layer_name)
        conv_layer._trainable = trainable
        bn_layer._trainable = trainable
            
def variable_summaries(var, step):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean, step=step)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev, step=step)
    tf.summary.scalar('max', tf.reduce_max(var), step=step)
    tf.summary.scalar('min', tf.reduce_min(var), step=step)
    tf.summary.histogram('histogram', var, step=step)

#def plot(model):
#    tf.keras.utils.plot_model(bkbn.model, to_file='cspdarknet53.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
#    tf.keras.utils.plot_model(neck.build_graph(), to_file='panet.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
#    tf.keras.utils.plot_model(head.build_graph(), to_file='yolov4.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
#    tf.keras.utils.plot_model(fpn_classifier.build_graph(), to_file='mrcnn_box.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
#    tf.keras.utils.plot_model(fpn_mask.build_graph(), to_file='mrcnn_mask.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
#    tf.keras.utils.plot_model(CustomProposalLayer().build_graph(), to_file='prop.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)

def custom_build(model, writer, step, folder):
    tf.summary.trace_on(graph=True, profiler=False)
    model.build((tf.newaxis, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3))
    with writer.as_default():
        tf.summary.trace_export(
                name="model",
                step=step,
                profiler_outdir="./{}/logdir".format(folder))
     
def adapt_lr(optimizer, epoch, epochs):
    if epoch < epochs * 0.5 :
        lr = cfg.LR
    elif epoch <= epochs * 0.75 and epoch > epochs * 0.5:
        lr = cfg.LR * 0.1
    else:
        lr = cfg.LR * 0.01
    optimizer.lr.assign(lr)
    
def save(model, name):
    model.save_weights(name)

def load(model, name):
    model.load_weights(name)
    