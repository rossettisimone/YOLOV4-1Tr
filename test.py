#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 19:25:16 2021

@author: fiorapirri
"""
import tensorflow as tf
from utils import *
import config as cfg

def main():
    test_check_proposals()
    test_nms_proposals()
    test_preprocess_mrcnn();
    test_loss_mrcnn()
    
def test_check_proposals(): # on CPU may return error due to tf.gather (which zero pad only on GPU)
    proposals = tf.random.uniform((20,1000,5))
    proposals1 = check_proposals_tensor(proposals)
    proposals2 = tf.map_fn(check_proposals, proposals, fn_output_signature = tf.float32)
    
    assert tf.reduce_all(proposals1 == proposals2)
    
def test_nms_proposals(): # on CPU may return error due to tf.gather (which zero pad only on GPU)
    proposals = tf.random.uniform((20,1000,5))
    proposals1 = nms_proposals_tensor(proposals)
    proposals2 = tf.map_fn(nms_proposals, proposals, fn_output_signature = tf.float32)
    
    assert tf.reduce_all(proposals1 == proposals2)
    
    
def test_preprocess_mrcnn():
    proposals_1 = tf.random.uniform((2,10,2))*0.5
    proposals_2 = tf.random.uniform((2,10,2))*0.5 + 0.5
    proposals = tf.concat([proposals_1, proposals_2], axis=-1)
    bad_proposals = tf.zeros((2,10,4))
    proposals = tf.concat([proposals, bad_proposals], axis=1)
    gt_bboxes_1 = tf.random.uniform((2,5,2))*0.5
    gt_bboxes_2 = tf.random.uniform((2,5,2))*0.5 + 0.5
    gt_bboxes = tf.concat([gt_bboxes_1, gt_bboxes_2], axis=-1)* cfg.TRAIN_SIZE
    gt_masks = tf.round(tf.random.uniform((2,5,28,28)))
#    proposals = tf.zeros((2,20,4))
#    gt_bboxes = tf.zeros((2,10,4))
#    gt_masks = tf.zeros((2,10,28,28))
    target_class_ids, target_bbox, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks)
    
    return target_class_ids, target_bbox, target_masks

def test2_preprocess_mrcnn():
    proposals = tf.constant([[[0,0,0.5,0.5]]])*0.9
    bad_proposals = tf.zeros((1,9,4))
    proposals = tf.concat([proposals, bad_proposals], axis=1)
    gt_bboxes = tf.constant([[[0,0,0.5,0.5]]])* cfg.TRAIN_SIZE
    gt_masks = tf.round(tf.random.uniform((1,1,28,28)))
    target_class_ids, target_bbox, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks)
    
    return target_class_ids, target_bbox, target_masks

def test_loss_mrcnn():
    
    target_class_ids, target_bbox, target_masks = test_preprocess_mrcnn()
    pred_class_logits = tf.tile(target_class_ids[...,None],(1,1,2))
    pred_class_logits_1 = tf.where(pred_class_logits[...,0:1]==1,-1,1)
    pred_class_logits_2 = tf.where(pred_class_logits[...,1:2]==1,1,-1)
    pred_class_logits = tf.concat([pred_class_logits_1,pred_class_logits_2],axis=-1)*10
    pred_class_logits = tf.cast(pred_class_logits,tf.float32)
    pred_bbox = tf.tile(target_bbox[:,:,None,:],(1,1,2,1))
    pred_masks = tf.tile(target_masks[...,None],(1,1,1,1,2))
    loss = mrcnn_class_loss_graph(target_class_ids, pred_class_logits)
    loss += mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox)
    loss += mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks)
    
    assert tf.equal(loss, 0)
    
if __name__ == '__main__':
    
    main()
    
    