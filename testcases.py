#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 19:25:16 2021

@author: fiorapirri
"""
import tensorflow as tf

def main():
    test_check_proposals()
    test_nms_proposals()
    test_preprocess_mrcnn();
    test_loss_mrcnn()
    test_encode_decode()
    test_encode_decode_loss()
    
def test_check_proposals(): # on CPU may return error due to tf.gather (which zero pad only on GPU)
    from utils import check_proposals_tensor, check_proposals
    proposals = tf.random.uniform((20,1000,5))
    proposals1 = check_proposals_tensor(proposals)
    proposals2 = tf.map_fn(check_proposals, proposals, fn_output_signature = tf.float32)
    
    assert tf.reduce_all(proposals1 == proposals2)
    
def test_nms_proposals(): # on CPU may return error due to tf.gather (which zero pad only on GPU)
    from utils import nms_proposals_tensor, nms_proposals
    proposals = tf.random.uniform((20,1000,5))
    proposals1 = nms_proposals_tensor(proposals)
    proposals2 = tf.map_fn(nms_proposals, proposals, fn_output_signature = tf.float32)
    
    assert tf.reduce_all(proposals1 == proposals2)
    
    
def test_preprocess_mrcnn():
    from utils import preprocess_mrcnn
    import config as cfg
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

def test_loss_mrcnn():
    from model import mrcnn_class_loss_graph,mrcnn_bbox_loss_graph,mrcnn_mask_loss_graph
    target_class_ids, target_bbox, target_masks = test_preprocess_mrcnn()
    pred_class_logits = tf.cast(target_class_ids,tf.float32)*2-1
    pred_class_logits = tf.concat([-pred_class_logits[...,None],pred_class_logits[...,None]], axis=-1)*10
    pred_bbox = tf.tile(target_bbox[:,:,None,:],(1,1,2,1))
    pred_masks = tf.tile(target_masks[...,None],(1,1,1,1,2))
    loss = mrcnn_class_loss_graph(target_class_ids, pred_class_logits)
    loss += mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox)
    loss += mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks)
    
    assert tf.equal(loss, 0)
    
def test_encode_decode():
    from loader import DataLoader
    from utils import draw_bbox, encode_labels
    from utils import preprocess_mrcnn
    from utils import decode_target_mask
    from utils import decode_labels 
    import numpy as np
    import config as cfg
    
    ds = DataLoader(shuffle=True, augment=True)
    iterator = ds.train_ds.unbatch().batch(1).__iter__()
    data = iterator.next()
    image, gt_mask, gt_masks, gt_bboxes = data
    draw_bbox(image[0].numpy(), bboxs = gt_bboxes[0].numpy(), masks=tf.transpose(gt_masks[0],(1,2,0)).numpy(), conf_id = np.arange(20), mode= 'PIL')
    label_2, label_3, label_4, label_5 = tf.map_fn(encode_labels, (gt_bboxes, gt_mask), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
    proposals = decode_labels([label_2,label_3,label_4,label_5])
    draw_bbox(image[0].numpy(), bboxs = proposals[0,:,:4].numpy()*cfg.TRAIN_SIZE,masks=tf.transpose(gt_masks[0],(1,2,0)).numpy(), conf_id = None, mode= 'PIL')
    target_class_ids, target_bbox, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks)
    bbox_mrcnn, conf_mrcnn, mask_mrcnn = decode_target_mask(proposals[0], target_class_ids[0], target_bbox[0], target_masks[0])
    draw_bbox(image[0].numpy(), bboxs = bbox_mrcnn[...,:4],masks=mask_mrcnn, conf_id = np.arange(20), mode= 'PIL')
    
    
def test_encode_decode_loss():
    from loader import DataLoader
    from utils import encode_labels
    from utils import preprocess_mrcnn
    from utils import decode_labels 
    
    ds = DataLoader(shuffle=True, augment=True)
    iterator = ds.train_ds.unbatch().batch(1).__iter__()
    data = iterator.next()
    image, gt_mask, gt_masks, gt_bboxes = data
    label_2, label_3, label_4, label_5 = tf.map_fn(encode_labels, (gt_bboxes, gt_mask), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
    proposals = decode_labels([label_2,label_3,label_4,label_5])
    target_class_ids, target_bbox, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks)
    from model import mrcnn_class_loss_graph,mrcnn_bbox_loss_graph,mrcnn_mask_loss_graph
    pred_class_logits = tf.cast(target_class_ids,tf.float32)*2-1
    pred_class_logits = tf.concat([-pred_class_logits[...,None],pred_class_logits[...,None]], axis=-1)*10
    pred_bbox = tf.tile(target_bbox[:,:,None,:],(1,1,2,1))
    pred_masks = tf.tile(target_masks[...,None],(1,1,1,1,2))
    loss = mrcnn_class_loss_graph(target_class_ids, pred_class_logits)
    loss += mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox)
    loss += mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks)
    assert tf.equal(loss, 0)
    
if __name__ == '__main__':
    
    main()
    
    