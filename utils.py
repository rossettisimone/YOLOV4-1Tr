#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 18:40:02 2020

@author: fiorapirri
"""


import json
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
import config as cfg
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from compute_ap import compute_ap_range

def filter_inputs(image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes):
    return tf.greater(tf.reduce_sum(gt_bboxes[...,:4]), 0.0) and tf.greater(tf.reduce_sum(gt_masks), 0.0)

def file_reader(file_name):
    with open(file_name) as json_file:
        return json.load(json_file)

def file_writer(file, file_name):
    with open(file_name, 'w') as f:
        f.write(json.dumps(file)) 

def image_preprocess(self, image, target_size): # for test
    ih, iw    = target_size
    h,  w, _  = image.shape
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)    
    image = Image.fromarray(image)
    image_resized = image.resize((nw, nh),Image.ANTIALIAS)
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.
    return image_paded

def imm_resize(img):
    imgx = img.resize((cfg.TRAIN_SIZE, cfg.TRAIN_SIZE), Image.ANTIALIAS)
    return np.array(imgx)/255

def mask_resize(img, width=cfg.TRAIN_SIZE, height=cfg.TRAIN_SIZE):
    img = Image.fromarray(img)
    imgx = img.resize((width,height), Image.ANTIALIAS)
    return np.array(imgx,dtype=np.uint8)

def mask_clamp(mask):
    if np.any(mask>1): # CDCL RETURN VALUES FROM 0 TO 7 FOR 0 - BACKGROUND, 1 - BOXES, 2 - HEADS, ARMS, LEGS..
        mask[mask==1]=0 # SET BOX LABEL TO 0
    mask = np.clip(mask,0,1)
    return np.array(mask,dtype=np.float32)

def img_tranfrom(img, bbox):
    width, height = img.size
    T = [width, height, width, height]
    mask = np.zeros((height,width))
    bbox= np.array(bbox)
    xmin, ymin, xmax, ymax = (bbox*T)
    mask[int(round(ymin)): int(round(ymax)), int(round(xmin)) : int(round(xmax))] = 1
    single_person = np.multiply(np.array(img),mask[:,:, np.newaxis])
    return single_person.astype(int), width, height

def read_image(img_path):
    return Image.open(img_path)

def scale_coords(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain_w = float(img_size[0]) / img0_shape[1]  # gain  = old / new
    gain_h = float(img_size[1]) / img0_shape[0]
    gain = min(gain_w, gain_h)
    pad_x = (img_size[0] - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, 0:4] /= gain
    coords[:, :4] = np.clip(coords[:, :4], min=0)
    return coords

#
def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes
    """
    N, M = tf.shape(box1)[0], tf.shape(box2)[0]
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = tf.math.maximum(b1_x1[:,None], b2_x1)
    inter_rect_y1 = tf.math.maximum(b1_y1[:,None], b2_y1)
    inter_rect_x2 = tf.math.minimum(b1_x2[:,None], b2_x2)
    inter_rect_y2 = tf.math.minimum(b1_y2[:,None], b2_y2)

    # Intersection area
    inter_area = tf.math.maximum(inter_rect_x2 - inter_rect_x1, 0) * tf.math.maximum(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b1_area = tf.broadcast_to(tf.reshape((b1_x2 - b1_x1) * (b1_y2 - b1_y1),(-1,1)),[N,M])
    b2_area = tf.broadcast_to(tf.reshape((b2_x2 - b2_x1) * (b2_y2 - b2_y1),(1,-1)),[N,M])

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def decode_proposal(proposal, mode='cut'):
    bbox = proposal[...,:4]
    bbox = tf.clip_by_value(bbox,0.0,1.0)
    bbox = tf.round(bbox*cfg.TRAIN_SIZE).numpy()
    conf = proposal[...,4].numpy()
    if mode == 'cut':
        mask = conf>cfg.CONF_THRESH
        return bbox[mask], conf[mask]
    else:
        return bbox, conf

def decode_mask(proposal, prob, bbox_mrcnn, mask_mrcnn, mode='keep'):
    bbox = proposal[...,:4]
    bbox = xyxy2xywh(bbox)
    bbox_mrcnn = bbox_mrcnn[...,1,:4]
    bbox_mrcnn = decode_delta(bbox_mrcnn, bbox)
    bbox_mrcnn = xywh2xyxy(bbox_mrcnn)
    bbox_mrcnn = tf.clip_by_value(bbox_mrcnn,0.0,1.0)
    bbox_mrcnn = tf.round(bbox_mrcnn*cfg.TRAIN_SIZE).numpy()
    conf_mrcnn = prob[...,1].numpy()
    mask_mrcnn = mask_mrcnn[...,1]
    mask_mrcnn = tf.transpose(mask_mrcnn,(1,2,0)).numpy()
    if mode == 'cut':
        mask = conf_mrcnn>cfg.CONF_THRESH
        pred_class_id = tf.cast(conf_mrcnn[mask]>0, tf.int32).numpy()
        return bbox_mrcnn[mask], pred_class_id, conf_mrcnn[mask], mask_mrcnn[...,mask]
    else:
        pred_class_id = tf.cast(conf_mrcnn>cfg.CONF_THRESH, tf.int32).numpy()
        return bbox_mrcnn, pred_class_id, conf_mrcnn, mask_mrcnn
    
def decode_ground_truth(gt_bboxes, gt_masks):
    gt_bbox = gt_bboxes[...,:4]
    gt_class_id = tf.cast(tf.reduce_sum(gt_bbox,axis=-1)>0,tf.int32)
    valid_gt = tf.math.maximum(1,tf.reduce_sum(gt_class_id))
    gt_bbox = gt_bbox[:valid_gt].numpy()
    gt_class_id = gt_class_id[:valid_gt].numpy()
    gt_mask = gt_masks[:valid_gt]
    gt_mask = tf.transpose(gt_mask,(1,2,0)).numpy()
    return gt_bbox, gt_class_id, gt_mask
    
def show_infer(data, prediction):
    image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
    nB = tf.shape(image)[0]
    if len(prediction)>3:
        preds, embs, proposals, logits, probs, bboxes, masks = prediction
        for i in range(nB):
            bbox_mrcnn, id_mrcnn, conf_mrcnn, mask_mrcnn = decode_mask(proposals[i], probs[i], bboxes[i], masks[i], 'cut')
            if len(bbox_mrcnn)>0:
                draw_bbox(image[i], bboxs = bbox_mrcnn, masks = mask_mrcnn, conf_id = conf_mrcnn, mode= 'PIL')
            else:
                show_image(tf.cast(image*255,tf.uint8).numpy()[i])
            tf.print('Found {} subjects'.format(len(bbox_mrcnn)))
    else:
        preds, embs, proposals = prediction
        for i in range(nB):
            bbox, conf = decode_proposal(proposals[i], 'cut')
            if len(bbox)>0:
                draw_bbox(image[i], prop = bbox, conf_id = conf, mode= 'PIL')
            else:
                show_image(tf.cast(image*255,tf.uint8).numpy()[i])
            tf.print('Found {} subjects'.format(len(bbox)))
    
def show_mAP(data, prediction, iou_thresholds = None, verbose = 0):
    image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
    nB = tf.shape(image)[0]
    mean_AP = []
    if len(prediction)>3: # if masks are present
        preds, embs, proposals, logits, probs, bboxes, masks = prediction
        for i in range(nB):
            gt_bbox, gt_class_id, gt_mask = decode_ground_truth(gt_bboxes[i], gt_masks[i])
            pred_box, pred_class_id, pred_score, pred_mask = decode_mask(proposals[i], probs[i], bboxes[i], masks[i])
            if len(gt_bbox)>0: # this is never the case but better to put
                AP = compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                         pred_box, pred_class_id, pred_score, pred_mask,
                         iou_thresholds=iou_thresholds, verbose=verbose)
                mean_AP.append(AP)
    if len(mean_AP) == 0: # this is never the case, but better to exclude the situation
        mean_AP = [0.5]
    return np.mean(mean_AP)
#    else:
#        preds, embs, proposals = prediction
#        for i in range(nB):
            
    
def draw_bbox(image, bboxs=None, masks=None, conf_id=None, mode='return'):
    colors = []
    for name, code in ImageColor.colormap.items():
         colors.append(name)
#    random.shuffle(colors)
    img = Image.fromarray(np.array(image.numpy()*255,dtype=np.uint8))                   
    draw = ImageDraw.Draw(img)   
    if np.any(conf_id is None):
        conf_id = tf.zeros_like(bboxs)[...,0]
    if np.any(bboxs is not None):
        for i,(bbox,conf) in enumerate(zip(bboxs, conf_id)):
            if np.any(bbox>0):
                draw.rectangle(bbox, outline = colors[i%len(colors)]) 
                xy = ((bbox[2]+bbox[0])*0.5, (bbox[3]+bbox[1])*0.5)
                draw.text(xy, str(np.round(conf,3)), font=ImageFont.truetype("./other/arial.ttf"), fontsize = 15, fill=colors[i%len(colors)])
 
    img = np.array(img)
    if np.any(masks is not None):
        masks = np.transpose(masks,(2,0,1))
        for i, (mask, bbox) in enumerate(zip(masks, bboxs)):
            mask = Image.fromarray(mask)
            xyxy = np.array(bbox,dtype=np.int32)
            wh = np.array(bbox[2:4]-bbox[:2],dtype=np.int32)
            if np.all(wh>0):
                mask = np.round(mask.resize((wh[0], wh[1]),Image.ANTIALIAS))
                mask = np.array(Image.new('RGB', (wh[0], wh[1]), color = colors[i%len(colors)]))*np.tile(mask[...,None],(1,1,3))
                img[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2]][mask>0] = img[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2]][mask>0]*0.5+mask[mask>0]*0.5
    if mode == 'PIL':
        show_image(img)
    else:
        return img


def show_image(img):
    img = Image.fromarray(img)               
    img.show()
#
def xyxy2xywh(xyxy):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    # x, y are coordinates of center 
    # (x1, y1) and (x2, y2) are coordinates of bottom left and top right respectively. 
    xy = (xyxy[...,:2]+xyxy[...,2:4])*0.5
    wh = (xyxy[...,2:4]-xyxy[...,:2])
    return tf.concat([xy,wh],axis=-1)

#
def xywh2xyxy(xywh):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    # x, y are coordinates of center 
    # (x1, y1) and (x2, y2) are coordinates of bottom left and top right respectively. 
    x1y1 = xywh[...,:2] - xywh[...,2:4]*0.5
    x2y2 = xywh[...,:2] + xywh[...,2:4]*0.5
    return tf.concat([x1y1, x2y2], axis=-1)

#
def encode_target(target, anchor_wh, nA, nC, nGh, nGw):
    target = tf.cast(target, tf.float32)
    bbox = target[:,:4]/cfg.TRAIN_SIZE
    ids = target[:,4]
    
    gt_boxes = tf.clip_by_value(bbox, 0.0, 1.0)
    gt_boxes = xyxy2xywh(bbox)
    gt_boxes = gt_boxes * tf.cast([nGw, nGh, nGw, nGh], tf.float32)
    anchor_wh = tf.cast(anchor_wh, tf.float32)
    tbox = tf.zeros((nA, nGh, nGw, 4)) # batch size, anchors, grid size
    tconf = tf.zeros(( nA, nGh, nGw))
    tid = tf.zeros((nA, nGh, nGw, nC))-1
    
    anchor_mesh = generate_anchor(nGh, nGw, anchor_wh)
    anchor_list = tf.reshape(tf.transpose(anchor_mesh, (0,2,3,1)),(-1, 4))              # Shpae (nA x nGh x nGw) x 4 
    iou_pdist = bbox_iou(anchor_list, gt_boxes)                                      # Shape (nA x nGh x nGw) x Ng
    iou_max = tf.math.reduce_max(iou_pdist, axis=1)                              # Shape (nA x nGh x nGw), both
    max_gt_index = tf.math.argmax(iou_pdist, axis=1)
    
    iou_map = tf.reshape(iou_max, (nA, nGh, nGw))     
    gt_index_map = tf.reshape(max_gt_index,(nA, nGh, nGw))     
    
    id_index = iou_map > cfg.ID_THRESH
    fg_index = iou_map > cfg.FG_THRESH                                                    
    bg_index = iou_map < cfg.BG_THRESH 
    ign_index = tf.cast(tf.cast((iou_map < cfg.FG_THRESH),tf.float32) * tf.cast((iou_map > cfg.BG_THRESH),tf.float32),tf.bool)
    tconf = tf.where(fg_index, 1.0, tconf)
    tconf = tf.where(bg_index, 0.0, tconf)
    tconf = tf.where(ign_index, -1.0, tconf)

    gt_index = tf.boolean_mask(gt_index_map,fg_index)
    gt_box_list = tf.gather(gt_boxes,gt_index)
    gt_id_list = tf.gather(ids,tf.boolean_mask(gt_index_map,id_index))

    if tf.reduce_sum(tf.cast(fg_index,tf.float32)) > 0:
        tid = tf.scatter_nd(tf.where(id_index),  gt_id_list[:,None], (nA, nGh, nGw, nC))
        tid = tf.where(tf.equal(tid,0.0),  -1.0, tid)
        fg_anchor_list = tf.reshape(anchor_list,(nA, nGh, nGw, 4))[fg_index] 
        delta_target = encode_delta(gt_box_list, fg_anchor_list)
        tbox = tf.scatter_nd(tf.where(fg_index),  delta_target, (nA, nGh, nGw, 4))
#    tconf = tf.transpose(tconf,(0,2,1))
#    tbox = tf.transpose(tbox,(0,2,1,3))
#    tid = tf.transpose(tid,(0,2,1,3))
#    tconf, tbox, tid = tf.cast(tconf,tf.float32), tf.cast(tbox,tf.float32), tf.cast(tid,tf.float32)
    label = tf.concat([tbox,tconf[...,None],tid],axis=-1)
    label = tf.transpose(label,(0,2,1,3))
    return label



#def true_proposals_deltas(deltas):
#    condition = tf.logical_and(tf.greater(proposals[...,3],proposals[...,1]),tf.greater(proposals[...,2],proposals[...,0]))
#    valid_indices = tf.squeeze(tf.where(condition),axis=-1)
#    proposals = tf.gather(proposals,valid_indices)
#    return proposals

#
def generate_anchor(nGh, nGw, anchor_wh):
    nA = tf.shape(anchor_wh)[0]
    yy, xx =tf.meshgrid(tf.range(nGh), tf.range(nGw))
    mesh = tf.stack([xx, yy], axis=0)                                              # Shape 2, nGh, nGw
    mesh = tf.cast(tf.tile(mesh[tf.newaxis],(nA,1,1,1)),tf.float32)                          # Shape nA x 2 x nGh x nGw
    anchor_offset_mesh = tf.tile(anchor_wh[...,tf.newaxis][...,tf.newaxis],(1, 1, nGh,nGw))  # Shape nA x 2 x nGh x nGw
    anchor_mesh = tf.concat([mesh, anchor_offset_mesh], axis=1)                       # Shape nA x 4 x nGh x nGw
    return anchor_mesh

#
def encode_delta(gt_box_list, fg_anchor_list):
    '''
    :param: gt_box_list, shape (num_boxes, (gx, gy, gw, gh))
    :param: fg_anchor_list, shape (nA, nGh, nGw, (px, py, pw, ph))
    
    :output: nA, nGh, nGw, (dx, dy, dw, dh) (log)
    '''
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], \
                     gt_box_list[:, 2], gt_box_list[:, 3]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = tf.math.log(gw/pw)
    dh = tf.math.log(gh/ph)
    return tf.stack([dx, dy, dw, dh], axis=1)


def decode_delta(delta, fg_anchor_list):
    '''
    :param: delta, shape (total_anchors, (dx, dy, dw, dh)) (log)
    :param: fg_anchor_list, shape (total_anchors, (px, py, pw, ph))
    
    :output: total_anchors, (gx, gy, gw, gh)
    '''
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    dx, dy, dw, dh = delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3]
    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * tf.math.exp(dw)
    gh = ph * tf.math.exp(dh)
    return tf.stack([gx, gy, gw, gh], axis=1)


def decode_delta_map(delta_map, anchors):
    '''
    :param: delta_map, shape (nB, nA, nGh, nGw, 4)
    :param: anchors, shape (nA,4)
    
    :output: nB, nA, nGh, nGw, (gx, gy, gw, gh)
    '''
    _, nA, nGh, nGw, _ = delta_map.shape
    nB = tf.shape(delta_map)[0] # error in building if None
    anchor_mesh = generate_anchor(nGh, nGw, anchors) 
    anchor_mesh = tf.transpose(anchor_mesh, (0,2,3,1))              # Shpae (nA x nGh x nGw) x 4
    anchor_mesh = tf.tile(anchor_mesh[tf.newaxis],(nB,1,1,1,1))
    pred_list = decode_delta(tf.reshape(delta_map,(-1,4)), tf.reshape(anchor_mesh,(-1,4)))
    pred_map = tf.reshape(pred_list,(nB, nA, nGh, nGw, 4))
    return pred_map

def check_proposals_tensor(proposal):
    # indices = tf.squeeze(tf.where(tf.greater(proposal[..., 4],cfg.CONF_THRESH)),axis=1)
    # proposal = tf.gather(proposal,indices)

    # padding = tf.maximum(cfg.MAX_PROP-tf.shape(proposal)[0], 0)
    # proposal = tf.pad(proposal,paddings=[[0,padding],[0,0]], mode='CONSTANT', constant_values=0.0)
    
    # remove unconsistent bboxes; x2>x1 and y2>y1
    width = proposal[...,2] - proposal[...,0]
    height = proposal[...,3] - proposal[...,1]
    
    mask_dim = tf.logical_and(tf.greater(width, cfg.MIN_BOX_DIM), tf.greater(height, cfg.MIN_BOX_DIM))
    mask_ratio = tf.logical_and(tf.greater(width/height, cfg.MIN_BOX_RATIO),\
        tf.greater(height/width, cfg.MIN_BOX_RATIO))
    mask = tf.logical_and(mask_dim,mask_ratio)
    
    indices = tf.tile(tf.cast(mask, tf.float32)[...,None],(1,1,5))
    proposal = proposal * indices
    
    proposal = best_sort_batch(proposal)
    # padding = tf.maximum(cfg.MAX_PROP-tf.shape(proposal)[0], 0)
    # proposal = tf.pad(proposal,paddings=[[0,padding],[0,0]], mode='CONSTANT', constant_values=0.0)
    

    
    proposal = tf.gather(proposal,tf.range(cfg.MAX_PROP), axis=1) # automatic zero padding

    # padding = tf.maximum(cfg.MAX_PROP-tf.shape(proposal)[0], 0) # needed if cfg.MAX_PROP is high value
    # proposal = tf.pad(proposal,paddings=[[0,padding],[0,0]], mode='CONSTANT', constant_values=0.0) # useless
#    mask_non_zero_entry = tf.cast(tf.not_equal(tf.reduce_sum(proposal[...,:4],axis=-1),0.0)[...,tf.newaxis],tf.float32)
#    proposal = entry_stop_gradients(proposal, mask_non_zero_entry)
    return proposal

def check_proposals(proposal):
    # indices = tf.squeeze(tf.where(tf.greater(proposal[..., 4],cfg.CONF_THRESH)),axis=1)
    # proposal = tf.gather(proposal,indices)

    # padding = tf.maximum(cfg.MAX_PROP-tf.shape(proposal)[0], 0)
    # proposal = tf.pad(proposal,paddings=[[0,padding],[0,0]], mode='CONSTANT', constant_values=0.0)
    
    # remove unconsistent bboxes; x2>x1 and y2>y1
    width = proposal[...,2] - proposal[...,0]
    height = proposal[...,3] - proposal[...,1]
    
    mask_dim = tf.logical_and(tf.greater(width, cfg.MIN_BOX_DIM), tf.greater(height, cfg.MIN_BOX_DIM))
    mask_ratio = tf.logical_and(tf.greater(width/height, cfg.MIN_BOX_RATIO),\
        tf.greater(height/width, cfg.MIN_BOX_RATIO))
    mask = tf.logical_and(mask_dim,mask_ratio)
    
    indices = tf.squeeze(tf.where(mask),axis=-1)
    proposal = tf.gather(proposal,indices)

    # padding = tf.maximum(cfg.MAX_PROP-tf.shape(proposal)[0], 0)
    # proposal = tf.pad(proposal,paddings=[[0,padding],[0,0]], mode='CONSTANT', constant_values=0.0)
    
    indices = tf.argsort(proposal[..., 4], axis=-1, direction='DESCENDING')
    proposal = tf.gather(proposal,indices)
    
    proposal = tf.gather(proposal,tf.range(cfg.MAX_PROP)) # automatic zero padding
    
    # padding = tf.maximum(cfg.MAX_PROP-tf.shape(proposal)[0], 0) # needed if cfg.MAX_PROP is high value
    # proposal = tf.pad(proposal,paddings=[[0,padding],[0,0]], mode='CONSTANT', constant_values=0.0) # useless
#    mask_non_zero_entry = tf.cast(tf.not_equal(tf.reduce_sum(proposal[...,:4],axis=-1),0.0)[...,tf.newaxis],tf.float32)
#    proposal = entry_stop_gradients(proposal, mask_non_zero_entry)
    return proposal


def nms_proposals(proposal):
    # non max suppression
    indices = tf.image.non_max_suppression(proposal[...,:4], proposal[...,4], max_output_size=cfg.MAX_PROP, 
                         iou_threshold=cfg.NMS_THRESH) # score_threshold=cfg.CONF_THRESH
    proposal = tf.gather(proposal, indices) #b x n rois x (4+1+1+208)
    
    proposal = tf.gather(proposal,tf.range(cfg.MAX_PROP)) # automatic zero padding

    # padding = tf.maximum(cfg.MAX_PROP-tf.shape(proposal)[0], 0)
    # proposal = tf.pad(proposal,paddings=[[0,padding],[0,0]], mode='CONSTANT', constant_values=0.0)
        
    return proposal

def preprocess_target_bbox(proposal_gt_bbox):
    """ proposal: [num_rois, (x, y, w, h)] 
        gt_bbox: [num_gt_bbox, (x, y, w, h)] """
    proposal, gt_bbox, gt_indices = proposal_gt_bbox
    non_zero_proposals = tf.not_equal(tf.reduce_sum(proposal,axis=-1),0.0)
    valid_indices = tf.squeeze(tf.where(non_zero_proposals),axis=-1)
    valid_proposals = tf.gather(proposal, valid_indices, axis=0)
    assigned_gt_bboxes = tf.gather(gt_bbox,gt_indices,axis=0)
    valid_gt_bboxes = tf.gather(assigned_gt_bboxes,valid_indices, axis=0)
    encoded_delta = encode_delta(valid_gt_bboxes, valid_proposals)
    non_valid_indices = tf.squeeze(tf.where(tf.logical_not(non_zero_proposals)),axis=-1)
    non_valid_gt_bboxes = tf.gather(proposal,non_valid_indices)
    target_bbox = tf.concat([encoded_delta, non_valid_gt_bboxes], axis=0)

    return target_bbox


def preprocess_target_mask(proposal_gt_mask):
    """ proposal: [num_rois, (x, y, w, h)] 
        gt_mask: [num_gt_bbox, mask_dim, mask_dim] """
    proposal, gt_mask, gt_indices = proposal_gt_mask
    non_zero_proposals = tf.not_equal(tf.reduce_sum(proposal,axis=-1),0.0)
    valid_indices = tf.squeeze(tf.where(non_zero_proposals),axis=-1)
    non_valid_indices = tf.squeeze(tf.where(tf.logical_not(non_zero_proposals)),axis=-1)
    assigned_gt_masks = tf.gather(gt_mask,gt_indices,axis=0)
    valid_gt_masks = tf.gather(assigned_gt_masks, valid_indices, axis=0)
    non_valid_gt_masks = tf.gather(tf.zeros_like(assigned_gt_masks),non_valid_indices)
    target_mask = tf.concat([valid_gt_masks, non_valid_gt_masks], axis=0) #zero padded

    return target_mask


def preprocess_target_indices(proposal_gt_mask):
    """ proposal: [num_rois, (x, y, w, h)] 
        gt_bbox: [num_gt_bbox, (x, y, w, h)] """
    proposal, gt_bbox = proposal_gt_mask
    gt_intersect = bbox_iou(proposal,gt_bbox)
    target_indices = tf.math.argmax(gt_intersect,axis=-1)
    # Determine positive and negative ROIs
    valid_indices = tf.reduce_max(gt_intersect, axis=1)
    target_class_ids = tf.cast(valid_indices >= cfg.IOU_THRESH, tf.int64) # in this dataset if there is bbox, it's human, 1 class
    return target_indices, target_class_ids 


def preprocess_mrcnn(proposals, gt_bboxes, gt_masks):
    """ target_bbox: [batch, num_rois, (dx, dy, log(dw), log(dh))]
     target_class_ids: [batch, num_rois]. Integer class IDs.
     target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. """
#    proposals_1 = tf.random.uniform((2,10,2))*0.5
#    proposals_2 = tf.random.uniform((2,10,2))*0.5 + 0.5
#    proposals = tf.concat([proposals_1, proposals_2], axis=-1)
#    bad_proposals = tf.zeros((2,10,4))
#    proposals = tf.concat([proposals, bad_proposals], axis=1)
#    gt_bboxes_1 = tf.random.uniform((2,5,2))*0.5
#    gt_bboxes_2 = tf.random.uniform((2,5,2))*0.5 + 0.5
#    gt_bboxes = tf.concat([gt_bboxes_1, gt_bboxes_2], axis=-1)* cfg.TRAIN_SIZE
#    gt_masks = tf.round(tf.random.uniform((2,5,28,28)))
#    proposals = tf.zeros((2,20,4))
#    gt_bboxes = tf.zeros((2,10,4))
#    gt_masks = tf.zeros((2,10,28,28))
    gt_bboxes = gt_bboxes[...,:4]
    gt_bboxes /= cfg.TRAIN_SIZE
    gt_bboxes = tf.clip_by_value(gt_bboxes, 0.0, 1.0) # redundant
    gt_bboxes = xyxy2xywh(gt_bboxes)
    proposals = tf.stop_gradient(proposals)
    proposals = proposals[...,:4]
    proposals = tf.clip_by_value(proposals, 0.0, 1.0) # redundant
    proposals = xyxy2xywh(proposals)  
    target_indices, target_class_ids = tf.map_fn(preprocess_target_indices, (proposals, gt_bboxes), fn_output_signature=(tf.int64,tf.int64))
    target_bbox = tf.map_fn(preprocess_target_bbox, (proposals, gt_bboxes, target_indices), fn_output_signature=tf.float32)
    target_masks = tf.map_fn(preprocess_target_mask, (proposals, gt_masks, target_indices), fn_output_signature=tf.float32)
    target_class_ids = tf.stop_gradient(target_class_ids)
    target_bbox = tf.stop_gradient(target_bbox)
    target_masks = tf.stop_gradient(target_masks)
#    pred_class_logits = -tf.stack([tf.concat([-tf.ones((2,10)),-tf.ones((2,10))],axis=-1),tf.concat([tf.ones((2,10)),tf.ones((2,10))],axis=-1)],axis=-1)*10
#    pred_bbox = tf.tile(target_bbox[:,:,None,:],(1,1,2,1))
#    pred_masks = tf.tile(target_masks[...,None],(1,1,1,1,2))
#    active_class_ids = tf.stop_gradient(active_class_ids)
    #decode_delta(encoded_delta, valid_proposals)

    return target_class_ids, target_bbox, target_masks #, active_class_ids


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.
    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
        active_class_ids = tf.concat([tf.zeros((nB,1)),tf.ones((nB,1))],axis=-1)
    """

    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
#    target_class_ids = tf.cast(target_class_ids, 'int32')

    # Find predictions of classes that are not in the dataset.
#    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    
#    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
#    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_mean(loss) #/ tf.reduce_sum(pred_active)
    return loss


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.
    target_bbox: [batch, num_rois, (dx, dy, log(dw), log(dh))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dx, dy, log(dw), log(dh))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))
    pred_bbox = tf.reshape(pred_bbox, (-1, tf.shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)
    # Smooth-L1 Loss
    if tf.size(target_bbox) > 0:
        loss = smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox)
    else:
        loss = tf.constant(0.0)
    loss = tf.reduce_mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.
    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = tf.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    if tf.size(y_true) > 0:
        loss =tf.keras.losses.binary_crossentropy(y_true, y_pred)
    else:
        loss = tf.constant(0.0)

    loss = tf.reduce_mean(loss)
    return loss


def entry_stop_gradients(target, mask):
    mask_h = tf.abs(mask-1)
    return tf.stop_gradient(mask_h * target) + mask * target
