#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 18:40:02 2020

@author: fiorapirri
"""


import json
import numpy as np
import config as cfg
import tensorflow as tf
from compute_ap import compute_ap_range
import skimage.transform
from PIL import Image, ImageDraw, ImageFont, ImageColor

###############################################################################################
####################################   PREPROCESSING   ########################################
###############################################################################################

def filter_inputs(image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes):
    return tf.greater(tf.reduce_sum(gt_bboxes[...,:4]), 0) and tf.greater(tf.reduce_sum(gt_masks), 0)

def file_reader(file_name):
    with open(file_name) as json_file:
        return json.load(json_file)

def file_writer(file, file_name):
    with open(file_name, 'w') as f:
        f.write(json.dumps(file)) 

def mask_clamp(mask):
    if np.any(mask>1): # CDCL RETURN VALUES FROM 0 TO 7 FOR 0 - BACKGROUND, 1 - BOXES, 2 - HEADS, ARMS, LEGS..
        mask[mask==1]=0 # SET BOX LABEL TO 0
    mask = np.clip(mask,0,1)
    mask = np.round(mask)
    return np.array(mask,dtype=np.float32)

def read_image(img_path):
    return Image.open(img_path)

def data_check( masks, mini_masks, bboxes):#check consistency of bbox after data augmentation: dimension and ratio
    width = bboxes[...,2] - bboxes[...,0]
    height = bboxes[...,3] - bboxes[...,1]
    mask = (width > cfg.MIN_BOX_DIM*cfg.TRAIN_SIZE) \
        * (height > cfg.MIN_BOX_DIM*cfg.TRAIN_SIZE) \
        * ((width/height)>cfg.MIN_BOX_RATIO) \
        * ((height/width)>cfg.MIN_BOX_RATIO)
    bboxes = bboxes[mask]
    masks = masks[mask]
    mini_masks = mini_masks[mask]
    return masks, mini_masks, bboxes

def data_pad( masks, mini_masks, bboxes):
    bboxes_padded = np.zeros(( cfg.MAX_INSTANCES,5))
    bboxes_padded[:,4]=-1
    masks_padded = np.zeros(( cfg.MAX_INSTANCES,masks.shape[1],masks.shape[2]))
    mini_masks_padded = np.zeros(( cfg.MAX_INSTANCES,mini_masks.shape[1],mini_masks.shape[2]))
    min_bbox = min(bboxes.shape[0],  cfg.MAX_INSTANCES)
    bboxes_padded[:min_bbox,...]=bboxes[:min_bbox,...]
    masks_padded[:min_bbox,...]=masks[:min_bbox,...]
    mini_masks_padded[:min_bbox,...]=mini_masks[:min_bbox,...]
    return masks_padded, mini_masks_padded, bboxes_padded
    
def mini_masks_generator( masks, bboxes):
    masks_resized = []
    for mask, bbox in zip(masks, bboxes):
        try:
            mask = Image.fromarray(mask[bbox[1]:bbox[3],bbox[0]:bbox[2]])
        except:
            mask = Image.fromarray(mask)
        mask = mask.resize((cfg.MASK_SIZE,cfg.MASK_SIZE), Image.ANTIALIAS)
        mask = np.clip(mask,0,1)
        mask = np.round(mask)
        masks_resized.append(mask)

    masks_resized = np.stack(masks_resized,axis=0)
    return masks_resized
    
def data_preprocess( image, gt_boxes, masks):
    ih, iw    = (cfg.TRAIN_SIZE, cfg.TRAIN_SIZE)
    h,  w, _  = image.shape
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)    
    image = Image.fromarray(image)
    image_resized = image.resize((nw, nh),Image.ANTIALIAS)
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.
    image_paded=np.clip(image_paded,0,1)
    gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
    gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
    masks_padded = np.zeros((masks.shape[0],iw, ih))
    for i, maskk in enumerate(masks):
        mask = Image.fromarray(maskk)
        mask_resized = np.round(mask.resize((nw, nh),Image.ANTIALIAS))
        masks_padded[i, dh:nh+dh, dw:nw+dw] = mask_resized
    gt_boxes = np.array(gt_boxes,dtype=np.uint32)
    gt_boxes = np.clip(gt_boxes,0,cfg.TRAIN_SIZE-1)
    return image_paded, masks_padded, gt_boxes

def data_augment( image, masks, bboxes):
    image, masks, bboxes = random_horizontal_flip(image, masks, bboxes)
    image, masks, bboxes = random_crop(image, masks, bboxes)
    image, masks, bboxes = random_translate(image, masks, bboxes)
    return image, masks, bboxes
    
def random_horizontal_flip( image, masks, bboxes):
    if np.random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        masks = masks[:,:,::-1]
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
    return image, masks, bboxes

def random_crop( image, masks, bboxes):
    if np.random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate(
            [
                np.min(bboxes[:, 0:2], axis=0),
                np.max(bboxes[:, 2:4], axis=0),
            ],
            axis=-1,
        )
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]
        crop_xmin = max(
            0, int(max_bbox[0] - np.random.uniform(0, max_l_trans))
        )
        crop_ymin = max(
            0, int(max_bbox[1] - np.random.uniform(0, max_u_trans))
        )
        crop_xmax = max(
            w, int(max_bbox[2] + np.random.uniform(0, max_r_trans))
        )
        crop_ymax = max(
            h, int(max_bbox[3] + np.random.uniform(0, max_d_trans))
        )
        image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        masks = masks[:,crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        bboxes[:, [0, 2]] -= crop_xmin
        bboxes[:, [1, 3]] -= crop_ymin

    return image, masks, bboxes

def random_translate( image, masks, bboxes):
    if np.random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate(
            [
                np.min(bboxes[:, 0:2], axis=0),
                np.max(bboxes[:, 2:4], axis=0),
            ],
            axis=-1,
        )
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]
        tx = int(np.random.uniform(-(max_l_trans - 1), (max_r_trans - 1)))
        ty = int(np.random.uniform(-(max_u_trans - 1), (max_d_trans - 1)))
        old_image = np.copy(image)
        old_masks = np.copy(masks)
        image = np.zeros_like(image)
        masks = np.zeros_like(masks)
        image[max(0,ty):min(h,h+ty),max(0,tx):min(w,w+tx),:] = old_image[max(0,-ty):min(h,h-ty),max(0,-tx):min(w,w-tx),:]
        masks[:,max(0,ty):min(h,h+ty),max(0,tx):min(w,w+tx)] = old_masks[:,max(0,-ty):min(h,h-ty),max(0,-tx):min(w,w-tx)]
        bboxes[:, [0, 2]] += tx
        bboxes[:, [1, 3]] += ty
        
    return image, masks, bboxes

###############################################################################################
####################################   OUTPUT DECODING   ######################################
###############################################################################################

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
    if len(prediction)>3: # if masks are present TODO
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
    
def draw_bbox(image, bboxs=None, masks=None, conf_id=None, mode='return'):
    colors = []
    for name, code in ImageColor.colormap.items():
         colors.append(name)
#    np.random.shuffle(colors)
    if np.any(bboxs is not None):
        bboxs = bboxs[...,:4]
    img = Image.fromarray(np.array(image*255,dtype=np.uint8))                   
    draw = ImageDraw.Draw(img)   
    if np.any(conf_id is not None):
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
            xyxy = np.array(bbox,dtype=np.int32)
            if np.any(xyxy>0):
                mask = unmold_mask(mask, xyxy, img.shape)
                mask = np.array(Image.new('RGB', (img.shape[0], img.shape[1]), color = colors[i%len(colors)]))*np.tile(mask[...,None],(1,1,3))
                img[mask>0] = img[mask>0]*0.5+mask[mask>0]*0.5
    if mode == 'PIL':
        show_image(img)
    else:
        return img

def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    x1, y1, x2, y2 = bbox
    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)
    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask

def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().
    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    return skimage.transform.resize(
        image, output_shape,
        order=order, mode=mode, cval=cval, clip=clip,
        preserve_range=preserve_range, anti_aliasing=anti_aliasing,
        anti_aliasing_sigma=anti_aliasing_sigma)
        
def show_image(img):
    img = Image.fromarray(img)               
    img.show()

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

###############################################################################################
######################################   BBOXES UTILS   #######################################
###############################################################################################

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


###############################################################################################
####################################   YOLO LABEL UTILS   #####################################
###############################################################################################

def encode_labels(data):
    bboxes, mask = data
    ANCHORS = tf.reshape(tf.constant(cfg.ANCHORS,dtype=np.int32),[cfg.LEVELS, cfg.NUM_ANCHORS, 2])
    label_2 = encode_label(bboxes, mask, ANCHORS[0]/cfg.STRIDES[0], cfg.NUM_ANCHORS, cfg.NUM_CLASS, cfg.TRAIN_SIZE//cfg.STRIDES[0], cfg.TRAIN_SIZE//cfg.STRIDES[0])
    label_3 = encode_label(bboxes, mask, ANCHORS[1]/cfg.STRIDES[1], cfg.NUM_ANCHORS, cfg.NUM_CLASS, cfg.TRAIN_SIZE//cfg.STRIDES[1], cfg.TRAIN_SIZE//cfg.STRIDES[1])
    label_4 = encode_label(bboxes, mask, ANCHORS[2]/cfg.STRIDES[2], cfg.NUM_ANCHORS, cfg.NUM_CLASS, cfg.TRAIN_SIZE//cfg.STRIDES[2], cfg.TRAIN_SIZE//cfg.STRIDES[2])
    label_5 = encode_label(bboxes, mask, ANCHORS[3]/cfg.STRIDES[3], cfg.NUM_ANCHORS, cfg.NUM_CLASS, cfg.TRAIN_SIZE//cfg.STRIDES[3], cfg.TRAIN_SIZE//cfg.STRIDES[3])
    return label_2, label_3, label_4, label_5

# encode label using masks
def encode_label(target, masks, anchor_wh, nA, nC, nGh, nGw):
        
    masks.set_shape([cfg.MAX_INSTANCES, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE])
    masks = tf.image.resize(masks[...,None], (nGh,nGw), method=tf.image.ResizeMethod.BILINEAR, \
                            preserve_aspect_ratio=True, antialias=True)[...,0]

    masks = tf.round(masks)
    masks = tf.clip_by_value(masks,0,1)
    masks = tf.transpose(masks, (0,2,1))
    masks = tf.tile(tf.cast(masks, tf.bool)[None],(nA,1,1,1))

    target = tf.cast(target, tf.float32)
    bbox = target[:,:4]/cfg.TRAIN_SIZE
    ids = target[:,4]
    
    gt_boxes = tf.clip_by_value(bbox, 0.0, 1.0)
    gt_boxes = xyxy2xywh(bbox)
    gt_boxes = gt_boxes * tf.cast([nGw, nGh, nGw, nGh], tf.float32)
    anchor_wh = tf.cast(anchor_wh, tf.float32)
    tbox = tf.zeros((nA, nGh, nGw, 4))
    tconf = tf.zeros(( nA, nGh, nGw))
    tid = tf.fill((nA, nGh, nGw, nC),-1.0)
    
    anchor_mesh = generate_anchor(nGh, nGw, anchor_wh)# Shape nA x 4 x nGh x nGw
    anchor_mesh = tf.transpose(anchor_mesh, (0,2,3,1))# Shpae nA x nGh x nGw x 4 
    anchor_list = tf.reshape(anchor_mesh,(-1, 4))# Shpae (nA x nGh x nGw) x 4 
    iou_pdist = bbox_iou(anchor_list, gt_boxes)# Shape (nA x nGh x nGw) x Ng
    iou_max = tf.math.reduce_max(iou_pdist, axis=1)# Shape (nA x nGh x nGw), both
    max_gt_index = tf.math.argmax(iou_pdist, axis=1)# Shape (nA x nGh x nGw), both
    
    iou_map = tf.reshape(iou_max, (nA, nGh, nGw))     
    gt_index_map = tf.reshape(max_gt_index,(nA, nGh, nGw))     
    
    iou_map = tf.tile(iou_map[:,None], (1,cfg.MAX_INSTANCES,1,1))
    
    id_index = iou_map > cfg.ID_THRESH
    id_index = tf.where(tf.cast(tf.tile(tf.reduce_max(tf.cast(id_index,tf.float32)*tf.cast(masks,tf.float32),axis=[2,3])[...,None,None],(1,1,nGh,nGw)),tf.bool),masks,False)
    id_index = tf.cast(tf.reduce_max(tf.cast(id_index,tf.float32),axis=1),tf.bool)
    fg_index = id_index
    bg_index = tf.logical_not(fg_index)

    tconf = tf.where(fg_index, 1.0, tconf)
    tconf = tf.where(bg_index, 0.0, tconf)

    gt_index = tf.boolean_mask(gt_index_map,fg_index)
    gt_box_list = tf.gather(gt_boxes,gt_index)
    gt_id_list = tf.gather(ids,tf.boolean_mask(gt_index_map,id_index))

    cond = tf.greater(tf.reduce_sum(tf.cast(fg_index,tf.float32)), 0)
    
    tid = tf.where(id_index[...,None], tf.scatter_nd(tf.where(id_index),  gt_id_list[:,None], (nA, nGh, nGw, nC)), tid)

    fg_anchor_list = anchor_mesh[fg_index] 
    delta_target = encode_delta(gt_box_list, fg_anchor_list)
    tbox = tf.cond( cond, lambda: tf.scatter_nd(tf.where(fg_index),  delta_target, (nA, nGh, nGw, 4)), lambda: tbox)
        
    label = tf.concat([tbox,tconf[...,None],tid],axis=-1)
    # need to transpose since for some reason the labels are rotated, maybe scatter_nd?
    label = tf.transpose(label,(0,2,1,3))
    return label

# original labeling code, no mask is provided
#def encode_label(target, anchor_wh, nA, nC, nGh, nGw):
#    target = tf.cast(target, tf.float32)
#    bbox = target[:,:4]/cfg.TRAIN_SIZE
#    ids = target[:,4]
#    
#    gt_boxes = tf.clip_by_value(bbox, 0.0, 1.0)
#    gt_boxes = xyxy2xywh(bbox)
#    gt_boxes = gt_boxes * tf.cast([nGw, nGh, nGw, nGh], tf.float32)
#    anchor_wh = tf.cast(anchor_wh, tf.float32)
#    tbox = tf.zeros((nA, nGh, nGw, 4))
#    tconf = tf.zeros(( nA, nGh, nGw))
#    tid = tf.zeros((nA, nGh, nGw, nC))-1
#    
#    anchor_mesh = generate_anchor(nGh, nGw, anchor_wh)# Shape nA x 4 x nGh x nGw
#    anchor_mesh = tf.transpose(anchor_mesh, (0,2,3,1))# Shpae nA x nGh x nGw x 4 
#    anchor_list = tf.reshape(anchor_mesh,(-1, 4))# Shpae (nA x nGh x nGw) x 4 
#    iou_pdist = bbox_iou(anchor_list, gt_boxes)# Shape (nA x nGh x nGw) x Ng
#    iou_max = tf.math.reduce_max(iou_pdist, axis=1)# Shape (nA x nGh x nGw), both
#    max_gt_index = tf.math.argmax(iou_pdist, axis=1)# Shape (nA x nGh x nGw), both
#    
#    iou_map = tf.reshape(iou_max, (nA, nGh, nGw))     
#    gt_index_map = tf.reshape(max_gt_index,(nA, nGh, nGw))     
#    
#    id_index = iou_map > cfg.ID_THRESH
#    fg_index = iou_map > cfg.FG_THRESH                                                    
#    bg_index = iou_map < cfg.BG_THRESH 
#    ign_index = tf.cast(tf.cast((iou_map < cfg.FG_THRESH),tf.float32) \
#                        * tf.cast((iou_map > cfg.BG_THRESH),tf.float32),tf.bool)
#    tconf = tf.where(fg_index, 1.0, tconf)
#    tconf = tf.where(bg_index, 0.0, tconf)
#    tconf = tf.where(ign_index, -1.0, tconf)
#
#    gt_index = tf.boolean_mask(gt_index_map,fg_index)
#    gt_box_list = tf.gather(gt_boxes,gt_index)
#    gt_id_list = tf.gather(ids,tf.boolean_mask(gt_index_map,id_index))
#
#    cond = tf.greater(tf.reduce_sum(tf.cast(fg_index,tf.float32)), 0)
#    
#    tid = tf.cond(cond, lambda: tf.scatter_nd(tf.where(id_index),  gt_id_list[:,None], (nA, nGh, nGw, nC)), lambda: tid)
#    tid = tf.cond( cond, lambda: tf.where(tf.equal(tid,0.0),  -1.0, tid), lambda: tid)
#    fg_anchor_list = anchor_mesh[fg_index] 
#    delta_target = encode_delta(gt_box_list, fg_anchor_list)
#    tbox = tf.cond( cond, lambda: tf.scatter_nd(tf.where(fg_index),  delta_target, (nA, nGh, nGw, 4)), lambda: tbox)
#        
#    label = tf.concat([tbox,tconf[...,None],tid],axis=-1)
#    # need to transpose since for some reason the labels are rotated, maybe scatter_nd?
#    label = tf.transpose(label,(0,2,1,3))
#    return label

def decode_label(label, anchors, stride):
    label = tf.transpose(label,(0,1,3,2,4)) # decode_delta_map has some issue
    pconf = label[..., 4]  # Conf
    pbox = label[..., :4]
    pbox = decode_delta_map(pbox, tf.divide(anchors,stride))
    pbox = tf.multiply(pbox,stride) # now in range [0, .... cfg.TRAIN_SIZE]
    pbox = tf.divide(pbox,cfg.TRAIN_SIZE) #now normalized in [0...1]
    pbox = xywh2xyxy(pbox) # to bbox
    pbox = tf.clip_by_value(pbox,0.0,1.0) # clip to avoid nan
    proposal = tf.concat([pbox, pconf[...,tf.newaxis]], axis=-1) 
    nB = tf.shape(proposal)[0]
    _, nA, nGh, nGw, nC = proposal.shape       
    proposal = tf.reshape(proposal, [nB, tf.multiply(nA, tf.multiply(nGh , nGw)), nC]) # b x nBB x (4 + 1 + 1 + 208) rois
    proposal = check_proposals_tensor(proposal)
    return proposal

def decode_labels(predictions, embeddings = None):
    ANCHORS = tf.reshape(tf.constant(cfg.ANCHORS,dtype=tf.float32),[cfg.LEVELS, cfg.NUM_ANCHORS, 2])
    p_2, p_3, p_4, p_5 = predictions      
    d_2 = decode_label(p_2, ANCHORS[0], cfg.STRIDES[0])    
    d_3 = decode_label(p_3, ANCHORS[1], cfg.STRIDES[1])    
    d_4 = decode_label(p_4, ANCHORS[2], cfg.STRIDES[2])    
    d_5 = decode_label(p_5, ANCHORS[3], cfg.STRIDES[3])
    
    proposals = tf.concat([d_2,d_3,d_4,d_5],axis=1) #concat along levels    
    proposals = nms_proposals_tensor(proposals)    
    
    proposals = proposals * tf.tile(proposals[...,4][...,None],(1,1,tf.shape(proposals)[-1]))
    return proposals

def generate_anchor(nGh, nGw, anchor_wh):
    nA = tf.shape(anchor_wh)[0]
    yy, xx =tf.meshgrid(tf.range(nGh), tf.range(nGw))
    mesh = tf.stack([xx, yy], axis=0)                                              # Shape 2, nGh, nGw
    mesh = tf.cast(tf.tile(mesh[tf.newaxis],(nA,1,1,1)),tf.float32)                          # Shape nA x 2 x nGh x nGw
    anchor_offset_mesh = tf.tile(anchor_wh[...,tf.newaxis][...,tf.newaxis],(1, 1, nGh,nGw))  # Shape nA x 2 x nGh x nGw
    anchor_mesh = tf.concat([mesh, anchor_offset_mesh], axis=1)                       # Shape nA x 4 x nGh x nGw
    return anchor_mesh

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
    
def decode_prediction(prediction, anchors, stride):
    prediction = tf.transpose(prediction,(0,1,3,2,4)) # decode_delta_map has some issue
    pconf = prediction[..., 4:6]  # Conf
    pconf = tf.nn.softmax(pconf, axis=-1)[...,1] # 1 is foreground
    pbox = prediction[..., :4]
    pbox = decode_delta_map(pbox, tf.divide(anchors,stride))
    pbox = tf.multiply(pbox,stride) # now in range [0, .... cfg.TRAIN_SIZE]
    pbox = tf.divide(pbox,cfg.TRAIN_SIZE) #now normalized in [0...1]
    pbox = xywh2xyxy(pbox) # to bbox
    pbox = tf.clip_by_value(pbox,0.0,1.0) # clip to avoid nan
    proposal = tf.concat([pbox, pconf[...,tf.newaxis]], axis=-1) 
    nB = tf.shape(proposal)[0]
    _, nA, nGh, nGw, nC = proposal.shape       
    proposal = tf.reshape(proposal, [nB, tf.multiply(nA, tf.multiply(nGh , nGw)), nC]) # b x nBB x (4 + 1 + 1 + 208) rois
    proposal = check_proposals_tensor(proposal)
    return proposal


###############################################################################################
#################################   PROPOSALS PROCESSING   ####################################
###############################################################################################

def entry_stop_gradients(target, mask):
    mask_h = tf.abs(mask-1)
    return tf.stop_gradient(mask_h * target) + mask * target

def check_proposals_tensor(proposal):
    """This function remove unconsistent bboxes: x2>x1 and y2>y1 or with bad ratio 
         and sort the proposals by score without iterating over the batch """
    
    nC = tf.shape(proposal)[2]
    width = proposal[...,2] - proposal[...,0]
    height = proposal[...,3] - proposal[...,1]
    
    mask_dim = tf.logical_and(tf.greater(width, cfg.MIN_BOX_DIM), tf.greater(height, cfg.MIN_BOX_DIM))
    mask_ratio = tf.logical_and(tf.greater(tf.divide(width,height), cfg.MIN_BOX_RATIO),\
        tf.greater(tf.divide(height,width), cfg.MIN_BOX_RATIO))
    mask = tf.logical_and(mask_dim,mask_ratio)
    
    indices = tf.tile(tf.cast(mask, tf.float32)[...,tf.newaxis],(1,1,nC))
    proposal = tf.multiply(proposal, indices) # remove the bad proposals
    
    indices = tf.argsort(proposal[..., 4], axis=-1, direction='DESCENDING', stable=True)

    proposal = tf.gather(proposal,indices, axis=1, batch_dims=1) 
    k_proposal = tf.range(cfg.PRE_NMS_LIMIT)
    k_proposal = tf.stop_gradient(k_proposal)
    proposal = tf.gather(proposal, k_proposal, axis=1) # automatic zero padding
    return proposal

def check_proposals(proposal):
    # remove unconsistent bboxes; x2>x1 and y2>y1
    width = proposal[...,2] - proposal[...,0]
    height = proposal[...,3] - proposal[...,1]
    
    mask_dim = tf.logical_and(tf.greater(width, cfg.MIN_BOX_DIM), tf.greater(height, cfg.MIN_BOX_DIM))
    mask_ratio = tf.logical_and(tf.greater(width/height, cfg.MIN_BOX_RATIO),\
        tf.greater(height/width, cfg.MIN_BOX_RATIO))
    mask = tf.logical_and(mask_dim,mask_ratio)
    
    indices = tf.squeeze(tf.where(mask),axis=-1)
    proposal = tf.gather(proposal,indices, axis=0)

    # padding = tf.maximum(cfg.MAX_PROP-tf.shape(proposal)[0], 0)
    # proposal = tf.pad(proposal,paddings=[[0,padding],[0,0]], mode='CONSTANT', constant_values=0.0)
    
    indices = tf.argsort(proposal[..., 4], axis=-1, direction='DESCENDING')
    proposal = tf.gather(proposal,indices, axis=0)
    
    proposal = tf.gather(proposal,tf.range(cfg.PRE_NMS_LIMIT), axis=0) # automatic zero padding only on GPU

    return proposal
    
def nms_proposals_tensor(proposal):
    """This function compute nms proposals without iterating over the batch """
    proposal, conf, *_ = tf.image.combined_non_max_suppression(
            proposal[...,:4][:,:,tf.newaxis,:], proposal[...,4][...,tf.newaxis], max_output_size_per_class=cfg.MAX_PROP, \
            max_total_size=cfg.MAX_PROP, iou_threshold=cfg.NMS_THRESH,pad_per_class=True, clip_boxes=True)

    proposal = tf.concat([proposal,conf[...,tf.newaxis]], axis=-1)    
    return proposal

def nms_proposals(proposal):
    # non max suppression
    indices = tf.image.non_max_suppression(proposal[...,:4], proposal[...,4], max_output_size=cfg.MAX_PROP, 
                         iou_threshold=cfg.NMS_THRESH) # score_threshold=cfg.CONF_THRESH
    proposal = tf.gather(proposal, indices) #b x n rois x (4+1+1+208)
    
    proposal = tf.gather(proposal,tf.range(cfg.MAX_PROP)) # automatic zero padding

    return proposal

###############################################################################################
#################################   MASK RCNN PROCESSING   ####################################
###############################################################################################

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

    return target_class_ids, target_bbox, target_masks
