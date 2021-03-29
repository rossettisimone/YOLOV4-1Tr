#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 18:40:02 2020

@author: fiorapirri
"""

import os
import json
import numpy as np
import config as cfg
import tensorflow as tf
from compute_ap import compute_ap_range
import skimage.transform
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageColor
from skimage.color import hsv2rgb, rgb2hsv
###############################################################################################
####################################   PREPROCESSING   ########################################
###############################################################################################

def rle_decoding( rle_arr, w, h):
    indices = []
    temp_idx = 0
    for idx, cnt in zip(rle_arr[0::2], rle_arr[1::2]):
        temp_idx += idx
        indices.extend(list(range(temp_idx, temp_idx+cnt-1)))  # RLE is 1-based index
        temp_idx += cnt
    mask = np.zeros(h*w, dtype=np.uint8)
    mask[indices] = 1
    return mask.reshape((w, h)).T

def folders():
    folder = "{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.mkdir(folder)
    logdir = os.path.join(folder, cfg.LOGDIR)
    os.mkdir(logdir)
    # save config file
    with open('config.py', mode='r') as in_file, open('{}/config.py'.format(folder), mode='w') as out_file:
        out_file.write(in_file.read())
    filepath = os.path.join(folder, cfg.WEIGHTS)
    os.mkdir(filepath)
    filepath = os.path.join(filepath,'model.{epoch:02d}-{val_alb_loss:.3f}.h5')
    return logdir, filepath
                  
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

def data_check( masks, bboxes):#check consistency of bbox after data augmentation: dimension and ratio
    width = bboxes[...,2] - bboxes[...,0]
    height = bboxes[...,3] - bboxes[...,1]
    mask = (width > cfg.MIN_BOX_DIM*cfg.TRAIN_SIZE) \
        * (height > cfg.MIN_BOX_DIM*cfg.TRAIN_SIZE) \
        * ((width/height)>cfg.MIN_BOX_RATIO) \
        * ((height/width)>cfg.MIN_BOX_RATIO)
    bboxes = bboxes[mask]
    masks = masks[mask]
    return masks, bboxes

def data_pad( masks, bboxes):
    bboxes_padded = np.zeros(( cfg.MAX_INSTANCES,bboxes.shape[1]))
    masks_padded = np.zeros(( cfg.MAX_INSTANCES,masks.shape[1],masks.shape[2]))
    min_bbox = min(bboxes.shape[0],  cfg.MAX_INSTANCES)
    bboxes_padded[:min_bbox,...]=bboxes[:min_bbox,...]
    masks_padded[:min_bbox,...]=masks[:min_bbox,...]
    return masks_padded, bboxes_padded
    
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
    gt_boxes[:,:4] = np.clip(gt_boxes[:,:4],0,cfg.TRAIN_SIZE-1)
    gt_boxes = gt_boxes.astype(np.uint64)
    return image_paded, masks_padded, gt_boxes

def data_augment( image, masks, bboxes):
    image = random_brightness(image)
    image, masks, bboxes = random_horizontal_flip(image, masks, bboxes)
    image, masks, bboxes = random_crop(image, masks, bboxes)
    image, masks, bboxes = random_translate(image, masks, bboxes)
    return image, masks, bboxes
    
def random_brightness(image):
    if np.random.random() < 0.5:
        image_hsv = rgb2hsv(image)
        scale = np.minimum(np.min(image_hsv[:,:,2]), 1.0-np.max(image_hsv[:,:,2]))
        sign = 1 if np.random.random() < 0.5 else -1
        image_hsv[:,:,2]+= sign*np.random.random()*scale
        image = np.round(hsv2rgb(image_hsv)*255).astype(np.uint8)
    return image

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
        image = np.full(shape=[h, w, 3], fill_value=128).astype(np.uint8)
        masks = np.zeros_like(masks)
        image[max(0,ty):min(h,h+ty),max(0,tx):min(w,w+tx),:] = old_image[max(0,-ty):min(h,h-ty),max(0,-tx):min(w,w-tx),:]
        masks[:,max(0,ty):min(h,h+ty),max(0,tx):min(w,w+tx)] = old_masks[:,max(0,-ty):min(h,h-ty),max(0,-tx):min(w,w-tx)]
        bboxes[:, [0, 2]] += tx
        bboxes[:, [1, 3]] += ty
        
    return image, masks, bboxes

###############################################################################################
####################################   OUTPUT DECODING   ######################################
###############################################################################################

COLORS = []
for name, code in ImageColor.colormap.items():
     COLORS.append(name)
np.random.shuffle(COLORS)

     
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

def decode_mask(box, conf, class_id, mask, mode='keep'):
    box, conf, class_id, mask = box.numpy(), conf.numpy(), class_id.numpy(), mask.numpy()
    if mode == 'cut':
        cut = (conf>cfg.CONF_THRESH)
        return box[cut], conf[cut], class_id[cut], mask[cut]
    else:
        return box, conf, class_id, mask

def decode_target_mask(proposal, target_masks, mode='keep'):
    bbox = proposal[...,:4]
    conf = proposal[...,4].numpy()
    bbox = tf.round(bbox*cfg.TRAIN_SIZE).numpy()
    mask_mrcnn = target_masks
    mask_mrcnn = tf.transpose(mask_mrcnn,(1,2,0)).numpy()
    if mode == 'cut':
        mask = conf>cfg.CONF_THRESH
        return bbox[mask], conf[mask], mask_mrcnn[...,mask]
    else:
        return bbox, conf, mask_mrcnn
    
def decode_ground_truth(gt_masks, gt_bboxes):
    gt_class_id = tf.cast(gt_bboxes[...,4],tf.int32).numpy()
    gt_bbox = gt_bboxes[...,:4]
    mask = gt_class_id>cfg.CONF_THRESH
    gt_bbox = gt_bbox.numpy()
    gt_mask = gt_masks.numpy()
    return gt_bbox[mask], gt_class_id[mask], gt_mask[mask]
    
def show_infer(data, prediction, class_dict):
    images, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
    boxes, confs, class_ids, masks = prediction
    for image, box, conf, class_id, mask in zip(images, boxes, confs, class_ids, masks):
        box, conf, class_id, mask = decode_mask(box, conf, class_id, mask, 'cut')
        if len(box)>0:
            draw_bbox(image, box, conf, class_id, mask, class_dict, mode= 'PIL')
        else:
            show_image(tf.cast(image*255,tf.uint8).numpy())
        tf.print('Found {} subjects'.format(len(box)))
 
    
def show_mAP(data, prediction, iou_thresholds = None, verbose = 0):
    image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
    mean_AP = []
    boxes, confs, class_ids, masks = prediction
    for gt_mask, gt_bbox, box, conf, class_id, mask in zip(gt_masks, gt_bboxes, boxes, confs, class_ids, masks):
        gt_bbox, gt_class_id, gt_mask = decode_ground_truth(gt_mask, gt_bbox)
        pred_box, pred_score, pred_class_id, pred_mask = decode_mask(box, conf, class_id, mask,'cut')
        gt_mask = np.transpose(gt_mask,(1,2,0))
        pred_mask = np.transpose(pred_mask,(1,2,0))
        if len(gt_bbox)>0: # this is never the case but better to put
            AP = compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=iou_thresholds, verbose=verbose)
            mean_AP.append(AP)
    return np.mean(mean_AP)            
    
def draw_bbox(image, box=None, conf=None, class_id=None, mask=None, class_dict=None, mode='return'):
    if np.any(class_id is None):
        class_id = np.arange(len(box))
    elif class_dict is not None:
        class_id = [class_dict[int(c)] for c in class_id]
    img = Image.fromarray(np.array(image*255,dtype=np.uint8))                   
    draw = ImageDraw.Draw(img)   
    if np.any(box is not None):
        for i,(bb,cc) in enumerate(zip(box, class_id)):
            if np.any(bb>0):
                draw.rectangle(bb, outline = COLORS[i%len(COLORS)]) 
                xy = ((bb[2]+bb[0])*0.5, (bb[3]+bb[1])*0.5)
                if not type(cc) is str:
                    cc = str(np.round(cc,3))
                draw.text(xy, cc, \
                          font=ImageFont.truetype("./other/arial.ttf"), \
                          fontsize = 15, fill=COLORS[i%len(COLORS)])
    img = np.array(img)
    if np.any(mask is not None):
        for i, (mm, bb) in enumerate(zip(mask, box)):
            xyxy = np.array(bb,dtype=np.int32)
            if np.any(xyxy>0):
                mm = unmold_mask(mm, xyxy, img.shape)
                mm = np.array(Image.new('RGB', (img.shape[0], img.shape[1]), \
                                        color = COLORS[i%len(COLORS)])) * \
                                        np.tile(mm[...,None],(1,1,3))
                img[mm>0] = img[mm>0]*0.5+mm[mm>0]*0.5
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

def encode_labels(bboxes):
    ANCHORS = tf.reshape(tf.constant(cfg.ANCHORS,dtype=np.int32),[cfg.LEVELS, cfg.NUM_ANCHORS, 2])
    label_2 = encode_label(bboxes, ANCHORS[0]/cfg.STRIDES[0], cfg.NUM_ANCHORS, cfg.TRAIN_SIZE//cfg.STRIDES[0], cfg.TRAIN_SIZE//cfg.STRIDES[0], cfg.NUM_CLASSES, cfg.TOLERANCE[0])
    label_3 = encode_label(bboxes, ANCHORS[1]/cfg.STRIDES[1], cfg.NUM_ANCHORS, cfg.TRAIN_SIZE//cfg.STRIDES[1], cfg.TRAIN_SIZE//cfg.STRIDES[1], cfg.NUM_CLASSES, cfg.TOLERANCE[1])
    label_4 = encode_label(bboxes, ANCHORS[2]/cfg.STRIDES[2], cfg.NUM_ANCHORS, cfg.TRAIN_SIZE//cfg.STRIDES[2], cfg.TRAIN_SIZE//cfg.STRIDES[2], cfg.NUM_CLASSES, cfg.TOLERANCE[2])
    label_5 = encode_label(bboxes, ANCHORS[3]/cfg.STRIDES[3], cfg.NUM_ANCHORS, cfg.TRAIN_SIZE//cfg.STRIDES[3], cfg.TRAIN_SIZE//cfg.STRIDES[3], cfg.NUM_CLASSES, cfg.TOLERANCE[3])
    return label_2, label_3, label_4, label_5

# original labeling code, no mask is provided
def encode_label(target, anchor_wh, nA, nGh, nGw, nC, tol):
    # remove zero pad
    non_zero_entry = tf.where(tf.logical_not(tf.reduce_all(tf.equal(target,0.0),axis=-1)))[...,0]
    target = tf.gather(target, non_zero_entry,axis=0)
    target = tf.cast(target, tf.float32)
    gt_boxes = target[:,:4]/cfg.TRAIN_SIZE
    # encode from 0 to num_class-1
    ids = tf.math.subtract(target[:,4],1.0)
    
    gt_boxes = tf.clip_by_value(gt_boxes, 0.0, 1.0)
    gt_boxes = xyxy2xywh(gt_boxes)
    gt_boxes = gt_boxes * tf.cast([nGw, nGh, nGw, nGh], tf.float32)
    anchor_wh = tf.cast(anchor_wh, tf.float32)
    tbox = tf.zeros((nA, nGh, nGw, 4))
    tconf = tf.zeros(( nA, nGh, nGw))
    tid = tf.fill(dims=(nA, nGh, nGw, 1),value=-1.)
    
    anchor_mesh = generate_anchor(nGh, nGw, anchor_wh)# Shape nA x 4 x nGh x nGw
    anchor_mesh = tf.transpose(anchor_mesh, (0,2,3,1))# Shpae nA x nGh x nGw x 4 
    anchor_list = tf.reshape(anchor_mesh,(-1, 4))# Shpae (nA x nGh x nGw) x 4 
    iou_pdist = bbox_iou(anchor_list, gt_boxes)# Shape (nA x nGh x nGw) x Ng
    iou_max = tf.math.reduce_max(iou_pdist, axis=1)# Shape (nA x nGh x nGw), both
    max_gt_index = tf.math.argmax(iou_pdist, axis=1)# Shape (nA x nGh x nGw), both
     
    iou_map = tf.reshape(iou_max, (nA, nGh, nGw))     
    gt_index_map = tf.reshape(max_gt_index,(nA, nGh, nGw))     
    
    id_index = iou_map > cfg.ID_THRESH * tol
    fg_index = iou_map > cfg.FG_THRESH * tol                                              
    bg_index = iou_map < cfg.BG_THRESH * tol
    ign_index = tf.cast(tf.cast(tf.logical_not(fg_index),tf.float32) \
                        * tf.cast(tf.logical_not(bg_index),tf.float32),tf.bool)
    tconf = tf.where(fg_index, 1.0, tconf)
    tconf = tf.where(bg_index, 0.0, tconf)
    tconf = tf.where(ign_index, -1.0, tconf)
    tconf = tconf[...,None]

    gt_id_list = tf.gather(ids,tf.boolean_mask(gt_index_map,id_index))
    tid = tf.where(id_index[...,None], tf.scatter_nd(tf.where(id_index),  gt_id_list[:,None], (nA, nGh, nGw, 1)), tid)
    
    fg_anchor_list = anchor_mesh[fg_index] 
    gt_box_list = tf.gather(gt_boxes,tf.boolean_mask(gt_index_map,fg_index))
    delta_target = encode_delta(gt_box_list, fg_anchor_list)
#    cond = tf.greater(tf.reduce_sum(tf.cast(fg_index,tf.float32)), 0)
#    tbox = tf.cond(cond, lambda: tf.scatter_nd(tf.where(fg_index),  delta_target, (nA, nGh, nGw, 4)), lambda: tbox)
    tbox = tf.where(fg_index[...,None], tf.scatter_nd(tf.where(fg_index),  delta_target, (nA, nGh, nGw, 4)), tbox)
        
    label = tf.concat([tbox,tconf,tid],axis=-1)
    # need to transpose since for some reason the labels are rotated, maybe scatter_nd?
    label = tf.transpose(label,(0,2,1,3))
    return label

def decode_label(label, anchors, stride):
    label = tf.transpose(label,(0,1,3,2,4)) # decode_delta_map has some issue
    pconf = label[..., 4]
    pclass = label[..., 5]# Conf
    pbox = label[..., :4]
    pbox = decode_delta_map(pbox, tf.divide(anchors,stride))
    pbox = tf.multiply(pbox,stride) # now in range [0, .... cfg.TRAIN_SIZE]
    pbox = tf.divide(pbox,cfg.TRAIN_SIZE) #now normalized in [0...1]
    pbox = xywh2xyxy(pbox) # to bbox
    pbox = tf.clip_by_value(pbox,0.0,1.0) # clip to avoid nan
    proposal = tf.concat([pbox, pconf[...,tf.newaxis], pclass[...,tf.newaxis]], axis=-1) 
    nB = tf.shape(proposal)[0]
    _, nA, nGh, nGw, nC = proposal.shape       
    proposal = tf.reshape(proposal, [nB, tf.multiply(nA, tf.multiply(nGh , nGw)), nC]) # b x nBB x (4 + 1 + 1 + 208) rois
   
    pconf = proposal[..., 4]
    indices = tf.argsort(pconf, axis=-1, direction='DESCENDING', stable=True)
    proposal = tf.gather(proposal,indices, axis=1, batch_dims=1) 
    
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
    
    pbox, pconf, pclass = tf.split(proposals,(4,1,1),axis=-1)
    
    pclass = tf.one_hot(tf.cast(pclass[...,0],tf.int32),cfg.NUM_CLASSES)
    
    proposals = tf.concat([pbox, pconf, pclass], axis=-1)
    
    proposals = nms_proposals_tensor(proposals)
    
#    proposals = proposals * tf.tile(proposals[...,4][...,None],(1,1,tf.shape(proposals)[-1]))
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
    
def decode_prediction(prediction, embedding, anchors, stride):
    prediction = tf.transpose(prediction,(0,1,3,2,4)) # decode_delta_map has some issue
    pconf = prediction[..., 4:6]  # class
    pconf = tf.nn.softmax(pconf, axis=-1)[...,1:] # class
    pclass = embedding #prediction[..., 6:]  # class
    pclass = tf.nn.softmax(pclass, axis=-1) # class
    pclass = tf.tile(pclass[:,tf.newaxis,:,:,:],(1,cfg.NUM_ANCHORS,1,1,1))
    pbox = prediction[..., :4]
    pbox = decode_delta_map(pbox, tf.divide(anchors,stride))
    pbox = tf.multiply(pbox,stride) # now in range [0, .... cfg.TRAIN_SIZE]
    pbox = tf.divide(pbox,cfg.TRAIN_SIZE) #now normalized in [0...1]
    pbox = xywh2xyxy(pbox) # to bbox
    pbox = tf.clip_by_value(pbox,0.0,1.0) # clip to avoid nan
    proposal = tf.concat([pbox, pconf, pclass], axis=-1) 
    nB = tf.shape(proposal)[0]
    _, nA, nGh, nGw, nC = proposal.shape       
    proposal = tf.reshape(proposal, [nB, tf.multiply(nA, tf.multiply(nGh , nGw)), nC])
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
    pconf = proposal[..., 4]
    indices = tf.argsort(pconf, axis=-1, direction='DESCENDING', stable=True)
    proposal = tf.gather(proposal,indices, axis=1, batch_dims=1) 
    proposal = nms_proposals(proposal)

    return proposal
#    
#def nms_proposals_tensor(proposal,size = cfg.MAX_PROP):
#    """This function compute nms proposals without iterating over the batch """
#    pconf = proposal[...,4:5]
#    ppred = proposal[...,5:]
#    pconf = pconf * ppred
#    pboxes = tf.tile(proposal[...,:4][:,:,tf.newaxis,:],(1,1,cfg.NUM_CLASSES,1))
#    proposal, conf, category, *_ = tf.image.combined_non_max_suppression(
#            boxes = pboxes, scores = pconf, max_output_size_per_class=cfg.MAX_PROP_PER_CLASS, \
#            max_total_size=size, iou_threshold=cfg.NMS_THRESH,pad_per_class=True, clip_boxes=True)
#    proposal = tf.concat([proposal,conf[...,tf.newaxis],category[...,tf.newaxis]], axis=-1)    
#    return proposal

#def nms_proposals_tensor(proposal):
#    """This function compute nms proposals without iterating over the batch """
#    pconf = proposal[...,4:5] * proposal[...,5:]
#    pboxes = tf.tile(proposal[...,:4][:,:,tf.newaxis,:],(1,1,cfg.NUM_CLASSES,1))
#    boxes, conf, category, *_ = tf.image.combined_non_max_suppression(
#            boxes = pboxes, scores = pconf, max_output_size_per_class=cfg.MAX_PROP_PER_CLASS, \
#            max_total_size=cfg.MAX_PROP, iou_threshold=cfg.NMS_THRESH,pad_per_class=True, clip_boxes=True)
#    category+=1.
#    proposal = tf.concat([boxes,conf[...,tf.newaxis],category[...,tf.newaxis]], axis=-1)    
#    return proposal

def nms_proposals(proposal):
   # non max suppression
   pboxes = proposal[...,:4]
   pconf = proposal[..., 4] 
   indices, _ = tf.image.non_max_suppression_padded(pboxes, pconf, max_output_size=cfg.PRE_NMS_LIMIT, iou_threshold=cfg.NMS_THRESH, pad_to_max_output_size=True)

   proposal = tf.gather(proposal, indices, axis=1, batch_dims=1) #b x n rois x (4+1+1+208)
   
   return proposal
#
def nms_proposals_tensor(proposal):
   # non max suppression
   pboxes = proposal[...,:4]
   pconf = proposal[..., 4]
   indices, _ = tf.image.non_max_suppression_padded(pboxes, pconf, max_output_size=cfg.MAX_PROP, iou_threshold=cfg.NMS_THRESH, pad_to_max_output_size=True)

   proposal = tf.gather(proposal, indices, axis=1, batch_dims=1) #b x n rois x (4+1+1+208)
   
   return proposal
#
## Non-max suppression
#def nms(proposals):
#    boxes, scores = proposals[...,:4], proposals[...,4]
#    indices = tf.image.non_max_suppression(
#        boxes, scores, cfg.MAX_PROP,
#        cfg.NMS_THRESH)
#    proposals = tf.gather(boxes, indices)
#    # Pad if needed
#    padding = tf.maximum(cfg.MAX_PROP - tf.shape(proposals)[0], 0)
#    proposals = tf.pad(proposals, [(0, padding), (0, 0)])
#    return proposals

###############################################################################################
#################################   MASK RCNN PROCESSING   ####################################
###############################################################################################


def preprocess_target_bbox(proposal_gt_bbox):
    """ proposal: [num_rois, (x, y, w, h)] 
        gt_bbox: [num_gt_bbox, (x, y, w, h)] """
    proposal, gt_bbox, gt_indices = proposal_gt_bbox
    non_zero_proposals = tf.not_equal(tf.reduce_sum(proposal,axis=-1),0.0)
    valid_indices = tf.where(non_zero_proposals)[...,0]
    valid_proposals = tf.gather(proposal, valid_indices, axis=0)
    assigned_gt_bboxes = tf.gather(gt_bbox,gt_indices,axis=0)
    valid_gt_bboxes = tf.gather(assigned_gt_bboxes,valid_indices, axis=0)
    encoded_delta = encode_delta(valid_gt_bboxes, valid_proposals)
    non_valid_indices = tf.where(tf.logical_not(non_zero_proposals))[...,0]
    non_valid_gt_bboxes = tf.gather(proposal,non_valid_indices)
    target_bbox = tf.concat([encoded_delta, non_valid_gt_bboxes], axis=0)
    return target_bbox

def preprocess_target_mask(proposal_gt_mask):
    """ proposal: [num_rois, (x, y, w, h)] 
        gt_mask: [num_gt_bbox, mask_dim, mask_dim] """
    proposal, gt_mask, gt_indices = proposal_gt_mask
    non_zero_proposals = tf.not_equal(tf.reduce_sum(proposal,axis=-1),0.0)
    valid_indices = tf.where(non_zero_proposals)[...,0]
    non_valid_indices = tf.where(tf.logical_not(non_zero_proposals))[...,0]
    assigned_gt_masks = tf.gather(gt_mask,gt_indices,axis=0)
    valid_gt_masks = tf.gather(assigned_gt_masks, valid_indices, axis=0)
    non_valid_gt_masks = tf.gather(tf.zeros_like(assigned_gt_masks),non_valid_indices)
    target_mask = tf.concat([valid_gt_masks, non_valid_gt_masks], axis=0) #zero padded

    return target_mask


def preprocess_target_indices(proposal_gt_mask):
    """ proposal: [num_rois, (x, y, w, h)] 
        gt_bbox: [num_gt_bbox, (x, y, w, h)] """
    proposal, gt_bbox, gt_class_id = proposal_gt_mask
    gt_intersect = bbox_iou(proposal,gt_bbox)
    target_indices = tf.math.argmax(gt_intersect,axis=-1)
    # Determine positive and negative ROIs
    valid_indices = tf.reduce_max(gt_intersect, axis=-1)
    target_class_ids = tf.cast(valid_indices >= cfg.IOU_THRESH, tf.int64) # there is bbox
    gt_class_id = tf.cast(gt_class_id,  tf.int64)
    gt_class_id = tf.gather(gt_class_id, target_indices)
    target_class_ids *= tf.cast(gt_class_id,  tf.int64)
    return target_indices, target_class_ids 

def crop_and_resize(proposal_gt_bbox_gt_mask):
    proposal, target_class_id, target_mask = proposal_gt_bbox_gt_mask
    nP = tf.shape(proposal)[0]
    targets = tf.where(tf.greater(target_class_id,0))[:,0]
    proposals_ = tf.gather(proposal, targets, axis=0)
    x1, y1, x2, y2 = tf.unstack(xywh2xyxy(proposals_[...,:4]), axis=-1)
    proposals_ = tf.concat([y1[...,tf.newaxis],x1[...,tf.newaxis],y2[...,tf.newaxis],x2[...,tf.newaxis]],axis=-1)
    target_mask = tf.gather(target_mask, targets, axis=0)
    box_indices = tf.range((tf.shape(targets)[0]),dtype=tf.int32)
    target_mask = tf.image.crop_and_resize(target_mask[...,tf.newaxis], proposals_,\
                                           box_indices = box_indices, crop_size=(cfg.MASK_SIZE,cfg.MASK_SIZE),method="bilinear")[...,0]
    target_mask = tf.round(target_mask)
    target_mask = tf.clip_by_value(target_mask,0.0,1.0)
    target_mask = tf.scatter_nd(targets[...,tf.newaxis],target_mask,(nP,cfg.MASK_SIZE,cfg.MASK_SIZE))
    return target_mask
    
def preprocess_mrcnn(proposals, gt_bboxes, gt_masks):
    """ target_bboxs: [batch, num_rois, (dx, dy, log(dw), log(dh))]
     target_class_ids: [batch, num_rois]. Integer class IDs.
     target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. """
    gt_class_ids = gt_bboxes[...,4]
    gt_bboxes = gt_bboxes[...,:4]
    gt_bboxes /= cfg.TRAIN_SIZE
    gt_bboxes = xyxy2xywh(gt_bboxes)
    proposals = tf.stop_gradient(proposals)
    proposals = proposals[...,:4]
    proposals = xyxy2xywh(proposals)  
    target_indices, target_class_ids = tf.map_fn(preprocess_target_indices, (proposals, gt_bboxes, gt_class_ids), fn_output_signature=(tf.int64,tf.int64))
    target_bboxes = tf.map_fn(preprocess_target_bbox, (proposals, gt_bboxes, target_indices), fn_output_signature=tf.float32)
    target_masks = tf.map_fn(preprocess_target_mask, (proposals, gt_masks, target_indices), fn_output_signature=tf.float32)
    target_masks = tf.cond(tf.greater(tf.reduce_sum(target_class_ids),0), lambda: tf.map_fn(crop_and_resize, (proposals, target_class_ids, target_masks), fn_output_signature=tf.float32),lambda: target_masks)
    target_class_ids = tf.stop_gradient(target_class_ids)# from o to num_class
    target_masks = tf.stop_gradient(target_masks)
    target_bboxes = tf.stop_gradient(target_bboxes)

    return target_class_ids, target_bboxes, target_masks


class FreezeBackbone(tf.keras.callbacks.Callback):
    def __init__(self, n_epochs=2):
        super().__init__()
        self.n_epochs = n_epochs

    def on_epoch_start(self, epoch, logs=None):
        if epoch <= self.n_epochs:
            freeze_backbone(self.model, trainable=False)
        else:
            freeze_backbone(self.model, trainable=True)

class FreezeBatchNorm(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_start(self, epoch, logs=None):
        freeze_batch_norm(self.model)


def freeze_batch_norm(model, trainable = False):  
    for layer in model.layers:
        bn_layer_name = layer.name
        if bn_layer_name[:10] == 'batch_norm':
            bn_layer = model.get_layer(bn_layer_name)
            bn_layer._trainable = trainable
        else:
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
            
def freeze_rpn(model, trainable = False):
    cutoff = 561
    for layer in model.layers[:cutoff]:
        layer._trainable = trainable

def fine_tuning(model):
    out_layers = [16, 37, 58, 77]
    conv_0 = int(model.layers[1].name.split('_')[-1]) if not model.layers[1].name == 'conv2d' else 0
    batch_0 = int (model.layers[2].name.split('_')[-1]) if not model.layers[2].name == 'batch_normalization' else 0
    for i in out_layers:
        k = i + conv_0
        j = i + batch_0
        conv_layer_name = 'conv2d_%d' %k if k > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'
        conv_layer = model.get_layer(conv_layer_name)       
        bn_layer = model.get_layer(bn_layer_name)
        conv_layer.trainable = True
        bn_layer.trainable = True

class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("alb_total_loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class EarlyStoppingRPN(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience1=0,patience2=0):
        super(EarlyStoppingRPN, self).__init__()
        self.patience1 = patience1
        self.patience2 = patience2
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        self.freeze_rpn = False
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("alb_total_loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if not self.freeze_rpn:
                if self.wait >= self.patience1:
                    self.wait = 0
                    self.stopped_epoch = epoch
                    self.freeze_rpn = True
                    self.model.set_weights(self.best_weights) 
                    print("Restoring model weights from the end of the best epoch.")
            else:
                if self.wait >= self.patience2:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print("Restoring model weights from the end of the best epoch.")
                    self.model.set_weights(self.best_weights)


    def on_epoch_start(self, epoch, logs=None):
        if self.freeze_rpn:
            freeze_rpn(self.model,trainable=False)
            print('Freezed RPN')

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

def log_confusion_matrix(epoch, logs):
  # Use the model to predict the values from the validation dataset.
  test_pred_raw = model.predict(test_images)
  test_pred = np.argmax(test_pred_raw, axis=1)

  # Calculate the confusion matrix.
  cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
  # Log the confusion matrix as an image summary.
  figure = plot_confusion_matrix(cm, class_names=class_names)
  cm_image = plot_to_image(figure)

  # Log the confusion matrix as an image summary.
  with file_writer_cm.as_default():
    tf.summary.image("Confusion Matrix", cm_image, step=epoch)

# Define the per-epoch callback.
#cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)