#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 18:40:02 2020

@author: fiorapirri
"""


import json
from PIL import Image
import numpy as np
import config as cfg
import tensorflow as tf

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

def plot_xywh(anchor_list):
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(0,len(anchor_list)):
        x,y,w,h = anchor_list[i]
        x = [x-w/2,x+w/2,x+w/2,x-w/2,x-w/2]
        y = [y+h/2,y+h/2,y-h/2,y-h/2,y+h/2]
#        w,h = anchor_list[i:i+2]
#        x = [-w/2,w/2,w/2,-w/2,-w/2]
#        y = [-h/2,-h/2,h/2,h/2,-h/2]
        plt.plot(x,y,'-', color=tuple(np.random.rand(3,1).squeeze()))
    plt.axis('equal')
    plt.show()
    
def plot_anchor(anchor_list):
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(0,len(anchor_list)):
        w,h = anchor_list[i]
        x=0
        y=0
        x = [x-w/2,x+w/2,x+w/2,x-w/2,x-w/2]
        y = [y+h/2,y+h/2,y-h/2,y-h/2,y+h/2]
#        w,h = anchor_list[i:i+2]
#        x = [-w/2,w/2,w/2,-w/2,-w/2]
#        y = [-h/2,-h/2,h/2,h/2,-h/2]
        plt.plot(x,y,'-', color=tuple(np.random.rand(3,1).squeeze()))
    plt.axis('equal')
    plt.show()

def plot_boxes(anchor_list):
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(0,len(anchor_list)):
        x1,y1,x2,y2 = anchor_list[i]
        x = [x1, x1,x2,x2,x1]
        y = [y1,y2,y2,y1,y1]
#        w,h = anchor_list[i:i+2]
#        x = [-w/2,w/2,w/2,-w/2,-w/2]
#        y = [-h/2,-h/2,h/2,h/2,-h/2]
        plt.plot(x,y,'-', color=tuple(np.random.rand(3,1).squeeze()))
    plt.axis('equal')
    plt.show()

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
    
def batch_bbox_iou(box1, box2, x1y1x2y2=False):
    batch_iou = []
    for b1, b2 in  zip(box1, box2):
        batch_iou.append(bbox_iou(b1,b2,x1y1x2y2))
    return tf.stack(batch_iou, axis=0)
           
def get_top_proposals(batch_proposals):
    proposal_bests = []
    for pred in batch_proposals:
        pred = pred[pred[..., 4] > cfg.CONF_THRESH]
        indices = tf.argsort(pred[..., 4], axis=-1, direction='DESCENDING')[:cfg.MAX_PRED]
        pred = tf.gather(pred,indices)
        pred = tf.pad(pred,[[0,cfg.MAX_PRED-tf.shape(pred)[0]],[0,0]])
        proposal_bests.append(pred)
    proposals = tf.stack(proposal_bests, axis=0)
    return proposals

def nms_proposals(batch_proposals):
    # we havefor each level proposals for each image in the batch, thus to be more flexible we transpose the list of lists
    # pythonic transpose list of list of irregular size: [[image1 - fpn1, image2 - fpn1, ..], [image1 - fpn2], [image2 - fpn2], ..]
#        l = [len(i) for i in proposals]
#        proposals = [[i[o] for ix, i in enumerate(proposals) if l[ix] > o] for o in range(max(l))] 
#        proposals = [tf.concat(p, axis=0) for p in proposals]
    proposals_filtered = []
    for pred in batch_proposals: # batch images
        pred = pred[pred[..., 4] > cfg.CONF_THRESH]
        boxes = xywh2xyxy(pred)
        pred = tf.concat([boxes,pred[...,4:]],axis=-1) # to bbox
            # pred now has lesser number of proposals. Proposals rejected on basis of object confidence score
        if len(pred) > 0: 
            boxes = pred[...,:4]
            scores = pred[...,4]
            selected_indices = tf.image.non_max_suppression(
                                boxes, scores, max_output_size=cfg.MAX_PROP, iou_threshold=cfg.NMS_THRESH,
                                score_threshold=cfg.CONF_THRESH
                            )
            pred_emb = tf.gather(pred, selected_indices) #b x n rois x (4+1+1+208)
            proposals_filtered.append(tf.pad(pred_emb,[[0,cfg.MAX_PROP-pred_emb.shape[0]],[0,0]]))
        else:
            proposals_filtered.append(tf.zeros((cfg.MAX_PROP,batch_proposals.shape[-1])))
    proposals_filtered = tf.stack(proposals_filtered, axis=0)
    return proposals_filtered

def xyxy2xywh(xyxy):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    # x, y are coordinates of center 
    # (x1, y1) and (x2, y2) are coordinates of bottom left and top right respectively. 
    xy = (xyxy[...,:2]+xyxy[...,2:4])/2
    wh = (xyxy[...,2:4]-xyxy[...,:2])
    return tf.concat([xy,wh],axis=-1)


def xywh2xyxy(xywh):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    # x, y are coordinates of center 
    # (x1, y1) and (x2, y2) are coordinates of bottom left and top right respectively. 
    x1y1 = xywh[...,:2] - xywh[...,2:4]*0.5
    x2y2 = xywh[...,:2] + xywh[...,2:4]*0.5
    return tf.concat([x1y1, x2y2], axis=-1)


def encode_target(target, anchor_wh, nA, nC, nGh, nGw):
    assert(tf.shape(anchor_wh)[0]==nA)
    target = tf.constant(target, dtype=tf.float32)
    bbox = target[:,:4]/cfg.TRAIN_SIZE
    ids = target[:,4]

    gt_boxes = xyxy2xywh(bbox)
    gt_boxes = gt_boxes * tf.constant([nGw, nGh, nGw, nGh], dtype=tf.float32)
    gt_boxes = tf.clip_by_value(gt_boxes, [0,0,0,0], [nGw, nGh, nGw, nGh])                                        # Shape Ngx4 (xc, yc, w, h)
    anchor_wh = tf.constant(anchor_wh, dtype=tf.float32)
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
    tconf = tf.where(fg_index, 1, tconf)
    tconf = tf.where(bg_index, 0, tconf)
    tconf = tf.where(ign_index, -1, tconf)

    gt_index = tf.boolean_mask(gt_index_map,fg_index)
    gt_box_list = tf.gather(gt_boxes,gt_index)
    gt_id_list = tf.gather(ids,tf.boolean_mask(gt_index_map,id_index))

    if tf.reduce_sum(tf.cast(fg_index,tf.float32)) > 0:
        tid = tf.scatter_nd(tf.where(id_index),  gt_id_list[:,None], (nA, nGh, nGw, nC))
        tid = tf.where(tf.equal(tid,0),  -1, tid)
        fg_anchor_list = tf.reshape(anchor_list,(nA, nGh, nGw, 4))[fg_index] 
        delta_target = encode_delta(gt_box_list, fg_anchor_list)
        tbox = tf.scatter_nd(tf.where(fg_index),  delta_target, (nA, nGh, nGw, 4))
#    tconf = tf.transpose(tconf,(0,2,1))
#    tbox = tf.transpose(tbox,(0,2,1,3))
#    tid = tf.transpose(tid,(0,2,1,3))
    tconf, tbox, tid = tf.cast(tconf,tf.float32), tf.cast(tbox,tf.float32), tf.cast(tid,tf.float32)
    label = tf.concat([tbox,tconf[...,None],tid],axis=-1)
#    label = tf.transpose(label,(0,2,1,3))
    return label

def generate_anchor(nGh, nGw, anchor_wh):
    nA = tf.shape(anchor_wh)[0]
    yy, xx =tf.meshgrid(tf.range(nGh), tf.range(nGw))
    mesh = tf.stack([xx, yy], axis=0)                                              # Shape 2, nGh, nGw
    mesh = tf.cast(tf.tile(mesh[None],(nA,1,1,1)),tf.float32)                          # Shape nA x 2 x nGh x nGw
    anchor_offset_mesh = tf.tile(anchor_wh[...,None][...,None],(1, 1, nGh,nGw))  # Shape nA x 2 x nGh x nGw
    anchor_mesh = tf.concat([mesh, anchor_offset_mesh], axis=1)                       # Shape nA x 4 x nGh x nGw
    return anchor_mesh

def encode_delta(gt_box_list, fg_anchor_list):
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
    '''
    _, nA, nGh, nGw, _ = delta_map.shape
    nB = tf.shape(delta_map)[0] # error in building if None
    anchor_mesh = generate_anchor(nGh, nGw, anchors) 
    anchor_mesh = tf.transpose(anchor_mesh, (0,2,3,1))              # Shpae (nA x nGh x nGw) x 4
    anchor_mesh = tf.tile(anchor_mesh[tf.newaxis],(nB,1,1,1,1))
#    delta_map = tf.reshape(delta_map,(-1,4))#* tf.reshape([0.1, 0.1, 0.2, 0.2],(1,4))
    pred_list = decode_delta(tf.reshape(delta_map,(-1,4)), tf.reshape(anchor_mesh,(-1,4)))
    pred_map = tf.reshape(pred_list,(nB, nA, nGh, nGw, 4))
    return pred_map


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # RPN_ANCHOR_SCALES = ( 8,16 , 32 , 64 , 128 )
    # RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    # backbone_shape=[[32 32][16 16][ 8  8][ 4  4][ 2  2]]
    # BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # RPN_ANCHOR_STRIDE = 1
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)


    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.
    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    # RPN_ANCHOR_SCALES = ( 8,16 , 32 , 64 , 128 )
    # RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    # backbone_shape=[[32 32][16 16][ 8  8][ 4  4][ 2  2]]
    # BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # RPN_ANCHOR_STRIDE = 1
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)
