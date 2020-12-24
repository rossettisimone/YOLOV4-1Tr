#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:53:32 2020

@author: fiorapirri
"""


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


def imm_resize(img):
    imgx = img.resize((cfg.TRAIN_SIZE, cfg.TRAIN_SIZE), Image.ANTIALIAS)
    return np.array(imgx)/255

def mask_resize(img, width=cfg.TRAIN_SIZE, height=cfg.TRAIN_SIZE):
    img = Image.fromarray(img)
    imgx = img.resize((width,height), Image.ANTIALIAS)
    return np.array(imgx,dtype=np.uint8)

def mask_clamp(mask):
    if np.any(mask>1):
        mask[mask==1]=0
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

def plot_anchor(anchor_list):
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(0,len(anchor_list),2):
        w,h = anchor_list[i:i+2]
        x = [-w/2,w/2,w/2,-w/2,-w/2]
        y = [-h/2,-h/2,h/2,h/2,-h/2]
        plt.plot(x,y,'-', color=tuple(np.random.rand(3,1).squeeze()))
    plt.axis('equal')
    plt.show()

import torch 
def box_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes
    """
    N, M = len(box1), len(box2)
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
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1,1).expand(N,M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1,-1).expand(N,M)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def encode_target(target, anchor_wh, nA, nC, nGh, nGw):
    ID_THRESH = 0.5
    FG_THRESH = 0.5
    BG_THRESH = 0.4
    assert(len(anchor_wh)==nA)
    target = torch.Tensor(target.astype(np.float32))
    target[:,:4]/=416
#    print(target.shape)
    target[:,2:4] = target[:,2:4]-target[:,:2]
    anchor_wh = torch.Tensor(anchor_wh.numpy())
    tbox = torch.zeros( nA, nGh, nGw, 4).cuda()  # batch size, anchors, grid size
    tconf = torch.LongTensor( nA, nGh, nGw).fill_(0).cuda()
    tid = torch.LongTensor( nA, nGh, nGw, 1).fill_(-1).cuda() 
    t = target
    
    t_id = t[:, 4].clone().long().cuda()
    t = t[:,[0,1,2,3]]

    gxy, gwh = t[: , :2].clone() , t[:, 2:].clone()
    gxy[:, 0] = gxy[:, 0] * nGw
    gxy[:, 1] = gxy[:, 1] * nGh
    gwh[:, 0] = gwh[:, 0] * nGw
    gwh[:, 1] = gwh[:, 1] * nGh
    gxy[:, 0] = torch.clamp(gxy[:, 0], min=0, max=nGw -1)
    gxy[:, 1] = torch.clamp(gxy[:, 1], min=0, max=nGh -1)

    gt_boxes = torch.cat([gxy, gwh], dim=1)                                            # Shape Ngx4 (xc, yc, w, h)
    
    anchor_mesh = generate_anchor(nGh, nGw, anchor_wh)
    anchor_list = anchor_mesh.permute(0,2,3,1).contiguous().view(-1, 4)              # Shpae (nA x nGh x nGw) x 4
    #print(anchor_list.shape, gt_boxes.shape)
    iou_pdist = box_iou(anchor_list.cuda(), gt_boxes.cuda())                                      # Shape (nA x nGh x nGw) x Ng
    iou_max, max_gt_index = torch.max(iou_pdist, dim=1)                              # Shape (nA x nGh x nGw), both

    iou_map = iou_max.view(nA, nGh, nGw)       
    gt_index_map = max_gt_index.view(nA, nGh, nGw)

    #nms_map = pooling_nms(iou_map, 3)
    
    id_index = iou_map > ID_THRESH
    fg_index = iou_map > FG_THRESH                                                    
    bg_index = iou_map < BG_THRESH 
    ign_index = (iou_map < FG_THRESH) * (iou_map > BG_THRESH)
    tconf[fg_index] = 1
    tconf[bg_index] = 0
    tconf[ign_index] = -1

    gt_index = gt_index_map[fg_index]
    gt_box_list = gt_boxes[gt_index]
    gt_id_list = t_id[gt_index_map[id_index]]
    #print(gt_index.shape, gt_index_map[id_index].shape, gt_boxes.shape)
    if torch.sum(fg_index) > 0:
        tid[id_index] =  gt_id_list.unsqueeze(1)
        fg_anchor_list = anchor_list.view(nA, nGh, nGw, 4)[fg_index] 
        delta_target = encode_delta(gt_box_list.cuda(), fg_anchor_list.cuda())
        tbox[fg_index] = delta_target
    tconf, tbox, tid = tf.cast(tconf.cpu().numpy(),tf.float32),tf.cast(tbox.cpu().numpy(),tf.float32),tf.cast(tid.cpu().numpy(),tf.float32)
    label = tf.concat([tbox.cpu().numpy(), tconf[...,None].cpu().numpy(), tid.cpu().numpy()], axis=-1)
    return label

def generate_anchor(nGh, nGw, anchor_wh):
    nA = len(anchor_wh)
    yy, xx =torch.meshgrid(torch.arange(nGh), torch.arange(nGw))
    xx, yy = xx.cuda(), yy.cuda()

    mesh = torch.stack([xx, yy], dim=0)                                              # Shape 2, nGh, nGw
    mesh = mesh.unsqueeze(0).repeat(nA,1,1,1).float()                                # Shape nA x 2 x nGh x nGw
    anchor_offset_mesh = anchor_wh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nGh,nGw).cuda() # Shape nA x 2 x nGh x nGw
    anchor_mesh = torch.cat([mesh, anchor_offset_mesh], dim=1)                       # Shape nA x 4 x nGh x nGw
    return anchor_mesh

def encode_delta(gt_box_list, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], \
                     gt_box_list[:, 2], gt_box_list[:, 3]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw/pw)
    dh = torch.log(gh/ph)
    return torch.stack([dx, dy, dw, dh], dim=1)

def decode_delta(delta, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    dx, dy, dw, dh = delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3]
    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)
    return torch.stack([gx, gy, gw, gh], dim=1)

def decode_delta_map():
    return 0
#