#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 19:25:16 2021

@author: fiorapirri
"""
import tensorflow as tf

def main():
#    test_check_proposals()
#    test_nms_proposals()
#    test_preprocess_mrcnn();
    test_loss_mrcnn()
    test_encode_decode()
    test_encode_decode_loss()
    test_tf_mask_transform()
    
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
    bad_proposals = tf.zeros((2,40,4))
    proposals = tf.concat([proposals, bad_proposals], axis=1)
    gt_bboxes_1 = tf.random.uniform((2,5,2))*0.5
    gt_bboxes_2 = tf.random.uniform((2,5,2))*0.5 + 0.5
    gt_class_ids = tf.round(tf.random.uniform((2,5,1),1,40))
    gt_bboxes = tf.concat([gt_bboxes_1, gt_bboxes_2], axis=-1)* cfg.TRAIN_SIZE
    gt_bboxes = tf.concat([gt_bboxes, gt_class_ids], axis=-1)
    gt_masks = tf.round(tf.random.uniform((2,5,28,28)))
#    proposals = tf.zeros((2,20,4))
#    gt_bboxes = tf.zeros((2,10,4))
#    gt_masks = tf.zeros((2,10,28,28))
#    gt_class_ids = tf.ones((2,10,1))
#    gt_bboxes = tf.concat([gt_bboxes, gt_class_ids], axis=-1)
    target_class_ids, target_bboxes, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks)
    
    return target_class_ids, target_bboxes, target_masks

def test_loss_mrcnn():
    from model import mask_loss_graph, class_loss_graph, bbox_loss_graph
    import config as cfg
    target_class_ids, target_bboxes, target_masks = test_preprocess_mrcnn()
    pred_masks = tf.concat([(((target_masks-1)*-2)-1)[...,None],((target_masks*2)-1)[...,None]],axis=-1)*100
    pred_class = tf.one_hot(target_class_ids,40)*100
    pred_bbox = target_bboxes
    mrcnn_class_loss = class_loss_graph(target_class_ids, pred_class)
    mrcnn_box_loss = bbox_loss_graph(target_bboxes, target_class_ids, pred_bbox)
    mrcnn_mask_loss = mask_loss_graph(target_masks, target_class_ids, pred_masks)
    assert tf.equal(mrcnn_class_loss+mrcnn_box_loss+mrcnn_mask_loss, 0)
    
def test_encode_decode():
    from loader_avakin import DataLoader
    from utils import draw_bbox, encode_labels
    from utils import preprocess_mrcnn
    from utils import decode_target_mask
    from utils import decode_labels,crop_and_resize,xyxy2xywh
    import numpy as np
    import config as cfg
    
    ds = DataLoader(shuffle=True, augment=True)
    iterator = ds.train_ds.unbatch().batch(1).__iter__()
    data = iterator.next()
    image, gt_masks, gt_bboxes = data
    gt_masks_ = tf.map_fn(crop_and_resize, (xyxy2xywh(gt_bboxes)/cfg.TRAIN_SIZE, tf.cast(tf.greater(gt_bboxes[...,4],-1.0),tf.float32), gt_masks), fn_output_signature=tf.float32)
    draw_bbox(image[0].numpy(), prop = gt_bboxes[0,...,:4].numpy(), bboxs = gt_bboxes[0].numpy(), masks=tf.transpose(gt_masks_[0],(1,2,0)).numpy(), conf_id = np.arange(20), mode= 'PIL')
    label_2, label_3, label_4, label_5 = tf.map_fn(encode_labels, (gt_bboxes, gt_masks_), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
    proposals = decode_labels([label_2,label_3,label_4,label_5])
    draw_bbox(image[0].numpy(), prop = proposals[0,:,:4].numpy()*cfg.TRAIN_SIZE, bboxs = proposals[0,:,:4].numpy()*cfg.TRAIN_SIZE,masks=tf.transpose(gt_masks_[0],(1,2,0)).numpy(), conf_id = np.arange(20), mode= 'PIL')
    target_class_ids, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks)
    bbox_mrcnn, mask_mrcnn = decode_target_mask(proposals[0], target_class_ids[0], target_masks[0])
    draw_bbox(image[0].numpy(), prop = proposals[0,:,:4].numpy()*cfg.TRAIN_SIZE, bboxs = bbox_mrcnn[...,:4],masks=mask_mrcnn, conf_id = np.arange(20), mode= 'PIL')
    
    
def test_encode_decode_loss():
    from loader_avakin import DataLoader
    from utils import encode_labels
    from utils import preprocess_mrcnn
    from utils import decode_labels,crop_and_resize,xyxy2xywh
    import config as cfg
    import tensorflow as tf

    ds = DataLoader(shuffle=True, augment=True)
    iterator = ds.train_ds.unbatch().batch(1).__iter__()
    data = iterator.next()
    image, gt_masks, gt_bboxes = data
    gt_masks = tf.map_fn(crop_and_resize, (xyxy2xywh(gt_bboxes)/cfg.TRAIN_SIZE, tf.cast(tf.greater(gt_bboxes[...,4],-1.0),tf.float32), gt_masks), fn_output_signature=tf.float32)
    label_2, label_3, label_4, label_5 = tf.map_fn(encode_labels, (gt_bboxes, gt_masks), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
    proposals = decode_labels([label_2,label_3,label_4,label_5])
    target_class_ids, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks)
    from model import mrcnn_mask_loss_graph
    pred_masks = tf.tile(target_masks[...,None],(1,1,1,1,1))
    loss = mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks)
    assert tf.equal(loss, 0)
    
    


def test_tf_mask_transform(NUM_TESTS=10, verbose = 0):

    from loader_avakin import DataLoader
    from utils import draw_bbox
    import matplotlib.pyplot as plt
    import numpy as np
    import config as cfg
    import tensorflow as tf
    from PIL import Image
    from utils import preprocess_mrcnn,bbox_iou
    from utils import decode_labels,crop_and_resize,xyxy2xywh

    ds = DataLoader(shuffle=True, augment=False)
    iterator = ds.train_ds.unbatch().batch(2).__iter__()

    def plot(bbox1,bbox2):
        b1x = np.array([bbox1[0],bbox1[2],bbox1[2],bbox1[0],bbox1[0]])
        b1y = np.array([bbox1[1],bbox1[1],bbox1[3],bbox1[3],bbox1[1]])
        b2x = np.array([bbox2[0],bbox2[2],bbox2[2],bbox2[0],bbox2[0]])
        b2y = np.array([bbox2[1],bbox2[1],bbox2[3],bbox2[3],bbox2[1]])
        x1 = np.maximum(bbox2[0], bbox1[0])
        y1 = np.maximum(bbox2[1], bbox1[1])
        x2 = np.minimum(bbox2[2], bbox1[2])
        y2 = np.minimum(bbox2[3], bbox1[3])
        plt.plot(b1x,b1y,'g-')
        plt.plot(b2x,b2y,'r-')
        plt.plot([x1,x2],[y1,y2],'bo')
        plt.xlim(0,416)
        plt.ylim(0,416)
        plt.gca().invert_yaxis()

    avg_score = 0
    subjects = 0
    for j in range(NUM_TESTS):

        data = iterator.next()
        
        image, gt_masks, gt_bboxes = data
        gt_masks = tf.map_fn(crop_and_resize, (xyxy2xywh(gt_bboxes)/cfg.TRAIN_SIZE, tf.cast(tf.greater(gt_bboxes[...,4],-1.0),tf.float32), gt_masks), fn_output_signature=tf.float32)
        proposals = gt_bboxes[...,:4]/cfg.TRAIN_SIZE
        noise = tf.random.uniform(proposals.shape,0.8,1.2)# avoid exit the IOU threshold
        proposals *= noise
        target_class_ids_tf, _, target_masks_tf = preprocess_mrcnn(proposals, gt_bboxes, gt_masks)
        
        
        image, gt_masks, gt_bboxes = data
        gt_masks = tf.map_fn(crop_and_resize, (xyxy2xywh(gt_bboxes)/cfg.TRAIN_SIZE, tf.cast(tf.greater(gt_bboxes[...,4],-1.0),tf.float32), gt_masks), fn_output_signature=tf.float32)
        proposals = gt_bboxes[...,:4]/cfg.TRAIN_SIZE
        proposals *= noise
        proposals *=cfg.TRAIN_SIZE
    
        for i in range(proposals.shape[1]):
            if tf.greater(tf.reduce_sum(gt_bboxes[0,i:i+1,:4]),0.0):
#                plt.imshow(draw_bbox(image[0].numpy(), prop = proposals[0,:,:4].numpy()*cfg.TRAIN_SIZE, bboxs = gt_bboxes[0].numpy(), prop = proposals, masks=tf.transpose(gt_masks[0],(1,2,0)).numpy(), conf_id = None, mode= 'return'))
            #    plt.show()
                iou = bbox_iou(gt_bboxes[0,i:i+1],proposals[0,i:i+1], x1y1x2y2 = True)
                print('IOU: ',iou[0,0].numpy())
                
            #    plt.show()
                if iou>cfg.IOU_THRESH:
                    m = gt_masks[0,i].numpy()
                #    plt.imshow(m)
                #    plt.show()
                #    from PIL import Image
                    bb = np.array(gt_bboxes[0,i,:4].numpy(),np.int32)
                    hw = np.round((bb[2:4]-bb[:2]))
                    n = np.round(Image.fromarray(m).resize(tuple(hw),resample = Image.BILINEAR))
                #    plt.imshow(n)
                #    plt.show()
                    p = np.array(proposals[0,i,:4],np.int32)
                    hhww = np.array((p[2:4]-p[:2]),np.int32)
                    c = np.zeros((hhww[1],hhww[0]))
                    
                    x1 = np.maximum(bb[0], p[0])
                    y1 = np.maximum(bb[1], p[1])
                    x2 = np.minimum(bb[2], p[2])
                    y2 = np.minimum(bb[3], p[3])
                    
                    xs = max(p[0],x1) - p[0]
                    ys = max(p[1],y1) - p[1] 
                    xe = min(p[2],x2) - p[0]
                    ye = min(p[3],y2) - p[1]
                    
                    xss = max(bb[0],x1) - bb[0]
                    yss = max(bb[1],y1) - bb[1]
                    xee = min(bb[2],x2) - bb[0]
                    yee = min(bb[3],y2) - bb[1]
                    
                    c[ys:ye,xs:xe] = n[yss:yee,xss:xee]
                #    plt.imshow(c)
                #    plt.show()
                    nn = np.round(Image.fromarray(c).resize((28,28),resample = Image.BILINEAR))
                    nn = np.clip(nn,0.0,1.0)
                    
                    if verbose:
                        plot((gt_bboxes[0,i]),(proposals[0,i]))
                        plt.show()
                        print('NUMPY CROP AND PAD')
                        plt.imshow(nn)
                        plt.show()
                        print('TF TRANSFORM')
                        plt.imshow(target_masks_tf[0,i])
                        plt.show()
                    
                    score = np.sum(np.abs(nn-target_masks_tf[0,i].numpy()))/(np.sum(nn)+1e-10)
                    print('SCORE (lower better): ',score)
                    if np.any(nn) and np.any(target_masks_tf[0,i].numpy()):
                        subjects+=1
                        avg_score+=score
                        if verbose:
                            print('avg score (lower better): ', avg_score/subjects)

    print('AVG SCORE (lower better): ', avg_score/subjects)
    
if __name__ == '__main__':
    
    main()
    
    