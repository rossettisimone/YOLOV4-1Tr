import os
import config as cfg
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPU

import tensorflow as tf
tf.compat.v1.reset_default_graph()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_devices = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_devices), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)

from models import tracker
from loader import DataLoader 

# tensorboard --logdir /media/data4/Models/simenv/tracker/logdir --port 6006
    
ds = DataLoader(shuffle=False)
model = tracker(data_loader = ds)
model.custom_build()
#model.plot()
#model.bkbn.model.summary() 
#model.neck.summary()
#model.head.summary()
#model.summary()
#model.load('tracker_weights_1.tf')
model.fit()
#import matplotlib.pyplot as plt
#import time 
#avg = 0
#start = time.time()
#from utils import decode_delta_map
#import numpy as np
#for image, label_2, label_3, label_4, label_5, bboxes in ds.train_ds.take(20).batch(1):
#    model.draw_bbox(image[0], bboxes[0][...,:4], bboxes[0][...,4:])
##    model.infer(image)
##    plt.imshow(image[0])
##    plt.show()
#    labels = [label_2, label_3, label_4, label_5]
##    for l in labels:
##        print(l.shape)
#    for label in labels:
#        for anchor in label[0]:
#            plt.imshow(anchor[:,:,4])
#            if (np.any(anchor[:,:,4])!=0):
#                print(anchor[:,:,4])
#            plt.show()
#    proposals = []
#    pb = []
#    ps = []
#    for i,label in enumerate(labels):
#        pred = label
#        pbox = pred[..., :4]
#        pconf = pred[..., 4:6]  # Conf
#        
#        pconf = pconf[...,0][...,tf.newaxis]
##        print(pbox.shape)
##        pemb = tf.math.l2_normalize(tf.tile(pemb[:,tf.newaxis],[1,cfg.NUM_ANCHORS,1,1,1]),axis=-1, epsilon=1e-12)
##            pcls = tf.zeros((tf.shape(pred)[0], tf.shape(pred)[1], tf.shape(pred)[2], tf.shape(pred)[3],1)) # useless
#        pbox = decode_delta_map(pbox, ds.anchors[i]/ds.strides[i])
#        pbox *= ds.strides[i]
#        preds = tf.concat([pbox, pconf], axis=-1)
#        preds = tf.reshape(preds, [preds.shape[0], -1, preds.shape[-1]]) # b x nBB x (4 + 1 + 1 + 208) rois
##            pred = tf.concat(preds, axis=1)
#        
#        for ii in range(preds.shape[0]): # batch images
#            pred = preds[ii]
#            pred = pred[pred[..., 4] > cfg.CONF_THRESH]
#            x1y1 = pred[...,:2] - pred[...,2:4]*0.5
#            x2y2 = pred[...,:2] + pred[...,2:4]*0.5
#            pred = tf.concat([tf.concat([x1y1, x2y2], axis=-1),pred[...,4:]],axis=-1) # to bbox
#            # pred now has lesser number of proposals. Proposals rejected on basis of object confidence score
#            if len(pred) > 0:    
#                boxes = pred[...,:4]
#                scores = pred[...,4]
#                pb.append(boxes)
#                ps.append(scores)
#    if len(pb)>0:
#        boxes = tf.concat(pb,axis=0)
#        scores = tf.concat(ps,axis=0)
#        selected_indices = tf.image.non_max_suppression(
#                            boxes, scores, max_output_size=20, iou_threshold=cfg.NMS_THRESH,
#                            score_threshold=0.9
#                        )
#        proposals = tf.gather(pred, selected_indices) #b x n rois x (4+1+1+208)
#        model.draw_bbox(image[0], proposals[...,:4], proposals[...,4:5])
