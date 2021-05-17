#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:37:02 2021

@author: fiorapirri
"""


#%%%%%%%%%%%%%%%%%%%%%%%%%%% BUILD ENV %%%%%%%%%%%%%%%%%%%%%%%%%%%%§%%%%%%
import env


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import config as cfg
import tensorflow as tf
from loader_ytvos import DataLoader 
from model import get_model
import numpy as np
from PIL import Image

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from model import get_model

model = get_model(infer=True)

#fine_tuning(model)

model.load_weights('/home/fiorapirri/tracker/weights/model.54--7.149.h5')

model.trainable = False

#model.summary()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import matplotlib.pyplot as plt
from loader_ytvos import DataLoader 
#import time
from utils import show_infer, compute_mAP, draw_bbox, show_mAP, encode_labels, crop_and_resize,xyxy2xywh, decode_ground_truth, unmold_mask_batch
from utils import rle_encoding
import json
ds = DataLoader(shuffle=False, augment=False)
iterator = ds.train_ds.unbatch().batch(1).__iter__()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
path = os.path.join('dataset')
os.makedirs(path,exist_ok=True)
path_frames = os.path.join(path,'frames')
os.makedirs(path_frames,exist_ok=True)
path_features = os.path.join(path,'features')
os.makedirs(path_features,exist_ok=True)
path_json = os.path.join(path,'instances.json')
path_gt_json = os.path.join(path,'instances_gt.json')

dataset = []
gt = []
for i,data in enumerate(ds.train_ds.unbatch().batch(1).skip(1000).take(1000)):
    image, gt_masks, gt_bboxes = data
    img_name = str(i)+'.jpeg'
    image_path = os.path.join(path_frames,img_name)
    Image.fromarray(np.array(image[0].numpy()*255,np.uint8)).save(image_path)
    predictions = model.infer(image)
    box, conf, class_id, mask, features = predictions
    mask = unmold_mask_batch(mask, box)
    predictions = box, conf, class_id, mask 
#    show_infer(image, predictions, ds.class_dict)
    lim = tf.reduce_sum(tf.cast(conf > 0.7,tf.int32))
    box, conf, class_id, mask, features = box[0,:lim].numpy(), conf[0,:lim].numpy(), class_id[0,:lim].numpy(), mask[0,:lim].numpy(), features[0,:lim].numpy()
 
    item = dict()
    for j,(b,cf,c,m,f) in enumerate(zip(box, conf, class_id, mask, features)):
        feature_name = str(i)+'_'+str(j)+'.npy'
        feature_path = os.path.join(path_features,feature_name)
        np.save(feature_path, f)
        item['frame_name'] = image_path
        item['feature_name'] = feature_path
        item['size'] = [416, 416]
        item['box'] = [int(x) for x in list(b)]
        item['class'] = int(c)
        item['conf'] = float(round(cf,2))
        item['mask'] = rle_encoding(m)
        dataset.append(item.copy())
    
    item = dict()
    lim = tf.reduce_sum(tf.cast(tf.cast(tf.reduce_sum(gt_bboxes,axis=-1),tf.bool),tf.int32))
    for m,b in zip(gt_masks[0,:lim], gt_bboxes[0,:lim]):
        item['frame_name'] = image_path
        item['size'] = [416, 416]
        item['box']  = [int(x) for x in list(b[:4])]
        item['class'] = int(b[4])
        item['mask'] = rle_encoding(m.numpy())
        gt.append(item.copy())
#    img = draw_bbox(image[0].numpy(), box = box, mask=mask, class_id = class_id, class_dict = ds.class_dict, mode= 'return')
#    plt.imshow(img)
#    plt.show()
#    gt_bbox, gt_class_id, gt_mask = decode_ground_truth(gt_masks[0], gt_bboxes[0])
#    img = draw_bbox(image[0].numpy(), box = gt_bbox, mask=gt_mask, class_id = gt_class_id, class_dict = ds.class_dict, mode= 'return')
#    plt.imshow(img)
#    plt.show()
    
with open(path_json, 'w') as f:
    f.write(json.dumps(dataset)) 
with open(path_gt_json, 'w') as f:
    f.write(json.dumps(gt)) 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
#for i, item in enumerate(dataset):
#    dataset[i]['box']= [int(x) for x in list(item['box'])] 
#    dataset[i]['class'] = int(item['class']) 
#    dataset[i]['conf'] = round(float(item['conf']),2)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

def rle_decoding(rle):
    # dict.get(key,[value,]) return a defult value if key not exists
    h, w = rle.get('size',[720,1280])
    rle_arr = rle.get('counts',[0])
    rle_arr = np.cumsum(rle_arr)
    indices = []
    extend = indices.extend
    list(map(extend, map(lambda s,e: range(s, e), rle_arr[0::2], rle_arr[1::2])));
    binary_mask = np.zeros(h*w, dtype=np.uint8)
    binary_mask[indices] = 1
    return binary_mask.reshape((w, h)).T


import json
with open('dataset/instances.json','r') as f:
    d = json.load(f)
    
for i in d:
    image_path = os.path.join(i['frame_name'])
    box = np.array(i['box'],dtype=np.float32)[None]
    class_id = np.array(i['class'])[None]
    print(cfg.CLASS_YTVIS19[i['class']])
    mask = np.array(rle_decoding(i['mask']))[None]
    img = np.array(Image.open(image_path))/255
    img = draw_bbox(img, box = box, mask=mask, class_id = class_id, class_dict = cfg.CLASS_YTVIS19, mode= 'return')
    plt.imshow(img,interpolation='nearest', aspect='auto')
    plt.show()
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
import matplotlib.pyplot as plt
from loader_ytvos import DataLoader 
#import time
from utils import draw_bbox, unmold_mask_batch, rle_encoding, bbox_iou_batch

import json
ds = DataLoader(shuffle=False, augment=False)
import pandas as pd
ds.val_list = pd.read_pickle(r'val_list.txt')
ds.val_ds = ds.initilize_val_ds()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

from model import get_model

model = get_model(infer=True)

#fine_tuning(model)

model.load_weights('/home/fiorapirri/tracker/weights/model.54--7.149.h5')

model.trainable = False

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
import sklearn.metrics
import io
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
  figure = plt.figure(figsize=(20, 20))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

#  # Compute the labels from the normalized confusion matrix.
#  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
#
#  # Use white text if squares are dark; otherwise black.
#  threshold = cm.max() / 2.
#  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#    color = "white" if cm[i, j] > threshold else "black"
#    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

preds = []
labels = []
confs = []
probs = []
thresh = 0.5
path = os.path.join('scores')
os.makedirs(path,exist_ok=True)
path = os.path.join(path,str(thresh))
os.makedirs(path,exist_ok=True)

for data in ds.val_ds.unbatch().batch(1).take(4907):
    image, gt_masks, gt_bboxes = data
    box, conf, class_id, mask, prob = model.infer(image)
    gate = tf.reduce_sum(tf.cast(conf>=thresh,tf.int32))
    if gate>0:
        conf = conf[:,:gate]
        prob = prob[:,:gate,:]
        box = box[:,:gate,:]
        class_id = class_id[:,:gate]
        mask = mask[:,:gate,:,:] 
        mask = unmold_mask_batch(mask, box)
    #    img = draw_bbox(image[0].numpy(), box = box[0].numpy(), mask=mask[0].numpy(), class_id = class_id[0].numpy(), class_dict = ds.class_dict, mode= 'return')
    #    plt.imshow(img)
    #    plt.show()
        gate = tf.reduce_sum(tf.cast(tf.cast(tf.reduce_sum(gt_bboxes,axis=-1),tf.bool),tf.int32))
        gt_class_id = gt_bboxes[:,:gate,4]
        gt_box = gt_bboxes[:,:gate,:4]
        gt_intersect = tf.map_fn(bbox_iou_batch, (box,gt_box), fn_output_signature=tf.float32)
        target_indices = tf.math.argmax(gt_intersect,axis=-1)
        # Determine positive and negative ROIs
        valid_indices = tf.reduce_max(gt_intersect, axis=-1)
        valid_indices = tf.cast(valid_indices >= cfg.IOU_THRESH, tf.int64) # there is bbox
        gt_class_id = tf.cast(gt_class_id,  tf.int64)
        gt_class_id = tf.gather(gt_class_id, target_indices,axis=-1,batch_dims=-1)
        target_class_ids = valid_indices * tf.cast(gt_class_id,  tf.int64)
        pred_class_ids = valid_indices * tf.cast(class_id,  tf.int64)
        conf = tf.cast(valid_indices,tf.float32) * conf
        prob = tf.tile(tf.cast(valid_indices,tf.float32)[...,None],(1,1,40)) * prob
        target_class_ids = tf.reshape(target_class_ids,(-1))
        pred_class_ids = tf.reshape(pred_class_ids,(-1))
        conf = tf.reshape(conf,(-1))
        prob = tf.reshape(prob,(-1,40))
        # remove zero padded bboxes
        indices = tf.reshape(tf.where(tf.greater(target_class_ids,0)),(-1))
        target_class_ids = tf.gather(target_class_ids, indices)
        pred_class_ids = tf.gather(pred_class_ids, indices)
        conf = tf.gather(conf, indices)
        prob = tf.gather(prob, indices)
        # remove zero padded predictions (low conf)
        indices = tf.reshape(tf.where(tf.greater(pred_class_ids,0)),(-1))
        target_class_ids = tf.gather(target_class_ids, indices)
        pred_class_ids = tf.gather(pred_class_ids, indices)
        conf = tf.gather(conf, indices)
        prob = tf.gather(prob, indices)
        assert len(target_class_ids) == len(pred_class_ids)
        labels.extend(list(target_class_ids.numpy()))
        preds.extend(list(pred_class_ids.numpy()))
        confs.extend(list(conf.numpy()))
        probs.append(prob)

probs = tf.concat(probs,axis=0).numpy()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
# Calculate the confusion matrix.
cm = sklearn.metrics.confusion_matrix(labels, preds)
# Log the confusion matrix as an image summary.
#figure = plot_confusion_matrix(cm, class_names=ds.class_dict)
#cm_image = plot_to_image(figure)
disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=ds.class_dict.values())
disp.plot(ax=plt.subplots(figsize=(20, 20))[1],xticks_rotation='vertical') 

macro_roc_auc_ovo = sklearn.metrics.roc_auc_score(labels, probs, multi_class="ovo",
                                  average="macro")
weighted_roc_auc_ovo = sklearn.metrics.roc_auc_score(labels, probs, multi_class="ovo",
                                     average="weighted")

macro_roc_auc_ovr = sklearn.metrics.roc_auc_score(labels, probs, multi_class="ovr",
                                  average="macro")
weighted_roc_auc_ovr = sklearn.metrics.roc_auc_score(labels, probs, multi_class="ovr",
                                     average="weighted")

with open(os.path.join(path,'scores.txt'),'a') as f:
    f.write("One-vs-One ROC AUC scores:{:.6f} (macro), {:.6f} "
          "(weighted by prevalence) "
          .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
with open(os.path.join(path,'scores.txt'),'a') as f:
    f.write("One-vs-Rest ROC AUC scores:{:.6f} (macro), {:.6f} "
          "(weighted by prevalence) "
          .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

ap = sklearn.metrics.accuracy_score(labels, preds)
with open(os.path.join(path,'scores.txt'),'a') as f:
    f.write('accuracy_score: {:.6f} '.format(ap))

prec, rec, f1, samples = sklearn.metrics.precision_recall_fscore_support(labels, preds, average=None)

with open(os.path.join(path,'scores.txt'),'a') as f:
    f.write('precision: {} '.format(str(list(prec))))
    f.write('recall: {} '.format(str(list(rec))))
    f.write('f1: {} '.format(str(list(f1))))
    f.write('samples number: {} '.format(str(list(samples))))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(40):
    fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(tf.one_hot(np.array(labels)-1,40).numpy()[:,i], np.array(probs)[:,i])
    roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

path = os.path.join(path,'roc')
os.makedirs(path,exist_ok=True)

for cl in range(0,40):
    plt.figure()
    plt.plot(fpr[cl], tpr[cl], color='darkorange',
             label='ROC curve (area = %0.2f)' % roc_auc[cl])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for class: '+ds.class_dict[cl+1])
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(path,'roc'+str(cl+1)+'.jpg'))
#    plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LIB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
