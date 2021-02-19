#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import os
import config as cfg

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPU

import gc 
gc.collect()

import tensorflow as tf

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
else: 
    print('No GPU found')
    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
from models import MSDS
from loader import DataLoader 

ds = DataLoader(shuffle=True, data_aug=False)
#with mirrored_strategy.scope():
model = MSDS(data_loader = ds, emb = False, mask = True)
model.custom_build()
#model.plot()
#model.bkbn.model.summary() 
#model.neck.summary()
#model.head.summary()
model.summary()
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#cspdarknet53 (cspdarknet53)  multiple                  38727520  
#_________________________________________________________________
#fpn (fpn)                    multiple                  20000512  
#_________________________________________________________________
#rpn (rpn)                    multiple                  4958368   
#_________________________________________________________________
#custom_proposal_layer (Custo multiple                  0         
#_________________________________________________________________
#fpn_classifier_AFP (fpn_clas multiple                  13180794  
#_________________________________________________________________
#fpn_mask_AFP (fpn_mask_AFP)  multiple                  19474058  
#=================================================================
#Total params: 96,341,257
#Trainable params: 57,582,969
#Non-trainable params: 38,758,288
#_________________________________________________________________

#model.load('./weights/MSDS_noemb_mask_28_0.46876_2021-02-15-20-17-44.tf')
#model.trainable = False # too fucking important for inferring

model.fit()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#import timeit
#
#input_layer = tf.random.uniform((1, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3))
#model.infer(input_layer); # Warm Up
#
#trials = 50
#print("Fps:", trials/timeit.timeit(lambda: model.infer(input_layer), number=trials))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#import time 
#from utils import show_infer, show_mAP, draw_bbox, filter_inputs
#import numpy as np
#
#import matplotlib.pyplot as plt
#from loader import DataLoader 
#
#i = 0
#sec = 0
#AP = 0
#ds = DataLoader(shuffle=True, data_aug=False)
#iterator = ds.train_ds.filter(filter_inputs).repeat().apply(tf.data.experimental.copy_to_device("/gpu:0"))\
#                .prefetch(tf.data.experimental.AUTOTUNE)
#data = iterator.batch(1).__iter__().next()
#image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes = data
#draw_bbox(image[0].numpy(), bboxs = gt_bboxes[0].numpy(), masks=tf.transpose(gt_masks[0],(1,2,0)).numpy(), conf_id = None, mode= 'PIL')
#plt.imshow(label_2[0,3,:,:,4])
#plt.show()
#plt.imshow(tf.reduce_sum(tf.reduce_sum(label_3[0],axis=0),axis=-1))
#plt.show()
#plt.imshow(tf.reduce_sum(tf.reduce_sum(label_4[0],axis=0),axis=-1))
#plt.show()
#plt.imshow(tf.reduce_sum(tf.reduce_sum(label_5[0],axis=0),axis=-1))
#plt.show()
#_ = model.infer(data[0])
#for data in iterator.take(100).batch(1):
#    image = data[0]
#    start = time.perf_counter()
#    predictions = model.infer(image)
#    end = time.perf_counter()-start
#    i+=1
#    sec += end
#    print(i/sec)
#    show_infer(data, predictions)
#    AP += show_mAP(data, predictions)
#    mAP = AP/i    
#    print(mAP)
#    preds, embs, proposals, logits, probs, bboxes, masks = predictions
#    image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
#    draw_bbox(image[0].numpy(), bboxs = gt_bboxes[0].numpy(), masks=tf.transpose(gt_masks[0],(1,2,0)).numpy(), conf_id = None, mode= 'PIL')


#%%
from loader import DataLoader
from utils import show_infer, show_mAP, draw_bbox, filter_inputs
import matplotlib.pyplot as plt
from utils import data_labels
ds = DataLoader(shuffle=True, data_aug=True)
iterator = ds.train_ds.repeat().__iter__()
data = iterator.next()
image, gt_masks, gt_bboxes = data
draw_bbox(image[0].numpy(), bboxs = gt_bboxes[0].numpy(), masks=tf.transpose(gt_masks[0],(1,2,0)).numpy(), conf_id = None, mode= 'PIL')

label_2, label_3, label_4, label_5 = tf.map_fn(data_labels, (gt_bboxes, gt_masks), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))

plt.imshow(tf.reduce_sum(tf.reduce_sum(label_2[0],axis=0),axis=-1))
plt.show()
plt.imshow(tf.reduce_sum(tf.reduce_sum(label_3[0],axis=0),axis=-1))
plt.show()
plt.imshow(tf.reduce_sum(tf.reduce_sum(label_4[0],axis=0),axis=-1))
plt.show()
plt.imshow(tf.reduce_sum(tf.reduce_sum(label_5[0],axis=0),axis=-1))
plt.show()

from utils import decode_labels 
p = [label_2,label_3,label_4,label_5]
proposals = decode_labels(p)

draw_bbox(image[0].numpy(), bboxs = proposals[0,:,:4].numpy()*cfg.TRAIN_SIZE,masks=tf.transpose(gt_masks[0],(1,2,0)).numpy(), mode= 'PIL')

#%%
import timeit
from loader import DataLoader
ds = DataLoader(shuffle=False, data_aug=False)
iterator = ds.train_ds.repeat().__iter__()
print('Time: ', timeit.timeit(lambda: iterator.next(), number = 100)/100)
