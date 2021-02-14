import os
import config as cfg
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='3'

import gc 
gc.collect()

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
else: 
    print('No GPU found')
    


from models import MSDS
from loader import DataLoader 

# tensorboard --logdir /media/data4/Models/simenv/tracker/logdir --port 6006
# scp /home/fiorapirri/Documents/workspace/tracker4/weights/yolov4.weights alcor@Alcor:/media/data4/Models/simenv/tracker/weights/yolov4.weights

ds = DataLoader(shuffle=True, data_aug=True)
model = MSDS(data_loader = ds, emb = False, mask = True)
model.custom_build()
model.summary()
# model.load('./weights/MSDS_noemb_mask_14_-6.43556_2021-02-14-02-21-56.tf')
# model.trainable = False # too fucking important for inferring
model.fit()

# TEST

# import time 
# from utils import show_infer, show_mAP, draw_bbox, filter_inputs
# import numpy as np

# i = 0
# fps = 0
# AP = 0
# iterator = ds.train_ds.apply(tf.data.experimental.copy_to_device("/gpu:0"))\
#                 .prefetch(tf.data.experimental.AUTOTUNE).repeat()\
#                 .filter(filter_inputs)
# for data in iterator.take(100).batch(1):
#     i=i+1
#     image = data[0]
#     start = time.perf_counter()
#     predictions = model.infer(image)
#     end = time.perf_counter()-start
#     fps += 1/end
#     mfps=fps/i
#     print(mfps)
#     preds, embs, proposals, logits, probs, bboxes, masks = predictions
#     image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
# 	draw_bbox(image[0], bboxs = gt_bboxes[0], masks=tf.transpose(gt_masks[0],(1,2,0)), conf_id = None, mode= 'PIL')
# 	show_infer(data, predictions)
# 	AP += show_mAP(data, predictions, mode='return')
# 	mAP = AP/i    
# 	print(mAP)