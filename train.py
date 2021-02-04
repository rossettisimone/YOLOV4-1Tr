import os
import config as cfg
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPU

import tensorflow as tf
#tf.get_logger().setLevel('WARNING')
tf.compat.v1.reset_default_graph()
#tf.debugging.enable_check_numerics()

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

#mirrored_strategy = tf.distribute.MirroredStrategy(devices=[device.name for device in logical_devices])
#print ('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

from models import MSDS
from loader import DataLoader 

# tensorboard --logdir /home/fiorapirri/Documents/workspace/tracker/logdir --port 6006
# scp /home/fiorapirri/Documents/workspace/tracker4/weights/yolov4.weights alcor@Alcor:/media/data4/Models/simenv/tracker/weights/yolov4.weights

ds = DataLoader(shuffle=True, data_aug=False)
#with mirrored_strategy.scope():
model = MSDS(data_loader = ds, emb = False, mask = True)
model.custom_build()
#model.plot()
#model.bkbn.model.summary() 
#model.neck.summary()
#model.head.summary()
model.summary()
#model.load('./weights/MSDS_noemb_mask_14_-22.57891_2021-02-01-21-35-00.tf')
#model.trainable = False # too fucking important for inferring
model.fit()

#/home/fiorapirri/.local/lib/python3.8/site-packages/tensorflow/python/framework/indexed_slices.py:435: UserWarning: Converting sparse IndexedSlices(IndexedSlices(indices=Tensor("gradient_tape/MSDS/proposal_layer/while_4/gradients/MSDS/proposal_layer/while_4/GatherV2_grad/Reshape_1:0", shape=(None,), dtype=int64), values=Tensor("gradient_tape/MSDS/proposal_layer/while_4/gradients/MSDS/proposal_layer/while_4/GatherV2_grad/Reshape:0", shape=(None, None), dtype=float32), dense_shape=Tensor("gradient_tape/MSDS/proposal_layer/while_4/gradients/MSDS/proposal_layer/while_4/GatherV2_grad/Cast:0", shape=(2,), dtype=int32))) to a dense Tensor of unknown shape. This may consume a large amount of memory.
#  warnings.warn(
          
#import time
#avg = 0
#start = time.time()
#from utils import decode_delta_map
#ds.anchors=tf.cast(tf.reshape(tf.constant(cfg.ANCHORS),(4,4,2)),tf.float32)
#ds.data_aug=False
#import numpy as np
#l = 0
#o = 0
#z = 0
#def tf_count(t, val):
#    elements_equal_to_value = tf.equal(t, val)
#    as_ints = tf.cast(elements_equal_to_value, tf.int32)
#    count = tf.reduce_sum(as_ints)
#    return count
#import matplotlib.pyplot as plt
#import numpy as np




#import time
#import contextlib
#@contextlib.contextmanager
#def options(options):
#  old_opts = tf.config.optimizer.get_experimental_options()
#  tf.config.optimizer.set_experimental_options(options)
#  try:
#    yield
#  finally:
#    tf.config.optimizer.set_experimental_options(old_opts)
#
#fps = 0
#i=0
#with options({'constant_folding': True}):
#for image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes in ds.train_ds.take(4).batch(1):
#        print(gt_bboxes[tf.reduce_sum(gt_bboxes,axis=-1)>0])
#        plt.imshow(image[0])
#        plt.show()
#        plt.imshow(gt_masks[0,0])
#        plt.show()
#        plt.imshow(gt_masks[0,1])
#        plt.show()
#        t0=time.time()
#    model.infer(image)
#        i+=1
#        fps+=1/(time.time()-t0)
#        print(fps/i)





#    print(gt_masks)
#    a=1
#    plt.imshow(image[0])
#    plt.show()
#    for i,m in enumerate(masks[0]):
#        if not np.all(m==0):for image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes in ds.val_ds.take(100).batch(1):
#    model.infer(image)
#            print(bboxes[0,i])
#            plt.imshow(m)
#            plt.show()
#    model.draw_bbox(image[0], bboxes[0][...,:4], bboxes[0][...,4:])
#    training = True
#    inferring = True
#    preds, embs, proposals, logits, probs, bboxes, masks = model(image, training=True, inferring=True)
#    for i,m in enumerate(masks[0]):
#        if not np.all(m==0):
#            print(bboxes[0,i])
#            plt.imshow(m)
#            plt.show()
##    plt.imshow(image[0])
##    plt.show()
#    labels = [label_2, label_3, label_4, label_5]
##    for l in labels:
##        print(l.shape)
#    for label in labels:
#        for anchor in label[0]:
#            l+=tf_count(anchor[:,:,4],-1)
#            o+=tf_count(anchor[:,:,4],1)
#            z+=tf_count(anchor[:,:,4],0)
#            plt.imshow(anchor[:,:,4])
##            if (np.any(anchor[:,:,4])!=0):
##                print(anchor[:,:,4])
#            plt.show()
            
#l
#Out[40]: <tf.Tensor: shape=(), dtype=int32, numpy=8767>
#
#o
#Out[41]: <tf.Tensor: shape=(), dtype=int32, numpy=7512>
#
#z
#Out[42]: <tf.Tensor: shape=(), dtype=int32, numpy=558321>
#+++++++++++++++++++++++++++++++++++++++++
#l
#Out[44]: <tf.Tensor: shape=(), dtype=int32, numpy=25565>
#
#o
#Out[45]: <tf.Tensor: shape=(), dtype=int32, numpy=18603>
#
#z
#Out[46]: <tf.Tensor: shape=(), dtype=int32, numpy=530432>
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
#            pred = pred[pred[..., 4] > 0.5]
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
#                            boxes, scores, max_output_size=20, iou_threshold=0.5,
#                            score_threshold=0.5
#                        )
#        boxes = tf.gather(boxes, selected_indices) #b x n rois x (4+1+1+208)
#        scores = tf.gather(scores, selected_indices) 
#        model.draw_bbox(image[0],boxes, scores)
