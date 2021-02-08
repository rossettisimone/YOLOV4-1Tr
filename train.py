import os
import config as cfg
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPU

import tensorflow as tf
#tf.get_logger().setLevel('WARNING')
tf.compat.v1.reset_default_graph()
# tf.debugging.enable_check_numerics()

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
#with mirrored_strategy.scope():

from models import MSDS
from loader import DataLoader 

# tensorboard --logdir /media/data4/Models/simenv/tracker/logdir --port 6006
# scp /home/fiorapirri/Documents/workspace/tracker4/weights/yolov4.weights alcor@Alcor:/media/data4/Models/simenv/tracker/weights/yolov4.weights

ds = DataLoader(shuffle=True, data_aug=True)
model = MSDS(data_loader = ds, emb = False, mask = True)
model.custom_build()
#model.plot()
#model.bkbn.model.summary() 
#model.neck.summary()
#model.head.summary()
model.summary()
#model.load('./weights/MSDS_noemb_nomask_20_-5.56_2021-01-26-11-09-44.tf')
#model.trainable = False # too fucking important for inferring
model.fit()

# import time
# import contextlib
# @contextlib.contextmanager
# def options(options):
#  old_opts = tf.config.optimizer.get_experimental_options()
#  tf.config.optimizer.set_experimental_options(options)
#  try:
#    yield
#  finally:
#    tf.config.optimizer.set_experimental_options(old_opts)

# fps = 0
# i=0
# with options({'constant_folding': True}):
# 	for image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes in ds.train_ds.take(4).batch(1):
# 	  t0=time.time()
# 		model.infer(image)
# 	  i+=1
# 	  fps+=1/(time.time()-t0)
# 	  print(fps/i)