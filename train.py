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

# tensorboard --logdir /home/fiorapirri/Documents/workspace/tracker/writer --port 6006
    
ds = DataLoader()
model = tracker(data_loader = ds)
model.custom_build()
#model.plot()
#model.bkbn.model.summary() 
#model.neck.summary()
#model.head.summary()
model.summary()
model.fit()
#import matplotlib.pyplot as plt
#import time 
#avg = 0
#start = time.time()
for image, *labels in ds.train_ds.take(10).batch(1):
#    print(image.shape)
#    avg += time.time()-start
#    start=time.time()
#
#print(avg/200)
#    tf.print(image.shape)
#    for label in labels:
#        tf.print(label.shape)
#     plt.imshow(label_5[0,0,:,:,4])
#     plt.show()
#     plt.imshow(image[0])
#     plt.show()
     model.infer(image)
#    plt.imshow(image[0])
#    plt.show()
#    for label in labels:
#        for anchor in label[0]:
#            plt.imshow(anchor[:,:,4])
#            plt.show()
