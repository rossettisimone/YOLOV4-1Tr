# GPU = '0'
GPU = '0'

DATASET_TYPE = 'ava'
NET_TYPE = '3'
# AVA_path 
AVA_VIDEOS_DATASET_PATH = "/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/frames"
AVA_SEGMENTS_DATASET_PATH = "/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/segments"
AVA_TRAIN_ANNOTATION_PATH = ["/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/kinetics_100_frames_boundings_train_v1.0.json"]
AVA_VAL_ANNOTATION_PATH = ["/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/kinetics_100_frames_boundings_train_v1.0.json"]

#AVA_VIDEOS_DATASET_PATH = "/media/data3/Kinetics_AVA/frames"
#AVA_SEGMENTS_DATASET_PATH = "/media/data3/Kinetics_AVA/segments/"
#AVA_TRAIN_ANNOTATION_PATH = ["/media/data3/Kinetics_AVA/kinetics_frames_masks_train_v1.0.json", "/media/data3/Kinetics_AVA/ava_frames_masks_train_v2.2.json"]
#AVA_VAL_ANNOTATION_PATH = ["/media/data3/Kinetics_AVA/ava_frames_masks_val_v2.2.json"]

YT_TRAIN_ANNOTATION_PATH = "/media/data3/YoutubeVOS19/train_all_frames/train.json"
YT_TRAIN_FRAMES_PATH = "/media/data3/YoutubeVOS19/train_all_frames/JPEGImages/"
YT_VAL_ANNOTATION_PATH = "/media/data3/YoutubeVOS19/valid_all_frames/valid.json"
YT_VAL_FRAMES_PATH = "/media/data3/YoutubeVOS19/valid_all_frames/JPEGImages/"
YT_TEST_ANNOTATION_PATH = "/media/data3/YoutubeVOS19/test_all_frames/test.json"
YT_TEST_FRAMES_PATH = "/media/data3/YoutubeVOS19/test_all_frames/JPEGImages/"

SPLIT_RATIO = .1
SHUFFLE = True
DATA_AUGMENT = True
LOGDIR = 'logdir'
WEIGHTS = 'weights'
MIN_BOX_DIM = 0.01
MIN_BOX_RATIO = 0.01

# Input 
BATCH = 10
TRAIN_SIZE = 416
INPUT_SHAPE= (TRAIN_SIZE, TRAIN_SIZE, 3)
MAX_INSTANCES = 15

ID_THRESH = 0.4
FG_THRESH = 0.4
BG_THRESH = 0.3

IOU_THRESH = 0.5
# Network
NUM_CLASSES = 41

ANCHORS = [2,2, 4,7, 6,14, 8,3, 
           20,15, 22,41, 25,24, 34,48, 
           69,61, 81,34, 100,102, 101,66, 
           268,219, 305,84, 314,154, 378, 222]

# [ 13,  41,  28,  82,  51, 104,  90, 117, 
#           27,  82,  57, 165, 102, 209, 181, 235,
#           41, 124,  86, 248, 154, 313, 272, 353,
#           55, 165, 114, 331, 205, 418, 363, 470]

# [ 12,  26,  21,  44,  29,  76,  39, 109,
#            53,  61,  62, 179,  66, 103,  83, 226,
#            111, 187, 136, 298, 138,  99, 167, 193, 
#            239, 376, 322, 414, 349, 256, 424, 423]

import numpy as np
ANCHORS = np.array(np.array(ANCHORS)/416*TRAIN_SIZE,np.int32) # if want to test different input  size

NUM_ANCHORS = 4
BBOX_REG = 4
BBOX_CONF = 1


STRIDES = [ 4, 8, 16, 32]

LEVELS = 4
EMB_DIM = TRAIN_SIZE//2
CSP_DARKNET53 = './weights/yolov4.weights'

# Train
WD = 1e-4
LR = 1e-3
MOM = 0.9
GRADIENT_CLIP = 5.0
EPOCHS = 60
FINE_TUNING = 2

CONF_THRESH = 0.7
NMS_THRESH = 0.4
STEPS_PER_EPOCH_TRAIN = 1000
STEPS_PER_EPOCH_VAL = 100

PRE_NMS_LIMIT = 200
MAX_PROP = 50
MAX_PROP_PER_CLASS = 10

TOP_DOWN_PYRAMID_SIZE = 256
MASK_LAYERS_SIZE = 256
FC_LAYER_SIZE = 1024
POOL_SIZE = 7
MASK_POOL_SIZE = POOL_SIZE*2
MASK_SIZE = MASK_POOL_SIZE*2 #28
