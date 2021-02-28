GPU = '0'
#GPU = '0,1,2,3'

DATASET_TYPE = 'ava'

# AVA_path 
VIDEOS_DATASET_PATH = "/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/frames"
SEGMENTS_DATASET_PATH = "/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/segments"
TRAIN_ANNOTATION_PATH = ["/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/kinetics_100_frames_boundings_train_v1.0.json"]
VAL_ANNOTATION_PATH = ["/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/kinetics_100_frames_boundings_train_v1.0.json"]

#VIDEOS_DATASET_PATH = "/media/data4/Datasets/Kinetics_AVA/frames"
#SEGMENTS_DATASET_PATH = "/media/data4/Datasets/Kinetics_AVA/segments/"
#TRAIN_ANNOTATION_PATH = ["/media/data4/Datasets/Kinetics_AVA/kinetics_frames_masks_train_v1.0.json", "/media/data4/Datasets/Kinetics_AVA/ava_frames_masks_train_v2.2.json"]
#VAL_ANNOTATION_PATH = ["/media/data4/Datasets/Kinetics_AVA/ava_frames_masks_val_v2.2.json"]

SHUFFLE = True
DATA_AUGMENT = True
LOGDIR = 'logdir'
WEIGHTS = 'weights'
MIN_BOX_DIM = 0.02
MIN_BOX_RATIO = 0.2

# Input 
BATCH = 8
TRAIN_SIZE = 416
INPUT_SHAPE= (TRAIN_SIZE, TRAIN_SIZE, 3)
MAX_INSTANCES = 15

ID_THRESH = 0.5
FG_THRESH = 0.5
BG_THRESH = 0.4

IOU_THRESH = 0.6
# Network
NUM_CLASS = 1
MAX_BBOX_PER_SCALE = 20

ANCHORS = [ 13,  41,  28,  82,  51, 104,  90, 117, 
          27,  82,  57, 165, 102, 209, 181, 235,
          41, 124,  86, 248, 154, 313, 272, 353,
          55, 165, 114, 331, 205, 418, 363, 470]

# [ 12,  26,  21,  44,  29,  76,  39, 109,
#            53,  61,  62, 179,  66, 103,  83, 226,
#            111, 187, 136, 298, 138,  99, 167, 193, 
#            239, 376, 322, 414, 349, 256, 424, 423]

import numpy as np
ANCHORS = np.array(np.array(ANCHORS)/416*TRAIN_SIZE,np.int32) # if want to test different input  size

NUM_ANCHORS = 4
BBOX_CLASS = 4
BBOX_REG = 1
MASK=0
STRIDES = [ 4, 8, 16, 32]

#XYSCALE= [1.4, 1.3, 1.2, 1.1, 1.05]

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
NMS_THRESH = 0.3
STEPS_PER_EPOCH_TRAIN = 10
STEPS_PER_EPOCH_VAL = 10

PRE_NMS_LIMIT = 150
MAX_PROP = 50

FC_LAYER_SIZE = TRAIN_SIZE*2

POOL_SIZE = 7
MASK_POOL_SIZE = POOL_SIZE*2
MASK_SIZE = MASK_POOL_SIZE*2 #28
