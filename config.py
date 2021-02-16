GPU = '0'
DATASET_TYPE = 'ava'

# AVA_path 
VIDEOS_DATASET_PATH = "/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/frames"
SEGMENTS_DATASET_PATH = "/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/segments"
TRAIN_ANNOTATION_PATH = ["/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/kinetics_100_frames_boundings_train_v1.0.json"]
VAL_ANNOTATION_PATH = ["/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/kinetics_100_frames_boundings_train_v1.0.json"]

MIN_BOX_DIM = 0.05
MIN_BOX_RATIO = 0.2
SPLIT_RATIO = 0.7
SUMMARY_LOGDIR = './logdir'
# Input 
BATCH = 4
TRAIN_SIZE = 416
INPUT_SHAPE= (BATCH, TRAIN_SIZE, TRAIN_SIZE, 3)
MAX_INSTANCES = 20

ID_THRESH = 0.2
FG_THRESH = 0.2
BG_THRESH = 0.1

IOU_THRESH = 0.5
# Network
DATA_AUGMENTATION = True
NUM_CLASS = 1
MAX_BBOX_PER_SCALE = 20
import numpy as np

ANCHORS = [ 13,  41,  28,  82,  51, 104,  90, 117, 
           27,  82,  57, 165, 102, 209, 181, 235,
           41, 124,  86, 248, 154, 313, 272, 353,
           55, 165, 114, 331, 205, 418, 363, 470]

#
#np.array([4,12, 6,22, 8,37, 11,20, 
#           24,96, 29,64, 33,109, 38,85, 
#           91,227, 96,124, 111,182, 123,239, 
#           293,403, 333,488, 396,412, 438, 491])
#
#ANCHORS = ANCHORS/416*TRAIN_SIZE
#ANCHORS = np.array(ANCHORS, np.int32)

# [   4,   10,    6,   21,    9,   35,   13,   53,   30,   42,   39,
#          76,   40,  127,   50,  227,  114,  203,  134,  130,  134,  286,
#         174,  513,  366,  431,  447,  594,  569, 1122,  867, 1179]




#[ 13,  41,  28,  82,  51, 104,  90, 117,  27,  82,  57, 165, 102,
#       209, 181, 235,  41, 124,  86, 248, 154, 313, 272, 353,  55, 165,
#       114, 331, 205, 418, 363, 470]
#[  8,  25,  12,  45,  17,  74,  22,  41,  48, 193,  59, 129,  66,
#       219,  77, 172, 137, 341, 144, 186, 166, 274, 184, 358, 294, 403,
#       333, 488, 396, 412, 438, 491]
#[  4,  12,   6,  23,   8,  37,  10,  19,  
#           24,  97, 28,  58,  33, 81,  34, 112,  
#           85, 190,  96, 130, 99, 234, 118, 186, 
#           267, 474, 335, 384, 349, 486, 442, 480]
# [ 6,  18,  12,  37,  23,  58,  48,  55,  53,  73,  12,  36,  25,
#        74,  46, 116,  97, 111, 106, 146,  24,  72,  50, 148,  92, 233,
#       194, 223, 212, 293,  48, 145, 101, 296, 184, 467, 388, 446, 425,
#       586]#[4,6, 5,7, 8,18, 9,19,  8,24, 11,34, 16,48, 23,68,  32,96, 45,135, 64,192, 90,271,  128,384, 180,540, 256,640, 512,640 ]#[8,24, 11,34, 16,48,  23,68,32,96, 45,135,  64,192, 90,271,128,384,  180,540, 256,640, 512,640 ]#[5,4, 12,7, 11,15, 12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
##
NUM_ANCHORS = 4
BBOX_CLASS = 4
BBOX_REG = 1
MASK=0
STRIDES = [ 4, 8, 16, 32]
#ANCHORS =  [  7,   7,  14,  28,  28, 21,  35,  56, 
#            13,  13,  26,  52,  52,  39,  65, 104, 
#            26,  26,  52, 104, 104,  78, 130, 208, 
#            52,  52, 104, 208, 208, 156, 260, 416] #1,   1,   2,   4,   4,   3,   5,   8,
#XYSCALE= [1.4, 1.3, 1.2, 1.1, 1.05]
LEVELS = 4
EMB_DIM = TRAIN_SIZE//2
CSP_DARKNET53 = './weights/yolov4.weights'
MSDS_WEIGHTS = './MSDS_noemb_mask_14_-22.57891_2021-02-01-21-35-00.tf'
# Train
WD = 1e-4
LR = 1e-3
MOM = 0.9
GRADIENT_CLIP = 5.0
EPOCHS = 30

CONF_THRESH = 0.7
NMS_THRESH = 0.3
STEPS_PER_EPOCH_TRAIN = 1000
STEPS_PER_EPOCH_VAL = 100

RPN_NMS_THRESHOLD = 0.7
PRE_NMS_LIMIT = 1000
POST_NMS_ROIS_TRAINING = 300
POST_NMS_ROIS_INFERENCE = 100

MAX_PROP = 100

POOL_SIZE = 7
MASK_POOL_SIZE = POOL_SIZE*2
MASK_SIZE = MASK_POOL_SIZE*2 #28
