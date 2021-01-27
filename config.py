# Paths
# VIDEOS_DATASET_PATH = "/home/temir/Documents/AVA/Dataset/Kin200/"
# ANNOTATION_PATH = "/home/temir/Documents/AVA/Dataset/prova.json"
#import numpy as np
GPU = '0'
DATASET_TYPE = 'ava'

# AVA_path 
VIDEOS_DATASET_PATH = "/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/frames"
SEGMENTS_DATASET_PATH = "/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/segments"
ANNOTATION_PATH = "/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/kinetics_100_frames_boundings_train_v1.0.json" # #ava_frames_boundings_train_v2.2.json
SPLIT_RATIO = 0.7
SUMMARY_LOGDIR = './logdir'
# Input 
BATCH = 8
TRAIN_SIZE = 416#416
INPUT_SHAPE= (BATCH, TRAIN_SIZE, TRAIN_SIZE, 3)
MAX_INSTANCES = 20

ID_THRESH = 0.5
FG_THRESH = 0.5
BG_THRESH = 0.3

# Network
DATA_AUGMENTATION = True
NUM_CLASS = 1
MAX_BBOX_PER_SCALE = 20 #150
ANCHORS = [ 13,  41,  28,  82,  51, 104,  90, 117, 
           27,  82,  57, 165, 102, 209, 181, 235,
           41, 124,  86, 248, 154, 313, 272, 353,
           55, 165, 114, 331, 205, 418, 363, 470]
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
BBOX_CLASS=4
BBOX_REG=1
MASK=0
STRIDES = [ 4, 8, 16, 32] #2
#ANCHORS =  [  7,   7,  14,  28,  28, 21,  35,  56, 
#            13,  13,  26,  52,  52,  39,  65, 104, 
#            26,  26,  52, 104, 104,  78, 130, 208, 
#            52,  52, 104, 208, 208, 156, 260, 416] #1,   1,   2,   4,   4,   3,   5,   8,
#XYSCALE= [1.4, 1.3, 1.2, 1.1, 1.05]
LEVELS = 4 #5
EMB_DIM = 208
CSP_DARKNET53 = './weights/yolov4.weights'
MSDS_WEIGHTS = './tracker_weights_10.tf'
# Train
WD = 1e-4
LR = 1e-2
MOM = 0.9
EPOCHS = 30

CONF_THRESH = 0.5
NMS_THRESH = 0.3
STEPS_PER_EPOCH_TRAIN = 5000
STEPS_PER_EPOCH_VAL = 500

MAX_PROP = 20

POOL_SIZE = 7
MASK_POOL_SIZE = 14
MASK_SIZE = 28


TRACK_BUFFER = 30