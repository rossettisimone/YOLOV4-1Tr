# Paths
# VIDEOS_DATASET_PATH = "/home/temir/Documents/AVA/Dataset/Kin200/"
# ANNOTATION_PATH = "/home/temir/Documents/AVA/Dataset/prova.json"
#import numpy as np
GPU = '1'
DATASET_TYPE = 'ava'

# AVA_path 
VIDEOS_DATASET_PATH = "/media/data4/Datasets/Kinetics_AVA/frames/"
SEGMENTS_DATASET_PATH = "/media/data4/Datasets/Kinetics_AVA/segments"
ANNOTATION_PATH = "/media/data4/Datasets/Kinetics_AVA/ava_frames_boundings_train_v2.2.json" # #ava_frames_boundings_train_v2.2.json
SPLIT_RATIO = 0.7
SUMMARY_LOGDIR = './logdir'
# Input 
BATCH= 8
TRAIN_SIZE = 416#416
INPUT_SHAPE= (BATCH, TRAIN_SIZE, TRAIN_SIZE, 3)

# Network
DATA_AUGMENTATION = True
NUM_CLASS = 1
MAX_BBOX_PER_SCALE = 20 #150
ANCHORS = [ 13,  41,  28,  82,  51, 104,  90, 117,  27,  82,  57, 165, 102,
       209, 181, 235,  41, 124,  86, 248, 154, 313, 272, 353,  55, 165,
       114, 331, 205, 418, 363, 470]
# [  4,  12,   6,  23,   8,  37,  10,  19,  24,  97,  28,  58,  33,
#         81,  34, 112,  85, 190,  96, 130,  99, 234, 118, 186, 267, 474,
#        335, 384, 349, 486, 442, 480]
#[7,22, 14,45, 24,58, 37,67, 59,72, 15,45, 28,91, 48,116, 74,135, 118,145, 30,91, 57,183, 96,232, 149,270, 237,291, 61,183, 115,367, 192,464, 299,540, 475, 583]

NUM_ANCHORS = 4
BBOX_CLASS = 4
BBOX_REG = 1
MASK = 0
STRIDES = [ 4, 8, 16, 32] #2
#ANCHORS =  [  7,   7,  14,  28,  28, 21,  35,  56, 
#            13,  13,  26,  52,  52,  39,  65, 104, 
#            26,  26,  52, 104, 104,  78, 130, 208, 
#            52,  52, 104, 208, 208, 156, 260, 416] #1,   1,   2,   4,   4,   3,   5,   8,
#XYSCALE= [1.4, 1.3, 1.2, 1.1, 1.05]
LEVELS = 4 #5
EMB_DIM = 208
CSP_DARKNET53 = './yolov4.weights'
# Train
WD = 1e-4
LR = 1e-2
MOM = 0.9
EPOCHS = 30

SCORE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.5
IOU_LOSS_THRESH = 0.5
CONF_THRESH = 0.7
NMS_THRESH = 0.8
STEPS_PER_EPOCH = 1000


ALIGN_H, ALIGN_W = 7, 7