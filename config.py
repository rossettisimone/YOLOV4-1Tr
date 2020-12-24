# Paths
# VIDEOS_DATASET_PATH = "/home/temir/Documents/AVA/Dataset/Kin200/"
# ANNOTATION_PATH = "/home/temir/Documents/AVA/Dataset/prova.json"
#import numpy as np
GPU = '0'
# AVA_path 
VIDEOS_DATASET_PATH = "/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/frames"
SEGMENTS_DATASET_PATH = "/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/segments"
ANNOTATION_PATH = "/home/fiorapirri/Documents/workspace/ava_kinetics_v1_0/dataset/kinetics_100_frames_boundings_train_v1.0.json" # #ava_frames_boundings_train_v2.2.json
SPLIT_RATIO = 0.7
SUMMARY_LOGDIR = './logdir'
# Input 
BATCH= 5
TRAIN_SIZE = 416
INPUT_SHAPE= (BATCH, TRAIN_SIZE, TRAIN_SIZE, 3)

# Network
DATA_AUGMENTATION = True
NUM_CLASS = 1
MAX_BBOX_PER_SCALE = 20 #150
ANCHORS = [4,5, 6,12, 13,9, 12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]

NUM_ANCHORS = 3
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
CSP_DARKNET53 = './yolov4.weights'
# Train
WD = 1e-4
LR = 1e-2
MOM = 0.9
EPOCHS = 30

SCORE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
IOU_LOSS_THRESH = 0.5
CONF_THRESH = 0.3
NMS_THRESH = 0.3