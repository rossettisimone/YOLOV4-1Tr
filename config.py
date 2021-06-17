GPU = '0'
#GPU = '0,1,2,3'

DATASET_TYPE = 'vis'
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
#
#YT_TRAIN_ANNOTATION_PATH = "/home/fiorapirri/Documents/workspace/YoutubeVOS19/train_all_frames/train.json"
#YT_TRAIN_FRAMES_PATH = "/home/fiorapirri/Documents/workspace/YoutubeVOS19/train_all_frames/JPEGImages/"
#YT_VAL_ANNOTATION_PATH = "/home/fiorapirri/Documents/workspace/YoutubeVOS19/valid_all_frames/valid.json"
#YT_VAL_FRAMES_PATH = "/home/fiorapirri/Documents/workspace/YoutubeVOS19/valid_all_frames/JPEGImages/"
#YT_TEST_ANNOTATION_PATH = "/home/fiorapirri/Documents/workspace/YoutubeVOS19/test_all_frames/test.json"
#YT_TEST_FRAMES_PATH = "/home/fiorapirri/Documents/workspace/YoutubeVOS19/test_all_frames/JPEGImages/"
# YT_TRAIN_ANNOTATION_PATH = "/home/fiorapirri/Documents/workspace/train_all_frames/train.json"
# YT_TRAIN_FRAMES_PATH = "/home/fiorapirri/Documents/workspace/train_all_frames/JPEGImages/"
# YT_VAL_ANNOTATION_PATH = "/media/data3/YoutubeVOS19/valid_all_frames/valid.json"
# YT_VAL_FRAMES_PATH = "/media/data3/YoutubeVOS19/valid_all_frames/JPEGImages/"
# YT_TEST_ANNOTATION_PATH = "/media/data3/YoutubeVOS19/test_all_frames/test.json"
# YT_TEST_FRAMES_PATH = "/media/data3/YoutubeVOS19/test_all_frames/JPEGImages/"
YT_TRAIN_ANNOTATION_PATH = "/home/fiorapirri/Documents/workspace/YouTubeVIS21/train/instances.json"
YT_TRAIN_FRAMES_PATH = "/home/fiorapirri/Documents/workspace/YouTubeVIS21/train/JPEGImages/"

YT_VALID_ANNOTATION_PATH = "/home/fiorapirri/Documents/workspace/YouTubeVIS21/valid/instances.json"
YT_VALID_FRAMES_PATH = "/home/fiorapirri/Documents/workspace/YouTubeVIS21/valid/JPEGImages/"

YT_TEST_ANNOTATION_PATH = "/home/fiorapirri/Documents/workspace/YouTubeVIS21/test/instances.json"
YT_TEST_FRAMES_PATH = "/home/fiorapirri/Documents/workspace/YouTubeVIS21/test/JPEGImages/"

SPLIT_RATIO = .1
SHUFFLE = True
DATA_AUGMENT = True
LOGDIR = 'logdir'
WEIGHTS = 'weights'
MIN_BOX_DIM = 5e-4
MIN_BOX_RATIO = 1e-3

# Input 
BATCH = 2
TRAIN_SIZE = 416
INPUT_SHAPE= (TRAIN_SIZE, TRAIN_SIZE, 3)
MAX_INSTANCES = 15

ID_THRESH = 0.4
FG_THRESH = 0.4
BG_THRESH = 0.3
TOLERANCE = [0.6,0.8,1.0,1.2]

IOU_THRESH = 0.5
# Network
NUM_CLASSES = 40

#ANCHORS = [2,2, 4,7, 6,14, 8,3, 
#           20,15, 22,41, 25,24, 34,48, 
#           69,61, 81,34, 100,102, 101,66, 
#           268,219, 305,84, 314,154, 378, 222]

#[  3,  24,  14,  25,  24,  25,   8,  37,  41,  71,  68,  49,  58,
#        64,  81,  58, 144, 145, 115, 164, 173, 142, 153, 170, 366, 326,
#       345, 374, 385, 364, 395, 396]

#ANCHORS = [  2,   2,   4,   4,   5,   6,   7,   7,   9,   8,  14,  11,
#           15,  19,  20,  24,  22,  25,  27,  33,  33,  35,  47,  45,  
#           56,  48, 72,  63,  86,  67, 106,  89, 113, 91, 143, 113,
#         184, 122, 253, 168, 287, 220, 323, 223, 376, 228, 398, 228]
#ANCHORS = [1,1, 1,2, 3,1, 3,3, 2,5, 8,2, 3,7, 5,4, 3,11, 5,8, 
#           22,8, 16,12, 11,27, 16,20, 23,15, 38,11, 22,23, 17,30, 13,39, 31,20, 
#           44,71, 57,57, 99,39, 40,97, 79,53, 55,84, 70,70, 173,31, 96,66, 61,110, 
#           273,102, 160,178, 247,144, 212,184, 179,229, 354,143, 289,180, 242,227, 312,230, 394, 220]
#

ANCHORS = [2, 2, 2, 4, 6, 2, 6, 6, 4, 10, 16, 4, 6, 14, 10, 8, 6, 22, 10, 16, 
           22,8, 16,12, 11,27, 16,20, 23,15, 38,11, 22,23, 17,30, 13,39, 31,20, 
           44,71, 57,57, 99,39, 40,97, 79,53, 55,84, 70,70, 173,31, 96,66, 61,110, 
           273,102, 160,178, 247,144, 212,184, 179,229, 354,143, 289,180, 242,227, 312,230, 394, 220]


#ANCHORS_5 = [  3,   2,   4,   5,   5,   6,   9,  10,  11,  11,  
#             15,  16,  21,  29,  29,  30,  34,  32, 42,  43,  
#             56,  45,  74,  78,  94,  84, 108, 110, 115, 110, 
#             252, 112, 288, 152, 321, 219, 368, 220, 398, 222]
#
#ANCHORS = [  1,   1,   5,   2,   2,   5,   6,   6,   4,  11,  
#              24,   9,  18,  18,  14,  32,  35,  17,  26,  26, 
#              42,  83, 89,  54,  68,  72, 144,  42,  59, 104,
#              217, 148, 178, 204, 322, 153, 254, 217, 367, 223]
#AnchorsFPN =  [  2 ,  2,   3,   5,   7,   3,   5,   9,  
#                24,  13,  19,  22,  13,  33,  31,  25,  
#                44,  80, 118,  39,  91,  66,  65,  95, 
#                194, 203, 300, 144, 264, 216, 370, 219]
#Anchors = [  3,   3,   4,   6,   6,   7,  10,  12,  
#           18,  19,  26,  22,  34,  35,  42,  36,  
#           50,  60,  85,  71,  99, 102, 131, 105, 
#           236, 122, 302, 209, 360, 213, 391, 220]
# [ 13,  41,  28,  82,  51, 104,  90, 117, 
#           27,  82,  57, 165, 102, 209, 181, 235,
#           41, 124,  86, 248, 154, 313, 272, 353,
#           55, 165, 114, 331, 205, 418, 363, 470]

# [ 12,  26,  21,  44,  29,  76,  39, 109,
#            53,  61,  62, 179,  66, 103,  83, 226,
#            111, 187, 136, 298, 138,  99, 167, 193, 
#            239, 376, 322, 414, 349, 256, 424, 423]

#import numpy as np
#ANCHORS = np.array(np.array(ANCHORS)/416*TRAIN_SIZE,np.int32) # if want to test different input  size

NUM_ANCHORS = 10
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

CONF_THRESH = 0.5
NMS_THRESH = 0.3
STEPS_PER_EPOCH_TRAIN = 10
STEPS_PER_EPOCH_VAL = 20

PRE_NMS_LIMIT = 200
MAX_PROP = 50
MAX_PROP_PER_CLASS = 20

TOP_DOWN_PYRAMID_SIZE = 256
MASK_LAYERS_SIZE = 256
FC_LAYER_SIZE = 1024
POOL_SIZE = 7
MASK_POOL_SIZE = POOL_SIZE*2
MASK_SIZE = MASK_POOL_SIZE*2 #28
MASK_CONF = 1

CLASS_YTVIS19 = {1: 'person',
 2: 'giant_panda',
 3: 'lizard',
 4: 'parrot',
 5: 'skateboard',
 6: 'sedan',
 7: 'ape',
 8: 'dog',
 9: 'snake',
 10: 'monkey',
 11: 'hand',
 12: 'rabbit',
 13: 'duck',
 14: 'cat',
 15: 'cow',
 16: 'fish',
 17: 'train',
 18: 'horse',
 19: 'turtle',
 20: 'bear',
 21: 'motorbike',
 22: 'giraffe',
 23: 'leopard',
 24: 'fox',
 25: 'deer',
 26: 'owl',
 27: 'surfboard',
 28: 'airplane',
 29: 'truck',
 30: 'zebra',
 31: 'tiger',
 32: 'elephant',
 33: 'snowboard',
 34: 'boat',
 35: 'shark',
 36: 'mouse',
 37: 'frog',
 38: 'eagle',
 39: 'earless_seal',
 40: 'tennis_racket'}

CLASS_YTVIS21 = {1: 'airplane',
 2: 'bear',
 3: 'bird',
 4: 'boat',
 5: 'car',
 6: 'cat',
 7: 'cow',
 8: 'deer',
 9: 'dog',
 10: 'duck',
 11: 'earless_seal',
 12: 'elephant',
 13: 'fish',
 14: 'flying_disc',
 15: 'fox',
 16: 'frog',
 17: 'giant_panda',
 18: 'giraffe',
 19: 'horse',
 20: 'leopard',
 21: 'lizard',
 22: 'monkey',
 23: 'motorbike',
 24: 'mouse',
 25: 'parrot',
 26: 'person',
 27: 'rabbit',
 28: 'shark',
 29: 'skateboard',
 30: 'snake',
 31: 'snowboard',
 32: 'squirrel',
 33: 'surfboard',
 34: 'tennis_racket',
 35: 'tiger',
 36: 'train',
 37: 'truck',
 38: 'turtle',
 39: 'whale',
 40: 'zebra'}