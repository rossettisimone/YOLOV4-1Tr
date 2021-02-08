
import os
import config as cfg

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cfg.GPU

import tensorflow as tf

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

import random
from PIL import Image
import numpy as np
from utils import file_reader, mask_clamp, read_image, encode_target
#import matplotlib.pyplot as plt

class Generator(object):
            
    def _single_input_generator_train(self, index):
        [video, frame_id] = self.annotation_train[index]
        image, masks, bboxes = self.data_generator(video, frame_id)
        if self.data_aug:
            image, masks, bboxes = self.data_augment(image, masks, bboxes)
        image, masks, bboxes = self.data_preprocess(image, bboxes, masks)
        label_2, label_3, label_4, label_5 = self.data_labels(bboxes)
        masks = self.masks_preprocess(masks, bboxes)
        masks, bboxes = self.data_pad(masks, bboxes)
        inputs = [image, label_2, label_3, label_4, label_5, masks, bboxes]
        try:
            for input_ in inputs:
                assert np.isfinite(input_).all(), 'not finite values in inputs'
        except:
            for index, input_ in enumerate(inputs):
                inputs[index] = np.zeros_like(input_)
        return inputs
    
    def _single_input_generator_val(self, index):
        image, masks, bboxes = self.data_generator(video, frame_id)
        image, masks, bboxes = self.data_preprocess(image, bboxes, masks)
        label_2, label_3, label_4, label_5 = self.data_labels(bboxes)
        masks = self.masks_preprocess(masks, bboxes)
        masks, bboxes = self.data_pad(masks, bboxes)
        inputs = [image, label_2, label_3, label_4, label_5, masks, bboxes]
        try:
            for input_ in inputs:
                assert np.isfinite(input_).all(), 'not finite values in inputs'
        except:
            for index, input_ in enumerate(inputs):
                inputs[index] = np.zeros_like(input_)
        return inputs
    
    def data_pad(self, masks, bboxes):
        bboxes_padded = np.zeros(( cfg.MAX_INSTANCES,5))
        bboxes_padded[:,4]=-1
        masks_padded = np.zeros(( cfg.MAX_INSTANCES,masks.shape[1],masks.shape[2]))
        #check consistency of bbox after data augmentation: dimension and ratio
        width = bboxes[...,2] - bboxes[...,0]
        height = bboxes[...,3] - bboxes[...,1]
        mask = (width > cfg.MIN_BOX_DIM) \
            * (height > cfg.MIN_BOX_DIM) \
            * ((width/height)>cfg.MIN_BOX_RATIO) \
            * ((height/width)>cfg.MIN_BOX_RATIO)
        bboxes = bboxes[mask]
        masks = masks[mask]
        #zero pad
        min_bbox = min(bboxes.shape[0],  cfg.MAX_INSTANCES)
        bboxes_padded[:min_bbox,...]=bboxes[:min_bbox,...]
        min_mask = min(masks.shape[0],  cfg.MAX_INSTANCES)
        masks_padded[:min_mask,...]=masks[:min_mask,...]
        return masks_padded, bboxes_padded
        
    def masks_preprocess(self, masks, bboxes):
        masks_resized = []
        for mask, bbox in zip(masks, bboxes):
            mask = Image.fromarray(mask[bbox[1]:bbox[3],bbox[0]:bbox[2]])
            mask = np.round(mask.resize((cfg.MASK_SIZE,cfg.MASK_SIZE), Image.ANTIALIAS))
            mask = np.clip(mask,0,1)
            masks_resized.append(mask)
        masks_resized = np.stack(masks_resized,axis=0)
        return masks_resized
        
    def data_preprocess(self, image, gt_boxes, masks):
        ih, iw    = (cfg.TRAIN_SIZE, cfg.TRAIN_SIZE)
        h,  w, _  = image.shape
        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)    
        image = Image.fromarray(image)
        image_resized = image.resize((nw, nh),Image.ANTIALIAS)
        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        image_paded = image_paded / 255.
        image_paded=np.clip(image_paded,0,1)
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        masks_padded = np.zeros((masks.shape[0],iw, ih))
        for i, maskk in enumerate(masks):
            mask = Image.fromarray(maskk)
            mask_resized = np.round(mask.resize((nw, nh),Image.ANTIALIAS))
            masks_padded[i, dh:nh+dh, dw:nw+dw] = mask_resized
        gt_boxes = np.array(gt_boxes,dtype=np.uint32)
        return image_paded, masks_padded, gt_boxes
        
    def data_generator(self, video, frame_id):
        frame_name = video['f_l'][frame_id]
        v_id =  video['v_id']
        path = os.path.join(cfg.VIDEOS_DATASET_PATH, v_id, frame_name+'.png' )
        bboxes = []
        masks = [] 
        try:
            image = np.array(read_image(path))
            # assert np.isfinite(image).all(), 'nan values in image'
            height, width, _ = image.shape
            for person in video['p_l']:
                try:
                    box = person["bb_l"][frame_id]
                    # assert np.isfinite(box).all(), 'nan values in box'
                    if len(box)==8: # Siammask returns 8 scalars (rotated bbox, rectify them)
                        xx, yy = [s for i,s in enumerate(box) if i%2==0 ], [s for i,s in enumerate(box) if not i%2==0]
                        box=np.array([min(xx),min(yy),max(xx),max(yy)])
                    box=np.clip(box,0,1)
                    try:
                        p_id = person['p_id']
                        # assert np.isfinite(p_id), 'nan values in p_id'
                    except:
                        p_id = 0
                    box = np.r_[box,1]*np.array([width, height, width, height, p_id]) 
                    bboxes.append(box)
                    try:
                        path_mask = os.path.join(cfg.SEGMENTS_DATASET_PATH, v_id, frame_name+'_'+str(person['p_id'])+'.png' )
                        mask = mask_clamp(np.array(read_image(path_mask)))
                        # assert np.isfinite(mask).all(), 'nan values in mask'
                    except:
                        mask = np.array(Image.new('L', (width, height), color=0))
                    masks.append(mask)
                except:
                    bboxes.append(np.zeros((1,5), dtype=np.float32))
                    masks.append(np.zeros((width, height), dtype=np.float32))
            bboxes = np.stack(bboxes,axis=0)
            masks = np.stack(masks,axis=0)
        except:
            image = np.array(Image.new('RGB', (cfg.TRAIN_SIZE, cfg.TRAIN_SIZE), color = (0, 0, 0)))
            bboxes = np.zeros((cfg.MAX_INSTANCES,5), dtype=np.float32) # bbox + pid
            masks = np.zeros((cfg.MAX_INSTANCES, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE), dtype=np.float32)
        return image, masks, bboxes

    def data_augment(self, image, masks, bboxes):
        image, masks, bboxes = self.random_horizontal_flip(image, masks, bboxes)
        image, masks, bboxes = self.random_crop(image, masks, bboxes)
        image, masks, bboxes = self.random_translate(image, masks, bboxes)
        return image, masks, bboxes
    
    def data_labels(self, bboxes):
        labels = []
        for level in range(cfg.LEVELS):
            label = encode_target(bboxes, self.anchors[level]/self.strides[level], cfg.NUM_ANCHORS, self.num_classes, self.train_output_sizes[level], self.train_output_sizes[level])
            labels.append(label)
        return labels
        
    def random_horizontal_flip(self, image, masks, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            masks = masks[:,:,::-1]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        return image, masks, bboxes

    def random_crop(self, image, masks, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]
            crop_xmin = max(
                0, int(max_bbox[0] - random.uniform(0, max_l_trans))
            )
            crop_ymin = max(
                0, int(max_bbox[1] - random.uniform(0, max_u_trans))
            )
            crop_xmax = max(
                w, int(max_bbox[2] + random.uniform(0, max_r_trans))
            )
            crop_ymax = max(
                h, int(max_bbox[3] + random.uniform(0, max_d_trans))
            )
            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            masks = masks[:,crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            bboxes[:, [0, 2]] -= crop_xmin
            bboxes[:, [1, 3]] -= crop_ymin

        return image, masks, bboxes
    
    def random_translate(self, image, masks, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]
            tx = int(random.uniform(-(max_l_trans - 1), (max_r_trans - 1)))
            ty = int(random.uniform(-(max_u_trans - 1), (max_d_trans - 1)))
            old_image = np.copy(image)
            old_masks = np.copy(masks)
            image = np.zeros_like(image)
            masks = np.zeros_like(masks)
            image[max(0,ty):min(h,h+ty),max(0,tx):min(w,w+tx),:] = old_image[max(0,-ty):min(h,h-ty),max(0,-tx):min(w,w-tx),:]
            masks[:,max(0,ty):min(h,h+ty),max(0,tx):min(w,w+tx)] = old_masks[:,max(0,-ty):min(h,h-ty),max(0,-tx):min(w,w-tx)]
            bboxes[:, [0, 2]] += tx
            bboxes[:, [1, 3]] += ty
            
        return image, masks, bboxes
    
class DataLoader(Generator):
    def __init__(self, shuffle=True, data_aug=True):
        print('Dataset loading..')
        self.shuffle = shuffle
        self.data_aug = data_aug
        self.json_train_dataset = []
        for file_path in cfg.TRAIN_ANNOTATION_PATH:
            self.json_train_dataset += file_reader(file_path)
            print('Train Dataset {} loaded'.format(file_path))
        self.json_val_dataset = []
        for file_path in cfg.VAL_ANNOTATION_PATH:
            self.json_val_dataset += file_reader(file_path)
            print('Validation Dataset {} loaded'.format(file_path))
        max_id_in_video = {}
        for i,k in enumerate(self.json_train_dataset):
            for j,_ in enumerate(k['p_l']):
                try:
                    if self.json_train_dataset[i]['p_l'][j]['p_id'] > max_id_in_video[self.json_train_dataset[i]['v_id']]:
                        max_id_in_video[self.json_train_dataset[i]['v_id']] = self.json_train_dataset[i]['p_l'][j]['p_id'] 
                except KeyError as e:
                    max_id_in_video[self.json_train_dataset[i]['v_id']] = 0
        self.nID = sum(max_id_in_video.values()) + len(max_id_in_video.values()) # id starts from zero, count them
        #self.annotation = [(video,frame_id) for video in self.json_train_dataset for frame_id in range(0,61) if not all(p['bb_l'][frame_id]==[0,0,0,0] for p in video['p_l'])] # (video,0),(video,10),..,(video,60) sample each 10 frames
        #self.train_list, self.val_list = self.split_dataset(len(self.annotation_train))
        self.annotation_train = [(video,frame_id) for video in self.json_train_dataset \
                for frame_id in range(0,61) if not all(sum(p['bb_l'][frame_id])==0 for p in video['p_l'])]
        self.annotation_val = [(video,frame_id) for video in self.json_val_dataset \
                for frame_id in range(0,61) if not all(sum(p['bb_l'][frame_id])==0 for p in video['p_l'])]
        self.train_list = np.arange(len(self.annotation_train))
        self.val_list = np.arange(len(self.annotation_val))
        if self.shuffle:
            np.random.shuffle(self.train_list)
            np.random.shuffle(self.val_list)
        self.train_ds = self.initilize_train_ds(self.train_list)
        self.val_ds = self.initilize_val_ds(self.val_list)
        self.num_classes = cfg.NUM_CLASS
        self.anchors = tf.reshape(tf.constant(cfg.ANCHORS,dtype=tf.float32),[cfg.LEVELS, cfg.NUM_ANCHORS, 2])
        self.train_input_size = cfg.TRAIN_SIZE
        self.strides = tf.cast(cfg.STRIDES,tf.float32)
        self.train_output_sizes = self.train_input_size // self.strides
        self.max_bbox_per_scale = cfg.MAX_BBOX_PER_SCALE
        print('Dataset loaded.')
        print('# identities:',self.nID)
            
    def split_dataset(self, ds_size):
        total_list = np.arange(ds_size)
        if self.shuffle:
            np.random.shuffle(total_list)
        divider =round(ds_size*cfg.SPLIT_RATIO)
        return total_list[:divider], total_list[divider:]

    @classmethod
    def input_generator(cls, id_list):
        for idx in range(len(id_list)):
            yield id_list[idx]
    
    def read_transform_train(self, idx):
        image, label_2, label_3, label_4, label_5, masks, bboxes = tf.py_function(self._single_input_generator_train, [idx], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
        return image, label_2, label_3, label_4, label_5, masks, bboxes

    def read_transform_val(self, idx):
        image, label_2, label_3, label_4, label_5, masks, bboxes = tf.py_function(self._single_input_generator_val, [idx], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
        return image, label_2, label_3, label_4, label_5, masks, bboxes

    def initilize_train_ds(self, list_ids):
        ds = tf.data.Dataset.from_generator(DataLoader.input_generator , args= [list_ids], output_types= (tf.int32))
        ds = ds.map(self.read_transform_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds
    
    def initilize_val_ds(self, list_ids):
        ds = tf.data.Dataset.from_generator(DataLoader.input_generator , args= [list_ids], output_types= (tf.int32))
        ds = ds.map(self.read_transform_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds