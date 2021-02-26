
import os
import config as cfg
import tensorflow as tf
import random
from PIL import Image
import numpy as np
from utils import file_reader, mask_clamp, read_image, encode_target
#import matplotlib.pyplot as plt

class Generator(object):
            
    def _single_input_generator_train(self, index):
        [video, frame_id] = self.annotation_train[index]
        image, masks, bboxes = self.data_generator(video, frame_id)
        if np.any(bboxes):
            if self.augment:
                image, masks, bboxes = self.data_augment(image, masks, bboxes)
            image, masks, bboxes = self.data_preprocess(image, bboxes, masks)
            mini_masks = self.masks_preprocess(masks, bboxes)
            masks, mini_masks, bboxes = self.data_pad(masks, mini_masks, bboxes)
        else:
            mini_masks = np.zeros((cfg.MAX_INSTANCES, cfg.MASK_SIZE, cfg.MASK_SIZE))
        return image, masks, mini_masks, bboxes
    
    def _single_input_generator_val(self, index):    
        [video, frame_id] = self.annotation_val[index]
        image, masks, bboxes = self.data_generator(video, frame_id)
        if np.any(bboxes):
            image, masks, bboxes = self.data_preprocess(image, bboxes, masks)
            mini_masks = self.masks_preprocess(masks, bboxes)
            masks, mini_masks, bboxes = self.data_pad(masks, mini_masks, bboxes)
        else:
            mini_masks = np.zeros((cfg.MAX_INSTANCES, cfg.MASK_SIZE, cfg.MASK_SIZE))
        return image, masks, mini_masks, bboxes
    
    def data_pad(self, masks, mini_masks, bboxes):
        bboxes_padded = np.zeros(( cfg.MAX_INSTANCES,5))
        bboxes_padded[:,4]=-1
        masks_padded = np.zeros(( cfg.MAX_INSTANCES,masks.shape[1],masks.shape[2]))
        mini_masks_padded = np.zeros(( cfg.MAX_INSTANCES,mini_masks.shape[1],mini_masks.shape[2]))
        #check consistency of bbox after data augmentation: dimension and ratio
        width = bboxes[...,2] - bboxes[...,0]
        height = bboxes[...,3] - bboxes[...,1]
        mask = (width > cfg.MIN_BOX_DIM*cfg.TRAIN_SIZE) \
            * (height > cfg.MIN_BOX_DIM*cfg.TRAIN_SIZE) \
            * ((width/height)>cfg.MIN_BOX_RATIO) \
            * ((height/width)>cfg.MIN_BOX_RATIO)
        bboxes = bboxes[mask]
        masks = masks[mask]
        mini_masks = mini_masks[mask]
        #zero pad
        min_bbox = min(bboxes.shape[0],  cfg.MAX_INSTANCES)
        bboxes_padded[:min_bbox,...]=bboxes[:min_bbox,...]
        masks_padded[:min_bbox,...]=masks[:min_bbox,...]
        mini_masks_padded[:min_bbox,...]=mini_masks[:min_bbox,...]
        return masks_padded, mini_masks_padded, bboxes_padded
        
    def masks_preprocess(self, masks, bboxes):
        masks_resized = []
        for mask, bbox in zip(masks, bboxes):
            try:
                mask = Image.fromarray(mask[bbox[1]:bbox[3],bbox[0]:bbox[2]])
            except:
                mask = Image.fromarray(mask)
            mask = mask.resize((cfg.MASK_SIZE,cfg.MASK_SIZE), Image.ANTIALIAS)
            mask = np.clip(mask,0,1)
            mask = np.round(mask)
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
        gt_boxes = np.clip(gt_boxes,0,cfg.TRAIN_SIZE-1)
        return image_paded, masks_padded, gt_boxes
        
    def data_generator(self, video, frame_id):
        frame_name = video['f_l'][frame_id]
        v_id =  video['v_id']
        path = os.path.join(cfg.VIDEOS_DATASET_PATH, v_id, frame_name+'.png' )
        bboxes = []
        masks = [] 
        try:
            image = np.array(read_image(path))
            height, width, _ = image.shape
            for person in video['p_l']:
                box = np.array(person["bb_l"][frame_id], dtype=np.float32)
                if box.shape[0] == 8: # Siammask returns 8 scalars (rotated bbox, rectify them)
                    xx, yy = box[::2], box[1::2]
                    box=np.array([np.min(xx),np.min(yy),np.max(xx),np.max(yy)])
                box=np.clip(box,0,1)
                p_id = person['p_id']
                box = np.array(np.r_[box,1]*np.array([width, height, width, height, p_id]), dtype = np.int32)
                if box[2]>box[0] and box[3]>box[1]:
                    try:
                        path_mask = os.path.join(cfg.SEGMENTS_DATASET_PATH, v_id, frame_name+'_'+str(person['p_id'])+'.png' )
                        mask = np.array(read_image(path_mask))
                        mask = mask_clamp(mask)
                        if np.any(mask[box[1]:box[3],box[0]:box[2]]):
                            bboxes.append(box)
                            masks.append(mask)
                    except:
                        pass
            bboxes = np.stack(bboxes,axis=0) # if bboxes = [] gives an exception --ok 
            masks = np.stack(masks,axis=0)
        except: # image not found or no valid bboxes or not valid masks
            image = np.zeros((cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3), dtype=np.uint8)
            bboxes = np.zeros((cfg.MAX_INSTANCES,5), dtype=np.int16) # bbox + pid
            masks = np.zeros((cfg.MAX_INSTANCES, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE), dtype=np.float16)
            
        return image, masks, bboxes

    def data_augment(self, image, masks, bboxes):
        image, masks, bboxes = self.random_horizontal_flip(image, masks, bboxes)
        image, masks, bboxes = self.random_crop(image, masks, bboxes)
        image, masks, bboxes = self.random_translate(image, masks, bboxes)
        return image, masks, bboxes
        
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
    def __init__(self, batch_size = cfg.BATCH, shuffle=True, augment=True):
        print('Dataset loading..')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
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
                except:
                    max_id_in_video[self.json_train_dataset[i]['v_id']] = 0
        self.nID = sum(max_id_in_video.values()) + len(max_id_in_video.values()) # id starts from zero, count them
        self.annotation_train = [(video,frame_id) for video in self.json_train_dataset \
                for frame_id in range(0,61) if not all(sum(p['bb_l'][frame_id])==0 for p in video['p_l'])]
        self.annotation_val = [(video,frame_id) for video in self.json_val_dataset \
                for frame_id in range(0,61) if not all(sum(p['bb_l'][frame_id])==0 for p in video['p_l'])]
        self.train_list = np.arange(len(self.annotation_train))
        self.val_list = np.arange(len(self.annotation_val))
        if self.shuffle:
            np.random.shuffle(self.train_list)
            np.random.shuffle(self.val_list)
        self.train_ds = self.initilize_train_ds()
        self.val_ds = self.initilize_val_ds()
        self.anchors = np.reshape(np.array(cfg.ANCHORS,dtype=np.int32),[cfg.LEVELS, cfg.NUM_ANCHORS, 2])
        self.train_input_size = np.array(cfg.TRAIN_SIZE,dtype=np.int32)
        self.strides = np.array(cfg.STRIDES,dtype=np.int32)
        self.train_output_sizes = self.train_input_size // self.strides
        print('Dataset loaded.')
        print('# identities:',self.nID)
            
    @classmethod
    def input_generator(cls, id_list):
        for idx in range(len(id_list)):
            yield id_list[idx]
    
    def filter_inputs(self, image, gt_mask, gt_masks, gt_bboxes):
        return tf.greater(tf.reduce_sum(gt_bboxes[...,:4]), 0) and tf.greater(tf.reduce_sum(gt_masks), 0)

    def read_transform_train(self, idx):
        image, mask, masks, bboxes = tf.py_function(self._single_input_generator_train, [idx], [tf.float32, tf.float32, tf.float32, tf.float32])
        return image, mask, masks, bboxes

    def read_transform_val(self, idx):
        image, mask, masks, bboxes = tf.py_function(self._single_input_generator_val, [idx], [tf.float32, tf.float32, tf.float32, tf.float32])
        return image, mask, masks, bboxes

    def generator_train(self, *args):
        return tf.data.Dataset.from_generator(DataLoader.input_generator, args= [self.train_list], output_types= (tf.int32))
    
    def generator_val(self, *args):
        return tf.data.Dataset.from_generator(DataLoader.input_generator, args= [self.val_list], output_types= (tf.int32))
    
    def initilize_train_ds(self):
        ds = tf.data.Dataset.range(1).interleave(self.generator_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.map(self.read_transform_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.filter(self.filter_inputs)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.repeat().prefetch(tf.data.experimental.AUTOTUNE)
        return ds
    
    def initilize_val_ds(self):
        ds = tf.data.Dataset.range(1).interleave(self.generator_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.map(self.read_transform_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.filter(self.filter_inputs)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds