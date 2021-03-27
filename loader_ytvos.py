import os
import config as cfg
import numpy as np
import tensorflow as tf
from utils import file_reader, mask_clamp, read_image,\
        data_augment, data_preprocess, data_pad, data_check, xywh2xyxy,random_brightness
np.random.seed(41296)
from PIL import Image
class DataLoader(object):
    def __init__(self, batch_size = cfg.BATCH, shuffle=cfg.SHUFFLE, augment=cfg.DATA_AUGMENT):
        print('Dataset loading..')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.annotation, self.annotation_list, self.class_dict, self.nIDs = self.preprocess_json_dataset()
        self.train_list, self.val_list = self.split_dataset()
        self.train_ds = self.initilize_train_ds()
        self.val_ds = self.initilize_val_ds()
        print('Dataset loaded.')
        print('# identities:',self.nIDs)

    @classmethod
    def input_generator(cls, id_list):
        for idx in range(len(id_list)):
            yield id_list[idx]
    
    def preprocess_json_dataset(self):
        json_train_dataset = file_reader(cfg.YT_TRAIN_ANNOTATION_PATH)
        # json_val_dataset = file_reader(cfg.YT_VAL_ANNOTATION_PATH)
        # json_test_dataset = file_reader(cfg.YT_TEST_ANNOTATION_PATH)
        class_dict = json_train_dataset['categories']
        class_dict = { d['id']:d['name'] for d in class_dict }
        nIDs = self.count(json_train_dataset)
        annotations = self.parse_frames(json_train_dataset)
        # annotation_val = self.parse_frames(json_val_dataset)
        train_list = np.arange(len(annotations))
        nIDs = len(annotations)
        # val_list = np.arange(len(annotation_val))
        return annotations, train_list, class_dict, nIDs
    
    def count(self, json_dataset):
        nID = 0
        for video in json_dataset['annotations']:
            for bbox in video['bboxes']:
                if bbox!=None:
                    nID+=1
        return nID
    
    def parse_frames(self, json_dataset): #exclude frames with no subjects
        parsed_dataset = []
        for video in json_dataset['videos']:
            video_id = video['id']
            annotations = [notes for notes in json_dataset['annotations'] if notes['video_id'] == video_id]
            for i in range(video['length']):
                new_note = dict()
                new_note['video_id'] = video_id
                new_note['width'] = video['width']
                new_note['height'] = video['height']
                new_note['frame_num'] = i
                new_note['file_names'] = video['file_names'][i]
                new_note['segmentations'] = []
                new_note['bboxes'] = []
                new_note['areas'] = []
                new_note['category_id'] = []
                new_note['instance_ids'] = []
                for notes in annotations:                    
                    if notes['bboxes'][i] != None and notes['areas'][i] != None and notes['segmentations'][i] != None:
                        new_note['segmentations'].append(notes['segmentations'][i]['counts'].copy())
                        new_note['bboxes'].append(notes['bboxes'][i].copy())
                        new_note['areas'].append(notes['areas'][i])
                        new_note['category_id'].append(notes['category_id'])
                        new_note['instance_ids'].append(notes['id'])
                if len(new_note['segmentations'])>0:
                    parsed_dataset.append(new_note)
        return parsed_dataset
            
        
    def filter_inputs(self, image, gt_masks, gt_bboxes):
        return tf.greater(tf.reduce_sum(gt_bboxes[...,:4]), 0) and tf.greater(tf.reduce_sum(gt_masks), 0)
    
    def read_transform_train(self, idx):
        image, masks, bboxes = tf.py_function(self._single_input_generator_train, [idx], [tf.float32, tf.float32, tf.float32])
        return image, masks, bboxes

    def read_transform_val(self, idx):
        image, masks, bboxes = tf.py_function(self._single_input_generator_val, [idx], [tf.float32, tf.float32, tf.float32])
        return image, masks, bboxes
    
    def split_dataset(self):
        if self.shuffle:
            np.random.shuffle(self.annotation_list)
        divider =round(len(self.annotation_list)*cfg.SPLIT_RATIO)
        return self.annotation_list[:divider], self.annotation_list[divider:]

    def initilize_train_ds(self):
        if self.shuffle:
            np.random.shuffle(self.train_list)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
        ds = tf.data.Dataset.from_generator(self.input_generator, args=[self.train_list], output_types=(tf.int32)).repeat()
        ds = ds.map(self.read_transform_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.filter(self.filter_inputs).with_options(options)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds
    
    def initilize_val_ds(self):
        if self.shuffle:
            np.random.shuffle(self.val_list)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
        ds = tf.data.Dataset.from_generator(self.input_generator, args=[self.val_list], output_types=(tf.int32)).repeat()
        ds = ds.map(self.read_transform_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.filter(self.filter_inputs).with_options(options)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.take(cfg.STEPS_PER_EPOCH_VAL).prefetch(tf.data.experimental.AUTOTUNE)
        return ds
    
    def _single_input_generator_train(self, index):
        image, masks, bboxes, good_sample = self._data_generator(index)
        if good_sample:
            if self.augment:
                image, masks, bboxes = data_augment(image, masks, bboxes)
            image, masks, bboxes = data_preprocess(image, bboxes, masks)
            masks, bboxes = data_check(masks, bboxes)
            masks, bboxes = data_pad(masks, bboxes)
                
        return image, masks, bboxes
    
    def _single_input_generator_val(self, index):    
        image, masks, bboxes, good_sample = self._data_generator(index)
        if good_sample:
            image, masks, bboxes = data_preprocess(image, bboxes, masks)
            masks, bboxes = data_check(masks, bboxes)
            masks, bboxes = data_pad(masks, bboxes)
        return image, masks, bboxes
    
    def rle_decoding(self, rle_arr, w, h):
        rle_arr = np.cumsum(rle_arr)
        indices = []
        extend = indices.extend
        list(map(extend, map(lambda s,e: range(s, e), rle_arr[0::2], rle_arr[1::2])));
        mask = np.zeros(h*w, dtype=np.uint8)
        mask[indices] = 1
        return mask.reshape((w, h)).T

    # from itertools import groupby

    # def binary_mask_to_rle(binary_mask):
    #     rle = {'counts': [], 'size': list(binary_mask.shape)}
    #     counts = rle.get('counts')
    #     for i, (value, elements) in enumerate(groupby(binazip(rle_arr[0::2], rle_arr[1::2])ry_mask.ravel(order='F'))):
    #         if i == 0 and value == 1:
    #             counts.append(0)
    #         counts.append(len(list(elements)))
    #     return rle

    def _data_generator(self, index):
        sample = self.annotation[index]
        height = sample['height']
        width = sample['width']
        category_ids = sample['category_id']
        segmentations = sample['segmentations']
        boxes = sample['bboxes']
#        areas = sample['areas']
        file_name = sample['file_names']
        instance_ids = sample['instance_ids']
        path = os.path.join(cfg.YT_TRAIN_FRAMES_PATH, file_name)
        bboxes = []
        masks = [] 
        try:
            image = np.array(read_image(path))
            for segmentation, bbox, category_id, instance_id in zip(segmentations, boxes, category_ids, instance_ids):
                bbox = np.array(bbox)
                bbox = np.array(np.r_[bbox[:2],(bbox[:2]+bbox[2:4]),category_id,instance_id], dtype = np.int32)
                if bbox[2]>bbox[0] and bbox[3]>bbox[1]:
                    try:
                        mask = self.rle_decoding(segmentation, width, height)
                        if np.any(mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]):
                            bboxes.append(bbox)
                            masks.append(mask)
                    except:
                        continue
            bboxes = np.stack(bboxes,axis=0) # if bboxes = [] gives an exception --ok 
            masks = np.stack(masks,axis=0)
            good_sample = True
        except: # image not found or no valid bboxes or not valid masks, use light tipe to speed up
            image = np.zeros((cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3), dtype=np.uint8)
            bboxes = np.zeros((cfg.MAX_INSTANCES,6), dtype=np.uint8) # bbox + pid
            masks = np.zeros((cfg.MAX_INSTANCES, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE), dtype=np.uint8)
            good_sample = False
        return image, masks, bboxes, good_sample