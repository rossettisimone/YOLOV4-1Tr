import os
import config as cfg
import numpy as np
import tensorflow as tf
from utils import file_reader, mask_clamp, read_image,\
        data_augment, data_preprocess, data_pad, data_check
np.random.seed(41296)
    
class DataLoader(object):
    def __init__(self, batch_size = cfg.BATCH, shuffle=cfg.SHUFFLE, augment=cfg.DATA_AUGMENT):
        print('Dataset loading..')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.annotation_train, self.annotation_val, \
            self.train_list, self.val_list, self.nIDs = self.preprocess_json_dataset()
        self.train_ds = self.initilize_train_ds()
        self.val_ds = self.initilize_val_ds()
        print('Dataset loaded.')
        print('# identities:',self.nIDs)

    @classmethod
    def input_generator(cls, id_list):
        for idx in range(len(id_list)):
            yield id_list[idx]
    
    def preprocess_json_dataset(self):
        json_train_dataset = self.read_json_list(cfg.AVA_TRAIN_ANNOTATION_PATH)
        json_val_dataset = self.read_json_list(cfg.AVA_VAL_ANNOTATION_PATH)
        nIDs = self.count_ids(json_train_dataset)
        annotation_train = self.parse_frames(json_train_dataset)
        annotation_val = self.parse_frames(json_val_dataset)
        train_list = np.arange(len(annotation_train))
        val_list = np.arange(len(annotation_val))
        return annotation_train, annotation_val, train_list, val_list, nIDs
        
    def read_json_list(self, json_list):
        json_dataset = []
        for file_path in json_list:
            json_dataset += file_reader(file_path)
            print('Dataset {} loaded'.format(file_path))
        return json_dataset
    
    def count_ids(self, json_dataset):
        max_id_in_video = {}
        for i,k in enumerate(json_dataset):
            for j,_ in enumerate(k['p_l']):
                try:
                    if json_dataset[i]['p_l'][j]['p_id'] > max_id_in_video[json_dataset[i]['v_id']]:
                        max_id_in_video[json_dataset[i]['v_id']] = json_dataset[i]['p_l'][j]['p_id'] 
                except:
                    max_id_in_video[json_dataset[i]['v_id']] = 0
        return sum(max_id_in_video.values()) + len(max_id_in_video.values()) # id starts from zero, count them
    
    def parse_frames(self, json_dataset): #exclude frames with no subjects
        return [(video,frame_id) for video in json_dataset \
                for frame_id in range(0,61) if not all(sum(p['bb_l'][frame_id])==0 for p in video['p_l'])]
        
    def filter_inputs(self, image, gt_masks, gt_bboxes):
        return tf.greater(tf.reduce_sum(gt_bboxes[...,:4]), 0) and tf.greater(tf.reduce_sum(gt_masks), 0)

    def read_transform_train(self, idx):
        image, masks, bboxes = tf.py_function(self._single_input_generator_train, [idx], [tf.float32, tf.float32, tf.float32])
        return image, masks, bboxes

    def read_transform_val(self, idx):
        image, masks, bboxes = tf.py_function(self._single_input_generator_val, [idx], [tf.float32, tf.float32, tf.float32])
        return image, masks, bboxes
    
    def initilize_train_ds(self):
        if self.shuffle:
            np.random.shuffle(self.train_list)
        ds = tf.data.Dataset.from_generator(self.input_generator, args=[self.train_list], output_types=(tf.int32))
        ds = ds.map(self.read_transform_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.filter(self.filter_inputs)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.repeat().prefetch(tf.data.experimental.AUTOTUNE)
        return ds
    
    def initilize_val_ds(self):
        if self.shuffle:
            np.random.shuffle(self.val_list)
        ds = tf.data.Dataset.from_generator(self.input_generator, args=[self.val_list], output_types=(tf.int32))
        ds = ds.map(self.read_transform_val, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.filter(self.filter_inputs)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.take(cfg.STEPS_PER_EPOCH_VAL).repeat().prefetch(tf.data.experimental.AUTOTUNE)
        return ds
    
    def _single_input_generator_train(self, index):
        [video, frame_id] = self.annotation_train[index]
        image, masks, bboxes, good_sample = self._data_generator(video, frame_id)
        if good_sample:
            if self.augment:
                image, masks, bboxes = data_augment(image, masks, bboxes)
            image, masks, bboxes = data_preprocess(image, bboxes, masks)
            masks, bboxes = data_check(masks, bboxes)
            masks, bboxes = data_pad(masks, bboxes)
        return image, masks, bboxes
    
    def _single_input_generator_val(self, index):    
        [video, frame_id] = self.annotation_val[index]
        image, masks, bboxes, good_sample = self._data_generator(video, frame_id)
        if good_sample:
            image, masks, bboxes = data_preprocess(image, bboxes, masks)
            masks, bboxes = data_check(masks, bboxes)
            masks, bboxes = data_pad(masks, bboxes)
        return image, masks, bboxes
    
    def _data_generator(self, video, frame_id):
        frame_name = video['f_l'][frame_id]
        v_id =  video['v_id']
        path = os.path.join(cfg.AVA_VIDEOS_DATASET_PATH, v_id, frame_name+'.png' )
        bboxes = []
        masks = [] 
        try:
            image = np.array(read_image(path))
            height, width, _ = image.shape
            for person in video['p_l']:
                box = np.array(person["bb_l"][frame_id], dtype=np.float32)
                xx, yy = box[::2], box[1::2] # Siammask returns 8 scalars (rotated bbox, rectify them)
                box=np.array([np.min(xx),np.min(yy),np.max(xx),np.max(yy)])
                box=np.clip(box,0.0,1.0)
                p_id = person['p_id']
                box = np.array(np.r_[box,1]*np.array([width, height, width, height, p_id]), dtype = np.int32)
                if box[2]>box[0] and box[3]>box[1]:
                    try:
                        path_mask = os.path.join(cfg.AVA_SEGMENTS_DATASET_PATH, v_id, frame_name+'_'+str(p_id)+'.png' )
                        mask = np.array(read_image(path_mask))
                        mask = mask_clamp(mask)
                        if np.any(mask[box[1]:box[3],box[0]:box[2]]):
                            bboxes.append(box)
                            masks.append(mask)
                    except:
                        continue
            bboxes = np.stack(bboxes,axis=0) # if bboxes = [] gives an exception --ok 
            masks = np.stack(masks,axis=0)
            good_sample = True
        except: # image not found or no valid bboxes or not valid masks, use light tipe to speed up
            image = np.zeros((cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3), dtype=np.uint8)
            bboxes = np.zeros((cfg.MAX_INSTANCES,5), dtype=np.uint8) # bbox + pid
            masks = np.zeros((cfg.MAX_INSTANCES, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE), dtype=np.uint8)
            good_sample = False
        return image, masks, bboxes, good_sample
