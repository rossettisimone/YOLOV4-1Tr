
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
            
    def _single_input_generator(self, index):
        [video, frame_id] = self.annotation[index]
        image, masks, bboxes = self.data_generator(video, frame_id)
        if self.data_aug:
            image, masks, bboxes = self.data_augment(image, masks, bboxes)
        image, masks, bboxes = self.data_resize(image, bboxes, masks, (self.train_input_size,self.train_input_size))
        label_2, labe_3, label_4, label_5 = self.data_pyramid_lables(bboxes)
        masks = self.masks_crop_resize(masks, bboxes, cfg.MASK_SIZE)
        masks, bboxes = self.data_pad_max_instances(masks, bboxes, cfg.MAX_INSTANCES)
        return image, label_2, labe_3, label_4, label_5, masks, bboxes
    
    def data_pad_max_instances(self, masks, bboxes, max_instances):
        bboxes_padded = np.zeros((max_instances,5))
        bboxes_padded[:,4]=-1
        #check consistency of bbox after data augmentation
        masks = masks[bboxes[...,2]>bboxes[...,0]]
        masks = masks[bboxes[...,3]>bboxes[...,1]]
        bboxes = bboxes[bboxes[...,2]>bboxes[...,0]]
        bboxes = bboxes[bboxes[...,3]>bboxes[...,1]]
        #zero pad
        min_bbox = min(bboxes.shape[0], max_instances)
        bboxes_padded[:min_bbox,:]=bboxes[:min_bbox,:]
        masks_padded = np.zeros((max_instances,masks.shape[1],masks.shape[2]))
        min_mask = min(masks.shape[0], max_instances)
        masks_padded[:min_mask,:,:]=masks[:min_mask,:,:]
        return masks_padded, bboxes_padded
        
    def masks_crop_resize(self, masks, bboxes, mask_size):
        masks_c_r = []
        for mask, bbox in zip(masks, bboxes):
            mask = Image.fromarray(mask[bbox[1]:bbox[3],bbox[0]:bbox[2]])
            mask = np.round(mask.resize((mask_size,mask_size), Image.ANTIALIAS))
            masks_c_r.append(mask)
        masks = np.stack(masks_c_r,axis=0)
        return masks
        
    def data_resize(self, image, gt_boxes, masks, target_size):
        ih, iw    = target_size
        h,  w, _  = image.shape
        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)    
        image = Image.fromarray(image)
        image_resized = image.resize((nw, nh),Image.ANTIALIAS)
        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        image_paded = image_paded / 255.
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
#        class_ = 1
        path = os.path.join(cfg.VIDEOS_DATASET_PATH, v_id, frame_name+'.png' )
        try:
            image = np.array(read_image(path))
        except:
            image = np.array(Image.new('RGB', (cfg.TRAIN_SIZE, cfg.TRAIN_SIZE), color = (0, 0, 0)))
        height, width, _ = image.shape
        bboxes = []
        masks = [] 
        for person in video['p_l']:
            box = person["bb_l"][frame_id]
            box=np.clip(box,0,1)
            if len(box)==8: # Siammask returns 8 scalars (rotated bbox, rectify them)
                xx, yy = [s for i,s in enumerate(box) if i%2==0 ], [s for i,s in enumerate(box) if not i%2==0]
                box=np.array([min(xx),min(yy),max(xx),max(yy)])
            if not np.all(box==0) and box[2]>box[0] and box[3]>box[1]:
                box = np.r_[box,1]*np.array([width, height, width, height, person['p_id']])
                try:
                     mask = mask_clamp(np.array(read_image(os.path.join(cfg.SEGMENTS_DATASET_PATH, v_id, frame_name+'_'+str(person['p_id'])+'.png' ))))
                except:
                     mask = np.array(Image.new('L', (width, height), color=0))
                bboxes.append(box)
                masks.append(mask)
        bboxes = np.stack(bboxes,axis=0)
        masks = np.stack(masks,axis=0)
        return image, masks, bboxes

    def data_augment(self, image, masks, bboxes):
        image, masks, bboxes = self.random_horizontal_flip(image, masks, bboxes)
        image, masks, bboxes = self.random_crop(image, masks, bboxes)
        image, masks, bboxes = self.random_translate(image, masks, bboxes)
        return image, masks, bboxes
    
    def data_pyramid_lables(self, bboxes):
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
        self.shuffle=shuffle
        self.data_aug = data_aug
        self.json_dataset = file_reader(cfg.ANNOTATION_PATH)
        self.nID = 0        
        if cfg.DATASET_TYPE == 'kinetics':
            for i,k in enumerate(self.json_dataset):
                for j,_ in enumerate(k['p_l']):
                    # self.json_dataset[i]['p_l'][j]['p_id_2'] = self.nID
                    self.nID += 1
        elif cfg.DATASET_TYPE == 'ava':
            self.max_id_in_video = {}
            for i,k in enumerate(self.json_dataset):
                for j,_ in enumerate(k['p_l']):
                    try:
                        if self.json_dataset[i]['p_l'][j]['p_id'] > self.max_id_in_video[self.json_dataset[i]['v_id']]:
                            self.max_id_in_video[self.json_dataset[i]['v_id']] = self.json_dataset[i]['p_l'][j]['p_id'] 
                    except KeyError as e:
                        self.max_id_in_video[self.json_dataset[i]['v_id']] = 0
        #     # keys are ordered
        #     self.keys_offset = {}
        #     self.keys = sorted(self.max_id_in_video.keys())
        #     self.offset = [1 if i == 0 else sum([self.max_id_in_video[m] for j,m in enumerate(self.keys) if j<i]) + 1 + i for i,k in enumerate(self.keys)]
        #     for i,k in enumerate(self.keys):
        #         self.keys_offset[k] = self.offset[i]
        #     for i,k in enumerate(self.json_dataset):
        #         for j,_ in enumerate(k['p_l']):
        #             self.json_dataset[i]['p_l'][j]['p_id_2'] = self.json_dataset[i]['p_l'][j]['p_id'] #+ self.keys_offset[self.json_dataset[i]['v_id']]
            self.nID = sum(self.max_id_in_video.values()) + len(self.max_id_in_video.values()) # id starts from zero, count them
        #     self.nID = max(self.max_id_in_video.values()) + 1 #* len(self.max_id_in_video.values()) + 1
        self.annotation = [(video,frame_id) for video in self.json_dataset for frame_id in range(0,61) if not all(p['bb_l'][frame_id]==[0,0,0,0] for p in video['p_l'])] # (video,0),(video,10),..,(video,60) sample each 10 frames
        self.train_list, self.val_list = self.split_dataset(len(self.annotation))
        self.train_ds = self.initilize_ds(self.train_list)
        self.val_ds = self.initilize_ds(self.val_list)
        self.num_classes = cfg.NUM_CLASS
        self.anchors = tf.reshape(tf.constant(cfg.ANCHORS,dtype=tf.float32),[cfg.LEVELS, cfg.NUM_ANCHORS, 2])
#        self.anchor_per_scale = cfg.ANCHOR_PER_SCALE
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
    
    def read_transform(self, idx):
        image, label_2, labe_3, label_4, label_5, masks, bboxes = tf.py_function(self._single_input_generator, [idx], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
        return image, label_2, labe_3, label_4, label_5, masks, bboxes

    
    def initilize_ds(self, list_ids):
        ds = tf.data.Dataset.from_generator(DataLoader.input_generator , args= [list_ids], output_types= (tf.int32))
        ds = ds.map(self.read_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds
