#%%
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

import numpy as np
import json

def file_reader(file_name):
    with open(file_name) as json_file:
        return json.load(json_file)


#%%
class Generator(object):
            
    def _single_input_generator_train(self, index):
        [video, frame_id] = self.annotation_train[index]
        data, label = self.data_generator(video, frame_id)
        return data, label
    
    def _single_input_generator_val(self, index):
        
        [video, frame_id] = self.annotation_val[index]
        data, label = self.data_generator(video, frame_id)
        return data, label
    
    # data: 
    #       batch x 100 (n^2 where n are the subjects) x 6 (cxt, cyt, vxt, vyt, cxt2, cyt2)
    # label:
    #       batch x 100 (n^2 where n are the subjects)
    def data_generator(self, video, frame_id):
        t = frame_id
        t1 = frame_id + 1
        t2 = frame_id + 2
        v_id =  video['v_id']
        s_l = video['p_l']
        max_instances = 10 #subjects
        c_v = np.zeros((max_instances,4))
        c_next = np.zeros((max_instances,2))
        for i in range(min(len(s_l),max_instances)):
            s = s_l[i]
            bb_l = np.array(s['bb_l'])
            c = (bb_l[...,:2]+bb_l[...,2:4])*0.5
            c_t = c[t]
            c_t2 = c[t2]
            if not np.sum(c_t) == 0 and not np.sum(c_t2) == 0:
                c_t1 = c[t1]
                dc_t_t1 = (c_t1 - c_t)/2 #offset within 2 frames
                angle = np.arctan2(dc_t_t1[0],dc_t_t1[1])#angle
                d_t = [np.sin(angle), np.cos(angle)] #direction
                c_v[i] = np.concatenate([c_t, d_t],axis=0) # centroid, direction
                c_next[i] = c_t2
        
        data = np.zeros((max_instances**2,6))
        label = np.zeros((max_instances**2))
        
        for i in range(max_instances):
            for j in range(max_instances):
                data[i*max_instances+j]=np.concatenate([c_v[i], c_next[j]],axis=0)
                if i == j:
                    label[i*max_instances+j] = 1
        
        indices = np.arange(max_instances**2)
        np.random.shuffle(indices)
        
        data = data[indices]
        label = label[indices]

        label = np.expand_dims(label,axis=-1)
        return data, label

#%%
class DataLoader(Generator):
    def __init__(self, shuffle=True, data_aug=True):
        print('Dataset loading..')
        self.shuffle = shuffle
        self.data_aug = data_aug
        self.max_instances = 10
        self.dim_data = 6
        self.json_train_dataset = []
        for file_path in cfg.TRAIN_ANNOTATION_PATH:
            self.json_train_dataset += file_reader(file_path)
            print('Train Dataset {} loaded'.format(file_path))
        self.json_val_dataset = []
        for file_path in cfg.VAL_ANNOTATION_PATH:
            self.json_val_dataset += file_reader(file_path)
            print('Validation Dataset {} loaded'.format(file_path))
        self.annotation_train = [(video,frame_id) for video in self.json_train_dataset \
                for frame_id in range(0,61-2)] # cut last two since we want to associate curresnt step with next two above
        self.annotation_val = [(video,frame_id) for video in self.json_val_dataset \
                for frame_id in range(0,61-2)]
        self.train_list = np.arange(len(self.annotation_train))
        self.val_list = np.arange(len(self.annotation_val))
        if self.shuffle:
            np.random.shuffle(self.train_list)
            np.random.shuffle(self.val_list)
        self.train_ds = self.initilize_train_ds(self.train_list)
        self.val_ds = self.initilize_val_ds(self.val_list)
        print('Dataset loaded.')
        # print('# identities:',self.nID)
            
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
        data, label = tf.py_function(self._single_input_generator_train, [idx], [tf.float32, tf.int32])
        return data, label

    def read_transform_val(self, idx):
        data, label = tf.py_function(self._single_input_generator_val, [idx], [tf.float32, tf.int32])
        return data, label

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
# %%
ds = DataLoader()
# %%
def filter_zeros(data,label): # all zeros is useless
    return not tf.reduce_sum(data)==0

train_iterator = ds.train_ds.repeat().filter(filter_zeros).batch(1).__iter__()
val_iterator = ds.val_ds.repeat().filter(filter_zeros).batch(1).__iter__()

# %%
data, label = train_iterator.next()

# %%
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

input_shape = data.shape[1:]
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal',input_shape=input_shape))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
# 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 
# %%
model.fit_generator(train_iterator, steps_per_epoch= 300, epochs=20, verbose=1)

loss, acc = model.evaluate_generator(val_iterator, steps = 500, verbose=1)

print('Test Accuracy: %.3f' % acc)

#%%
data, label = train_iterator.next()

threshold = 0.5
prob = model(data)
guess = tf.gather(data,tf.where(prob>threshold)[...,1], axis=1)
true = tf.gather(data,tf.where(label>0)[...,1],axis=1)
true = tf.gather(true, tf.where(tf.reduce_sum(true,axis=-1)!=0)[...,1], axis=1)

print('True:\n', true.numpy())
print('Guess:\n', guess.numpy())
