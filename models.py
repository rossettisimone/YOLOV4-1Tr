#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import config as cfg
#import matplotlib.pyplot as plt
from backbone import cspdarknet53
from layers import CustomUpsampleAndConcatAndShuffle, CustomDownsampleAndConcatAndShuffle, CustomDecode
from PIL import Image, ImageDraw, ImageFont
import gc
    
#class SoftmaxLoss(tf.keras.losses.Loss):
#  def call(self, y_true, y_pred):
#    return tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)

class fpn(tf.keras.Model):
    def __init__(self, name='fpn', **kwargs):
        super(fpn, self).__init__(name=name, **kwargs)
        
        self.up_1 = CustomUpsampleAndConcatAndShuffle(filters=256, n=1)
        self.up_2 = CustomUpsampleAndConcatAndShuffle(filters=128, n=2)
        self.up_3 = CustomUpsampleAndConcatAndShuffle(filters=64, n=3)
#        self.up_4 = CustomUpsampleAndConcatAndShuffle(filters=32, n=4)
#        self.down_4 = CustomDownsampleAndConcatAndShuffle(filters=64, n=4)
        self.down_3 = CustomDownsampleAndConcatAndShuffle(filters=128, n=3)
        self.down_2 = CustomDownsampleAndConcatAndShuffle(filters=256, n=2)
        self.down_1 = CustomDownsampleAndConcatAndShuffle(filters=512, n=1)

    def call(self, input_layers, training=False):
        b_2, b_3, b_4, b_5 = input_layers #b_1
        p_5 = b_5
        p_4 = self.up_1(p_5,b_4, training)
        p_3 = self.up_2(p_4,b_3, training)
        p_2 = self.up_3(p_3,b_2, training)
#        p_1 = self.up_4(p_2,b_1, training)
#        n_1 = p_1
#        n_2 = self.down_4(n_1,p_2, training)
        n_2 = p_2
        n_3 = self.down_3(n_2,p_3, training)
        n_4 = self.down_2(n_3,p_4, training)
        n_5 = self.down_1(n_4,p_5, training)
        return  n_2, n_3, n_4, n_5 #n_1
    
    def build_graph(self):
        inputs = [tf.keras.Input(shape=(13*2**(cfg.LEVELS-1-n), 13*2**(cfg.LEVELS-1-n), 
                                        2**(cfg.LEVELS+n+(1 if n<cfg.LEVELS-1 else 0)))) for n in range(cfg.LEVELS)]
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))

class rpn(tf.keras.Model):
    def __init__(self, name='rpn', **kwargs):
        super(rpn, self).__init__(name=name, **kwargs)
        
        self.heads = [CustomDecode(level, n=level+1) for level in range(cfg.LEVELS)]
        
    def call(self, input_layers, training=False, inferring=False):
        return [(f(n, training, inferring)) for f, n in zip(self.heads ,input_layers)]
    
    def build_graph(self):
        inputs = [tf.keras.Input(shape=(13*2**(cfg.LEVELS-1-n), 13*2**(cfg.LEVELS-1-n), 
                                        2**(cfg.LEVELS+n))) for n in range(cfg.LEVELS)]
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
    
class tracker(tf.keras.Model):
    def __init__(self, freeze_bkbn = True, freeze_bn = False, data_loader = None, name='tracker', **kwargs):
        super(tracker, self).__init__(name=name, **kwargs)
        
        self.nID = 57398
        if data_loader is not None:
            self.ds = data_loader
            self.train_ds = self.ds.train_ds
            self.val_ds = self.ds.val_ds
            self.nID =  self.ds.nID
            self.epochs = cfg.EPOCHS
            self.epoch = 0
            self.step_train = 0
            self.step_val = 0
            self.batch = cfg.BATCH
            self.steps_per_epoch = cfg.STEPS_PER_EPOCH
            self.step_trains = self.epochs * self.steps_per_epoch * self.batch
        self.freeze_bkbn = freeze_bkbn
        self.freeze_bn = freeze_bn
        self.LEVELS = cfg.LEVELS
        self.NUM_ANCHORS = cfg.NUM_ANCHORS
        self.ANCHORS = tf.reshape(tf.constant(cfg.ANCHORS,dtype=tf.float32),[self.LEVELS, self.NUM_ANCHORS, 2])
        self.STRIDES = tf.constant(cfg.STRIDES,dtype=tf.float32)
        self.emb_dim = cfg.EMB_DIM 
        self.writer = tf.summary.create_file_writer(cfg.SUMMARY_LOGDIR)
        self.emb_scale = (tf.math.sqrt(2.0) * tf.math.log(self.nID-1.0)) if self.nID>1.0 else 1.0
        self.SmoothL1Loss = tf.keras.losses.Huber(delta=1.0)
#        self.SoftmaxLoss = SoftmaxLoss()
        self.CrossEntropyLossLogits = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tfa.optimizers.SGDW( weight_decay = cfg.WD, learning_rate = cfg.LR, momentum = cfg.MOM, nesterov = False) #tf.keras.optimizers.Adam(learning_rate = cfg.LR)#
#        self.strategy = tf.distribute.MirroredStrategy()

        #CSPDARKNET53 
        self.bkbn = cspdarknet53()
        #PANET LIKE
        self.neck = fpn()
        #YOLO LIKE
        self.head = rpn()
        
        self.classifier = tf.keras.layers.Dense(self.nID) if self.nID>0 else None
        self.classifier.build((tf.newaxis,self.emb_dim))
        self.s_c = tf.Variable(initial_value=[-4.15], trainable=True) 
        self.s_r = tf.Variable(initial_value=[-4.85], trainable=True)
        self.s_id = tf.Variable(initial_value=[-2.3], trainable=True)
        
    def call(self, input_layers, training=False, inferring=False):
        b = self.bkbn(input_layers, training)
        n = self.neck(b, training)
        h = self.head(n, training, inferring)
        if training and inferring:
            # we havefor each level proposals for each image in the batch, thus to be more flexible we transpose the list of lists
            # pythonic transpose list of list of irregular size: [[image1 - fpn1, image2 - fpn1, ..], [image1 - fpn2], [image2 - fpn2], ..]
            l = [len(i) for i in h]
            h = [[i[o] for ix, i in enumerate(h) if l[ix] > o] for o in range(max(l))] 
            h = [tf.concat(p, axis=0) for p in h] # concat proposals from all levels for each image in batch
        return h
    
    def train_step(self, data):
        self.step_train += 1
        training = True
        inferring = False
        image, *labels = data
        with tf.GradientTape() as tape:
            preds = self(image, training=training, inferring=inferring)
            self.total_loss = [] 
            self.box_loss = []
            self.conf_loss = []
            self.id_loss = []
            for label, (pred, emb) in zip(labels, preds):
                loss, lbox, lconf, lid = self.compute_loss(label, pred, emb, training)
                self.total_loss.append(loss)
                self.box_loss.append(lbox)
                self.conf_loss.append(lconf)
                self.id_loss.append(lid)
            self.mean_total_loss = tf.reduce_mean(tf.concat(self.total_loss, axis=0),axis=0)
            self.mean_box_loss = tf.reduce_mean(tf.concat(self.box_loss, axis=0),axis=0)
            self.mean_conf_loss = tf.reduce_mean(tf.concat(self.conf_loss, axis=0),axis=0)
            self.mean_id_loss = tf.reduce_mean(tf.concat(self.id_loss, axis=0),axis=0)
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(self.mean_total_loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            self.loss_summary(training)
        tf.print("=> STEP %4d/%4d   lr: %.6f   mean_box_loss: %4.2f   mean_conf_loss: %4.2f   "
                     "mean_id_loss: %4.2f   mean_total_loss: %4.2f" % (self.step_train, self.step_trains, \
                                                             self.optimizer.lr.numpy(), self.mean_box_loss, \
                                                             self.mean_conf_loss, self.mean_id_loss, self.mean_total_loss))
#    @tf.function
    def compute_loss(self, label, pred, emb, training):
        pbox = pred[..., :4]
        pconf = pred[..., 4:6]
        tbox = label[...,:4]
        tconf = label[...,4]
        tids = label[...,5:6]
        mask = tconf>0
        lbox = self.SmoothL1Loss(tbox[mask],pbox[mask])
        pconf = self.entry_stop_gradients(pconf, tconf[...,tf.newaxis]<0) # stop gradient for regions labeled -1 below CONF threshold, look dataloader
        tconf = tf.cast(tf.where(tf.less(tconf, 0.0), 0.0, tconf), tf.int32)
        lconf =  self.CrossEntropyLossLogits(tconf,pconf) # apply softmax and do non negative log likelihood loss 
        emb_mask = tf.cast(tf.math.reduce_max(tf.cast(mask, tf.float32), axis=1), tf.bool)
        tids = tf.math.reduce_max(tids, axis=1)
        tids = tids[emb_mask]
        embedding = emb[emb_mask]
        embedding = self.emb_scale * tf.math.l2_normalize(embedding,axis=1, epsilon=1e-12)
        logits = self.classifier(embedding)
        logits = self.entry_stop_gradients(logits,tids<0)
        tids = tf.cast(tf.where(tf.less(tids, 0.0), 0.0, tids),tf.int32) # stop gradient for regions labeled -1
        lid = self.CrossEntropyLossLogits(tf.squeeze(tids),logits)
        self.classifier_summary(embedding, training)
        loss = tf.math.exp(-self.s_r)*lbox + tf.math.exp(-self.s_c)*lconf \
                + tf.math.exp(-self.s_id)*lid + (self.s_r + self.s_c + self.s_id) #Automatic Loss Balancing
        loss *= 0.5
        return loss, lbox, lconf, lid
    
    def entry_stop_gradients(self,target, mask):
        mask_neg = tf.math.logical_not(mask)
        return tf.stop_gradient(tf.cast(mask,tf.float32) * target) + tf.cast(mask_neg,tf.float32) * target

    def test_step(self, data):
        self.step_val += 1
        training = True
        inferring = False
        image, *labels = data
        preds = self(image, training=training, inferring=inferring)
        self.total_loss = [] 
        self.box_loss = []
        self.conf_loss = []
        self.id_loss = []
        for label, (pred, emb) in zip(labels, preds):
            loss, lbox, lconf, lid = self.compute_loss(label, pred, emb, training)
            self.total_loss.append(loss)
            self.box_loss.append(lbox)
            self.conf_loss.append(lconf)
            self.id_loss.append(lid)
        self.mean_total_loss = tf.reduce_mean(tf.concat(self.total_loss, axis=0),axis=0)
        self.mean_box_loss = tf.reduce_mean(tf.concat(self.box_loss, axis=0),axis=0)
        self.mean_conf_loss = tf.reduce_mean(tf.concat(self.conf_loss, axis=0),axis=0)
        self.mean_id_loss = tf.reduce_mean(tf.concat(self.id_loss, axis=0),axis=0) 
        self.loss_summary(training)
        tf.print("=> STEP %4d   mean_box_loss: %4.2f   mean_conf_loss: %4.2f   "
                     "mean_id_loss: %4.2f   mean_total_loss: %4.2f" % (self.step_val, self.mean_box_loss, \
                                                             self.mean_conf_loss, self.mean_id_loss, \
                                                             self.mean_total_loss))
    def fit(self, epochs = None, start_epoch = 0):
        assert self.ds is not None, 'please pass a DataLoader'
        if epochs is not None:
            self.epochs = epochs
        self.start_epoch = start_epoch
        for epoch in range(self.epochs):
            self.epoch += 1
            if self.freeze_bkbn and self.epoch < 2:
                self.bkbn.trainable = False
            else:
                self.bkbn.trainable = True
            if self.freeze_bn: # finetuning 
                self.freeze_batch_normalization() 
            self.adapt_lr()
            for data in self.train_ds.take(self.steps_per_epoch*self.batch).batch(self.batch):
                self.train_step(data)
            self.save('./tracker_weights_'+str(self.epoch)+'.tf')
            gc.collect()
            for data in self.val_ds.take(100*self.batch).batch(self.batch):
                self.test_step(data)               
            gc.collect()
    
    def infer(self, image):
        training = True
        inferring = True
        proposals = self(image, training=training, inferring=inferring)
        if len(proposals)==0:
            tf.print('None')
        for i,proposal in enumerate(proposals):
            if len(proposal)>0:
                bboxs = proposal[...,:4]
                confs = proposal[...,4] 
                embs = proposal[...,5:] # needed for tracking
                # Final proposals are obtained in dets. Information of bounding box and embeddings also included
                self.draw_bbox(image[i], bboxs, confs)
    #            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f.numpy(), 30) for
    #                          (tlbrs, f) in zip(dets[:, :5], dets[:, 6:])]
            else:
    #            detections = []
                tf.print('None')
                
    def draw_bbox(self,image, bboxs, conf_id):
        img = Image.fromarray(np.array(image.numpy()*255,dtype=np.uint8))                   
        draw = ImageDraw.Draw(img)   
        for bbox,conf in zip(bboxs.numpy(), conf_id.numpy()):
            draw.rectangle(bbox, outline ="red") 
            xy = ((bbox[2]+bbox[0])*0.5, (bbox[3]+bbox[1])*0.5)
            draw.text(xy, str(np.round(conf,3)), font=ImageFont.truetype("arial.ttf"))
        img.show() 
        
    def loss_summary(self, training):
        with self.writer.as_default():
            scope = 'train' if training else 'val'
            step = self.step_train if training else self.step_val
            with tf.name_scope(scope):
                with tf.name_scope('loss'):
                    tf.summary.scalar("s_c", tf.squeeze(self.s_c), step=step)
                    tf.summary.scalar("s_r", tf.squeeze(self.s_r), step=step)
                    tf.summary.scalar("s_id", tf.squeeze(self.s_id), step=step)
                    tf.summary.scalar("lr", self.optimizer.lr, step=self.step_train)
                    tf.summary.scalar("mean_total_loss", tf.squeeze(self.mean_total_loss), step=step)
                    tf.summary.scalar("mean_box_loss", tf.squeeze(self.mean_box_loss), step=step)
                    tf.summary.scalar("mean_conf_loss", tf.squeeze(self.mean_conf_loss), step=step)
                    tf.summary.scalar("mean_id_loss", tf.squeeze(self.mean_id_loss), step=step)
        self.writer.flush()
        
    def classifier_summary(self, input_tensor, training):
        with self.writer.as_default():
            scope = 'train' if training else 'val'
            step = self.step_train if training else self.step_val
            with tf.name_scope(scope):
                with tf.name_scope(self.classifier.name):
                    with tf.name_scope('weights'):
                        self.variable_summaries(self.classifier.kernel,step)
                    with tf.name_scope('biases'):
                        self.variable_summaries(self.classifier.bias,step)
                    with tf.name_scope('output'):
                        preactivate = tf.matmul(input_tensor, self.classifier.kernel) + self.classifier.bias
                        tf.summary.histogram('output', preactivate, step=step)
        self.writer.flush()

    def freeze_batch_normalization(self):
        for net in self.layers:
            if net.name == 'cspdarknet53':
                for layer in net.bkbn.layers:
                    if 'batch_normalization' in layer.name:
                        layer.trainable = False
            elif net.name == 'fpn':
                for layer in net.layers:
                    if layer.name == 'upsample_concat_shuffle':
                        layer.up_concat.conv_1.batch_norm.trainable = False
                        layer.up_concat.conv_2.batch_norm.trainable = False
                        layer.shuffle.conv_1.batch_norm.trainable = False
                        layer.shuffle.conv_2.batch_norm.trainable = False
                        layer.shuffle.conv_3.batch_norm.trainable = False
                        layer.shuffle.conv_4.batch_norm.trainable = False
                        layer.shuffle.conv_5.batch_norm.trainable = False
                    if layer.name == 'downsample_concat_shuffle':
                        layer.conv_down.batch_norm.trainable = False
                        layer.shuffle.conv_1.batch_norm.trainable = False
                        layer.shuffle.conv_2.batch_norm.trainable = False
                        layer.shuffle.conv_3.batch_norm.trainable = False
                        layer.shuffle.conv_4.batch_norm.trainable = False
                        layer.shuffle.conv_5.batch_norm.trainable = False
                    if layer.name == 'decode':
                        layer.conv_1.batch_norm.trainable = False
                        layer.conv_2.batch_norm.trainable = False
                        layer.conv_3.batch_norm.trainable = False
                
    def variable_summaries(self, var, step):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, step=step)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev, step=step)
        tf.summary.scalar('max', tf.reduce_max(var), step=step)
        tf.summary.scalar('min', tf.reduce_min(var), step=step)
        tf.summary.histogram('histogram', var, step=step)

    def plot(self):
        tf.keras.utils.plot_model(self.bkbn.model, to_file='cspdarknet53.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
        tf.keras.utils.plot_model(self.neck.build_graph(), to_file='fpn.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
        tf.keras.utils.plot_model(self.head.build_graph(), to_file='rpn.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
        
    def custom_build(self):
        self.build((tf.newaxis, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3))
         
    def adapt_lr(self):
        if self.epoch < self.epochs * 0.5 :
            lr = cfg.LR
        elif self.epoch <= self.epochs * 0.75 and self.epoch > self.epochs * 0.5:
            lr = cfg.LR * 0.01
        else:
            lr = cfg.LR * 0.001
        self.optimizer.lr.assign(lr)
        
    def save(self, name):
        self.save_weights(name)

    def load(self, name):
        self.load_weights(name)


