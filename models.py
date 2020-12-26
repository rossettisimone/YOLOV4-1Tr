#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import config as cfg
#import matplotlib.pyplot as plt
from backbone import cspdarknet53
from layers import CustomUpsampleAndConcatAndShuffle, CustomDownsampleAndConcatAndShuffle, CustomDecode
from PIL import Image, ImageDraw 


class pan(tf.keras.Model):
    def __init__(self, name='pan', **kwargs):
        super(pan, self).__init__(name=name, **kwargs)
        
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

class yolov3(tf.keras.Model):
    def __init__(self, name='yolov3', **kwargs):
        super(yolov3, self).__init__(name=name, **kwargs)
        
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
        
        self.nID = 0
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
        self.CrossEntropyLoss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.CrossEntropyLossLogits = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tfa.optimizers.SGDW( weight_decay = cfg.WD, learning_rate = cfg.LR, momentum = cfg.MOM, nesterov = False) #tf.keras.optimizers.Adam(learning_rate = cfg.LR)#
#        self.strategy = tf.distribute.MirroredStrategy()

        #BKBN
        self.bkbn = cspdarknet53()
        #FPN
        self.neck = pan()
        #YOLO
        self.head = yolov3()
        
        self.classifier = tf.keras.layers.Dense(self.nID) if self.nID>0 else None
        self.classifier.build((tf.newaxis,self.emb_dim))
        self.s_c = tf.Variable(initial_value=[-4.15], trainable=True) 
        self.s_r = tf.Variable(initial_value=[-4.85], trainable=True)
        self.s_id = tf.Variable(initial_value=[-2.3], trainable=True)
        
    def call(self, input_layers, training=False, inferring=False):
        b = self.bkbn(input_layers, training)
        n = self.neck(b, training)
        h = self.head(n, training, inferring)
        return h
    
    def train_step(self, data):
        self.step_train += 1
        training = True
        image, *labels = data
        with tf.GradientTape() as tape:
            preds = self(image, training=training)
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
#        pred = tf.transpose(tf.reshape(pred, [pred.shape[0], pred.shape[1], pred.shape[2], cfg.NUM_ANCHORS, cfg.NUM_CLASS + 5]), perm = [0, 3, 1, 2, 4])  # prediction        
        pbox = pred[..., :4]
        pconf = pred[..., 4:6]  # Conf
        tbox = label[...,:4]
        tconf = label[...,4]
        tids = label[...,5:6]
#        tmask = label[..., 6]
        mask = tconf>0
#        mask = tf.where(tf.greater(tconf, 0.0))
        lbox = self.SmoothL1Loss(tbox[mask],pbox[mask])
        tconf = tf.where(tf.less(tconf, 0.0), 0.0, tconf)
        pconf = tf.where(tf.less(tconf[...,tf.newaxis], 0.0), 0.0, pconf)
        lconf =  self.CrossEntropyLoss(tconf,pconf)
        emb_mask = tf.cast(tf.math.reduce_max(tf.cast(mask, tf.float32), axis=1), tf.bool)
        tids = tf.math.reduce_max(tids, axis=1)
        tids = tids[emb_mask]
        embedding = emb[emb_mask]
        embedding = self.emb_scale * tf.math.l2_normalize(embedding,axis=1, epsilon=1e-12)
        logits = self.classifier(embedding)
        tids = tf.where(tf.less(tids, 0.0), 0.0, tids)
        logits = tf.where(tf.less(tids, 0.0), 0.0, logits)
        lid = self.CrossEntropyLossLogits(tf.squeeze(tids),logits)
        self.classifier_summary(embedding, training)
        loss = tf.math.exp(-self.s_r)*lbox + tf.math.exp(-self.s_c)*lconf + tf.math.exp(-self.s_id)*lid + (self.s_r + self.s_c + self.s_id)
        loss *= 0.5
        return loss, lbox, lconf, lid
    
    def test_step(self, data):
        self.step_val += 1
        training = False
        image, *labels = data
        preds = self(image, training=training)
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
            for data in self.val_ds.take(100*self.batch).batch(self.batch):
                self.test_step(data)               
            self.save('./tracker_weights_'+str(self.epoch)+'.tf')
    
    def infer(self, image):
        assert tf.shape(image)[0]==1, 'batch must be one image at time'
        preds = self(image, training=False, inferring=True)
        pred = tf.concat(preds, axis=1)
        pred = pred[pred[..., 4] > cfg.CONF_THRESH]
        pred = tf.concat([tf.concat([pred[...,:2] - pred[...,2:4]*0.5, pred[...,:2] + pred[...,2:4]*0.5], axis=1),pred[...,4:]],axis=1)
        # pred now has lesser number of proposals. Proposals rejected on basis of object confidence score
        if len(pred) > 0:    
            boxes = pred[...,:4]
            scores = pred[...,4]
            selected_indices = tf.image.non_max_suppression(
                                boxes, scores, max_output_size=20, iou_threshold=cfg.NMS_THRESH,
                                score_threshold=cfg.CONF_THRESH
                            )
            selected_boxes = tf.gather(pred, selected_indices)
            # Final proposals are obtained in dets. Information of bounding box and embeddings also included
            # Next step changes the detection scales
#            scale_coords(self.opt.img_size, dets[:, :4], image.shape).round()
#            '''Detections is list of (x1, y1, x2, y2, object_conf, class_score, class_pred)'''
            # class_pred is the embeddings.
#            detections = [] 

            # creating new Image object 
            img = Image.fromarray(np.array(image[0].numpy()*255,dtype=np.uint8)) 
              
            # create rectangle image 
            img1 = ImageDraw.Draw(img)   
            for bbox in selected_boxes.numpy():
                shape = bbox[:4]
                img1.rectangle(shape, outline ="red") 
            img.show() 
#            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f.numpy(), 30) for
#                          (tlbrs, f) in zip(dets[:, :5], dets[:, 6:])]
        else:
#            detections = []
            tf.print('None')
            
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
            elif net.name == 'pan':
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
        tf.keras.utils.plot_model(self.neck.build_graph(), to_file='pan.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
        tf.keras.utils.plot_model(self.head.build_graph(), to_file='yolov3.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
        
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


