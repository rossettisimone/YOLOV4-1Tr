#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tensorflow_addons as tfa
import config as cfg
from backbone import cspdarknet53
from layers import CustomUpsampleAndConcatAndShuffle, CustomDownsampleAndConcatAndShuffle, CustomDecode,ProposalLayer, PyramidROIAlign_AFP
import gc
from utils import entry_stop_gradients, preprocess_mrcnn, mrcnn_class_loss_graph, mrcnn_bbox_loss_graph, mrcnn_mask_loss_graph, draw_bbox
from datetime import datetime

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
        return  n_2, n_3, n_4, n_5 #n_1 #embedding
    
    def build_graph(self):
        return tf.keras.Model(inputs=self.get_input_shape(), outputs=self.call(self.get_input_shape()))
    
    def get_input_shape(self):
        return [tf.keras.Input(shape=(cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], 
                                        128*2**(i-(0 if i<cfg.LEVELS-1 else 1)))) for i in range(cfg.LEVELS)]
    def get_output_shape(self):
        return self.call(self.get_input_shape())
        
class rpn(tf.keras.Model):
    def __init__(self, name='rpn', **kwargs):
        super(rpn, self).__init__(name=name, **kwargs)
        
        self.heads = [CustomDecode(level, n=level+1) for level in range(cfg.LEVELS)]
        
    def call(self, input_layers, training=False, inferring=False):
        preds = []
        embs = []
        for i in range(cfg.LEVELS):
            x = self.heads[i](input_layers[i], training, inferring)
            preds.append(x[0])
            embs.append(x[1])
        return preds, embs

    def get_input_shape(self):
        return [tf.keras.Input(shape=(cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], 
                                        64*2**i)) for i in range(cfg.LEVELS)]
    def get_output_shape(self):
        preds, embs = self.call(self.get_input_shape())
        return [out.shape for out in preds], [out.shape for out in embs]
    
class MSDS(tf.keras.Model): #MSDS, multi subject detection and segmentation
    def __init__(self, emb = True, mask = True, freeze_bkbn = True, freeze_bn = False, data_loader = None, name='MSDS', **kwargs):
        super(MSDS, self).__init__(name=name, **kwargs)
        
        self.nID = 972
        self.mask = mask
        self.emb = emb
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
            self.step_trains = self.epochs * self.steps_per_epoch
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
            
        self.s_c = tf.Variable(initial_value=0.0, trainable=True) #-4.15
        self.s_r = tf.Variable(initial_value=0.0, trainable=True) # -4.85
        
        if self.emb:
            self.s_id = tf.Variable(initial_value=0.0, trainable=True) # -2.3
            self.classifier = tf.keras.layers.Dense(self.nID) if self.nID>0 else None
            self.classifier.build((tf.newaxis,self.emb_dim))
            
        if self.mask:
            self.proposal = ProposalLayer() #proposal + embedding
            self.fpn_classifier = fpn_classifier_AFP()            
            self.fpn_mask = fpn_mask_AFP()
            self.s_mc = tf.Variable(initial_value=0.0, trainable=True) 
            self.s_mr = tf.Variable(initial_value=0.0, trainable=True)
            self.s_mm = tf.Variable(initial_value=0.0, trainable=True)
    
    def call(self, input_layers, training=False, inferring=False):
        features = self.bkbn(input_layers, training)
        pyramid = self.neck(features, training)
        rpn_pred, rpn_embeddings = self.head(pyramid, training, inferring) # (proposal, embedding),...(proposal_embedding) for each piramid level
        if self.mask:
            rpn_proposals = self.proposal(rpn_pred, rpn_embeddings, training, inferring) # p,e = proposals, embeddings [batch, rois, (x1, y1, x2, y2, embeddings...)]
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.fpn_classifier([rpn_proposals[...,:4],rpn_embeddings])
            mrcnn_mask = self.fpn_mask([rpn_proposals[...,:4],rpn_embeddings])
            return rpn_pred, rpn_embeddings, rpn_proposals, mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask
        else:
            return rpn_pred, rpn_embeddings
    
    def get_input_shape(self):
        return tf.zeros((cfg.BATCH,cfg.TRAIN_SIZE,cfg.TRAIN_SIZE, 3))
    
    def get_output_shape(self):
        return [out.shape for out in self.call(self.get_input_shape())]
    
    @tf.function
    def train_step(self, data):
        training = True
        inferring = False
        image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
        labels = [label_2, labe_3, label_4, label_5]
        with tf.GradientTape() as tape:
            if self.mask:
                preds, embs, proposals, pred_class_logits, pred_class, pred_bbox, pred_mask = self(image, training=training, inferring=inferring)
                target_class_ids, target_bbox, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks) # preprocess and tile labels according to IOU
                alb_total_loss, *loss_list = self.compute_loss(labels, preds, embs, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_mask, training)
            else:
                preds, embs = self(image, training=training, inferring=inferring)
                alb_total_loss, *loss_list = self.compute_loss_rpn(labels, preds, embs, training)
            
            gradients = tape.gradient(alb_total_loss, self.trainable_variables)
            self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.trainable_variables) if grad is not None)
        return [alb_total_loss] + loss_list
    
    @tf.function                    
    def compute_loss(self, labels, preds, embs, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_masks, training):
        # rpn loss 
        alb_total_loss, *rpn_loss_list = self.compute_loss_rpn(labels, preds, embs, training)
        # mrcnn loss
        alb_loss, *mrcnn_loss_list = self.compute_loss_mrcnn(proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_masks)
        alb_total_loss += alb_loss
        alb_total_loss *= 0.5

        return [alb_total_loss] + rpn_loss_list + mrcnn_loss_list

    @tf.function
    def compute_loss_rpn(self, labels, preds, embs, training):
        if self.emb:
            rpn_box_loss = []
            rpn_class_loss = []
            rpn_id_loss = []
            for label, pred, emb in zip(labels, preds, embs):
                lbox, lconf, lid = self.compute_loss_rpn_level(label, pred, emb, training)
                rpn_box_loss.append(lbox)
                rpn_class_loss.append(lconf)
                rpn_id_loss.append(lid) 
            rpn_box_loss, rpn_class_loss, rpn_id_loss = tf.reduce_mean(rpn_box_loss,axis=0), tf.reduce_mean(rpn_class_loss,axis=0), tf.reduce_mean(rpn_id_loss,axis=0) 
            alb_loss = tf.math.exp(-self.s_r)*rpn_box_loss + tf.math.exp(-self.s_c)*rpn_class_loss \
                        + tf.math.exp(-self.s_id)*rpn_id_loss + (self.s_r + self.s_c + self.s_id) #Automatic Loss Balancing        
            return alb_loss, rpn_box_loss, rpn_class_loss, rpn_id_loss
        else:
            rpn_box_loss = []
            rpn_class_loss = []
            for label, pred, emb in zip(labels, preds, embs):
                lbox, lconf = self.compute_loss_rpn_level(label, pred, emb, training)
                rpn_box_loss.append(lbox)
                rpn_class_loss.append(lconf)
            rpn_box_loss, rpn_class_loss = tf.reduce_mean(rpn_box_loss,axis=0), tf.reduce_mean(rpn_class_loss,axis=0)
            alb_loss = tf.math.exp(-self.s_r)*rpn_box_loss + tf.math.exp(-self.s_c)*rpn_class_loss \
                        + (self.s_r + self.s_c) #Automatic Loss Balancing        
            return alb_loss, rpn_box_loss, rpn_class_loss

    @tf.function
    def compute_loss_rpn_level(self, label, pred, emb, training):
        pbox = pred[..., :4]
        pconf = pred[..., 4:6]
        tbox = label[...,:4]
        tconf = label[...,4]
        mask = tconf>0
        if tf.reduce_sum(tf.cast(mask,tf.float32))>0:
            lbox = self.SmoothL1Loss(tf.boolean_mask(tbox,mask),tf.boolean_mask(pbox,mask))
        else:
            lbox = tf.constant(0.0)
        pconf = entry_stop_gradients(pconf, tconf[...,tf.newaxis]<0) # stop gradient for regions labeled -1 below CONF threshold, look dataloader
        tconf = tf.cast(tf.where(tf.less(tconf, 0.0), 0.0, tconf), tf.int32)
        lconf =  self.CrossEntropyLossLogits(tconf,pconf) # apply softmax and do non negative log likelihood loss 
        if self.emb:
            tids = label[...,5:6]
            emb_mask = tf.cast(tf.math.reduce_max(tf.cast(mask, tf.float32), axis=1), tf.bool)
            tids = tf.math.reduce_max(tids, axis=1)
            tids = tids[emb_mask]
            embedding = tf.boolean_mask(emb,emb_mask)
            embedding = self.emb_scale * tf.math.l2_normalize(embedding,axis=1, epsilon=1e-12)
            if tf.shape(tids)[0]>0:
                logits = self.classifier(embedding) 
                logits = entry_stop_gradients(logits,tids<0)
                tids = tf.cast(tf.where(tf.less(tids, 0.0), 0.0, tids),tf.int32) # stop gradient for regions labeled -1
                lid = self.CrossEntropyLossLogits(tf.squeeze(tids,axis=-1),logits)
            else:
                lid = tf.constant(0.0)
            self.classifier_summary(embedding, training)
            return lbox, lconf, lid
        else:
            return lbox, lconf

    @tf.function
    def compute_loss_mrcnn(self, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_masks):
        # prepare the ground truth
        mrcnn_class_loss = mrcnn_class_loss_graph(target_class_ids, pred_class_logits)
        mrcnn_box_loss = mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox)
        mrcnn_mask_loss = mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks)
        if tf.reduce_sum(proposals)>0:
            alb_loss = tf.math.exp(-self.s_mc)*mrcnn_class_loss + tf.math.exp(-self.s_mr)*mrcnn_box_loss \
                    + tf.math.exp(-self.s_mm)*mrcnn_mask_loss + (self.s_mr + self.s_mc + self.s_mm) #Automatic Loss Balancing        
        else:
            alb_loss = mrcnn_class_loss + mrcnn_box_loss + mrcnn_mask_loss
        return alb_loss, mrcnn_class_loss, mrcnn_box_loss, mrcnn_mask_loss
   
    @tf.function
    def test_step(self, data):
        training = False
        inferring = False
        image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
        labels = [label_2, labe_3, label_4, label_5]
        if self.mask:
            preds, embs, proposals, pred_class_logits, pred_class, pred_bbox, pred_mask = self(image, training=training, inferring=inferring)
            target_class_ids, target_bbox, target_masks = +(proposals, gt_bboxes, gt_masks) # preprocess and tile labels according to IOU
            alb_total_loss, *loss_list = self.compute_loss(labels, preds, embs, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_mask, training)
        else:
            preds, embs = self(image, training=training, inferring=inferring)
            alb_total_loss, *loss_list = self.compute_loss_rpn(labels, preds, embs, training)
            
        return [alb_total_loss] + loss_list
        
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
                self.step_train += 1
                losses = self.train_step(data)
                self.loss_summary(losses, training=True)
                self.print_loss(losses, training=True)
            path = "./weights/{}_{}_{}_{}_{}_{}.tf".format(self.name,'emb' if self.emb else 'noemb','mask' if self.mask else 'nomask',self.epoch, losses[0],datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            self.save(path)
            gc.collect()
            for data in self.val_ds.take(100*self.batch).batch(self.batch):
                self.step_val += 1
                losses = self.test_step(data) 
                self.loss_summary(losses, training=False)
                self.print_loss(losses, training=False)
            gc.collect()
    
    @tf.function
    def infer(self, image):
        training = True
        inferring = True
        if self.mask:
            preds, embs, proposals, logits, probs, bboxes, masks = self(image, training=training, inferring=inferring)
        else:
            preds, embs = self(image, training=training, inferring=inferring)
            proposals = ProposalLayer()(preds, embs, training, inferring)
        
        nB = tf.shape(image)[0]
        for i in range(nB):
            proposal = proposals[i]
            if tf.reduce_sum(proposal)>0:
                valid = tf.reduce_sum(tf.cast(tf.reduce_sum(proposals, axis=-1)>0,tf.int32))
                proposal = proposal[:valid,:]           
                bboxs = proposal[...,:4]*cfg.TRAIN_SIZE
                confs = proposal[...,4] 
                tf.print('Found')
                # Final proposals are obtained in dets. Information of bounding box and embeddings also included
                # draw_bbox(image[i], bboxs, confs)
            else:
                tf.print('None')
    
    def print_loss(self, losses, training=True):
        if training:
            res = "=> STEP {}/{}  lr: {:0.5f}  auto_loss_bal: "\
                "{:0.5f}  rpn_box_loss: {:0.5f}  rpn_class_loss: {:0.5f}  "\
                "".format(self.step_train, self.step_trains, self.optimizer.lr.numpy(), 
                          losses[0], losses[1], losses[2])
            if self.emb:
                res += "rpn_id_loss: {:0.5f}  ".format(losses[3])
            if self.mask:
               res += "mrcnn_class_loss: {:0.5f}  mrcnn_box_loss: {:0.5f}"\
               "  mrcnn_mask_loss: {:0.5f}".format(losses[-3], losses[-2], losses[-1])
            tf.print(res)
        else:
            res = "=> STEP {}  auto_loss_bal: {:0.5f}  rpn_box_loss: "\
                "{:0.5f}   rpn_class_loss: {:0.5f} ".format(self.step_val, losses[0], 
                 losses[1], losses[2])
            if self.emb:
                res += "rpn_id_loss: {:0.5f}  ".format(losses[3])
            if self.mask:
                res += "mrcnn_class_loss: {:0.5f}  mrcnn_box_loss: "\
                    "{:0.5f}   mrcnn_mask_loss: {:0.5f}".format(losses[-3], losses[-2], losses[-1])
            tf.print(res)

    def loss_summary(self, losses, training=True):
        with self.writer.as_default():
            scope = 'train' if training else 'val'
            step = self.step_train if training else self.step_val
            with tf.name_scope(scope):
                with tf.name_scope('loss'):
                    tf.summary.scalar("auto_loss_bal", tf.squeeze(losses[0]), step=step)
                    tf.summary.scalar("lr", self.optimizer.lr, step=self.step_train)
                    with tf.name_scope('rpn'):
                        tf.summary.scalar("s_c", tf.squeeze(self.s_c), step=step)
                        tf.summary.scalar("s_r", tf.squeeze(self.s_r), step=step)
                        tf.summary.scalar("rpn_box_loss", tf.squeeze(losses[1]), step=step)
                        tf.summary.scalar("rpn_class_loss", tf.squeeze(losses[2]), step=step)
                        if self.emb:
                            tf.summary.scalar("s_id", tf.squeeze(self.s_id), step=step)
                            tf.summary.scalar("rpn_id_loss", tf.squeeze(losses[3]), step=step)
                    with tf.name_scope('mrcnn'): 
                        if self.mask:
                            tf.summary.scalar("s_mc", tf.squeeze(self.s_mc), step=step)
                            tf.summary.scalar("s_mr", tf.squeeze(self.s_mr), step=step)
                            tf.summary.scalar("s_mm", tf.squeeze(self.s_mm), step=step)
                            tf.summary.scalar("mrcnn_class_loss", tf.squeeze(losses[-3]), step=step)
                            tf.summary.scalar("mrcnn_box_loss", tf.squeeze(losses[-2]), step=step)
                            tf.summary.scalar("mrcnn_mask_loss", tf.squeeze(losses[-1]), step=step)
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

class BatchNorm(tf.keras.layers.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.
    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)
#
############################################################
#  Feature Pyramid Network Heads
############################################################
class fpn_classifier_AFP(tf.keras.Model):
    def __init__(self, name='fpn_classifier_AFP', **kwargs):
        super(fpn_classifier_AFP, self).__init__(name=name, **kwargs)
        self.model = self.build_graph()
        
    def call(self, input_layers, training=False):
        return self.model(input_layers)
    
    def build_graph(self):
        input_shape = self.get_input_shape()
        return tf.keras.Model(inputs=input_shape, outputs=fpn_classifier_graph_AFP(input_shape))
    
    def get_input_shape(self):
        return tf.keras.layers.Input((cfg.MAX_PROP, 4)), [tf.keras.layers.Input((cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], 
                                        cfg.EMB_DIM)) for i in range(cfg.LEVELS)]
    def get_output_shape(self):
        return [out.shape for out in self.call(self.get_input_shape())]
        
def fpn_classifier_graph_AFP(inputs, pool_size=cfg.POOL_SIZE, num_classes=2, fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.
    rois: [batch, num_rois, (x1, y1, x2, y2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers
    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dx, dy, log(dw), log(dh))] Deltas to apply to
                     proposal boxes
    """
#    rois, feature_maps = inputs[0], inputs[1]
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x2, x3, x4, x5 = PyramidROIAlign_AFP((pool_size, pool_size),name="roi_align_classifier")(inputs)
    x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding='same'), name='roi_class_afp2')(x2)
    x2 = tf.keras.layers.TimeDistributed(BatchNorm(), name='roi_class_afp2_gn')(x2)
    x2 = tf.keras.layers.Activation('relu', name='roi_class_afp2_gn_relu')(x2)
    x3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding='same'), name='roi_class_afp3')(x3)
    x3 = tf.keras.layers.TimeDistributed(BatchNorm(), name='roi_class_afp3_gn')(x3)
    x3 = tf.keras.layers.Activation('relu', name='roi_class_afp3_gn_relu')(x3)
    x4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding='same'), name='roi_class_afp4')(x4)
    x4 = tf.keras.layers.TimeDistributed(BatchNorm(), name='roi_class_afp4_gn')(x4)
    x4 = tf.keras.layers.Activation('relu', name='roi_class_afp4_gn_relu')(x4)
    x5 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding='same'), name='roi_class_afp5')(x5)
    x5 = tf.keras.layers.TimeDistributed(BatchNorm(), name='roi_class_afp5_gn')(x5)
    x5 = tf.keras.layers.Activation('relu', name='roi_class_afp5_gn_relu')(x5)

    x = tf.keras.layers.Maximum()([x2, x3, x4, x5])
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding="same"), name="mrcnn_class_conv1")(x)
    x = tf.keras.layers.TimeDistributed(BatchNorm(), name='mrcnn_class_conv1_gn')(x)
    x = tf.keras.layers.Activation('relu', name='mrcnn_class_conv1_gn_relu')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding="same"), name="mrcnn_class_conv2")(x)
    x = tf.keras.layers.TimeDistributed(BatchNorm(), name='mrcnn_class_conv2_gn')(x)
    x = tf.keras.layers.Activation('relu', name='mrcnn_class_conv2_gn_relu')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding="same"), name="mrcnn_class_conv3")(x)
    x = tf.keras.layers.TimeDistributed(BatchNorm(), name='mrcnn_class_conv3_gn')(x)
    x = tf.keras.layers.Activation('relu', name='mrcnn_class_conv3_gn_relu')(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_shared")(x)
    x = tf.keras.layers.Activation('relu', name='mrcnn_class_shared_relu')(x)
    shared = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(tf.keras.backend.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dx, dy, log(dw), log(dh))]
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dx, dy, log(dw), log(dh))]
    s = tf.keras.backend.int_shape(x)
    mrcnn_bbox = tf.keras.layers.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

class fpn_mask_AFP(tf.keras.Model):
    def __init__(self, name='fpn_mask_AFP', **kwargs):
        super(fpn_mask_AFP, self).__init__(name=name, **kwargs)
        self.model = self.build_graph()
        
    def call(self, input_layers, training=False):
        return self.model(input_layers)
    
    def build_graph(self):
        input_shape = self.get_input_shape()
        return tf.keras.Model(inputs=input_shape, outputs=build_fpn_mask_graph_AFP(input_shape))
    
    def get_input_shape(self):
        return tf.keras.layers.Input((cfg.MAX_PROP, 4)), [tf.keras.layers.Input((cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], 
                                        cfg.EMB_DIM)) for i in range(cfg.LEVELS)]
    def get_output_shape(self):
        return self.call(self.get_input_shape()).shape
    
def build_fpn_mask_graph_AFP(inputs, pool_size =cfg.MASK_POOL_SIZE , num_classes=2):
    """Builds the computation graph of the mask head of Feature Pyramid Network.
    rois: [batch, num_rois, (x1, y1, x2, y2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
#    rois, feature_maps = inputs[0], inputs[1]
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x2, x3, x4, x5 = PyramidROIAlign_AFP((pool_size, pool_size),name="roi_align_mask")(inputs)
    x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding='same'), name='roi_mask_afp2')(x2)
    x2 = tf.keras.layers.TimeDistributed(BatchNorm(), name='roi_mask_afp2_gn')(x2)
    x2 = tf.keras.layers.Activation('relu', name='roi_mask_afp2_gn_relu')(x2)
    x3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding='same'), name='roi_mask_afp3')(x3)
    x3 = tf.keras.layers.TimeDistributed(BatchNorm(), name='roi_mask_afp3_gn')(x3)
    x3 = tf.keras.layers.Activation('relu', name='roi_mask_afp3_gn_relu')(x3)
    x4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding='same'), name='roi_mask_afp4')(x4)
    x4 = tf.keras.layers.TimeDistributed(BatchNorm(), name='roi_mask_afp4_gn')(x4)
    x4 = tf.keras.layers.Activation('relu', name='roi_mask_afp4_gn_relu')(x4)
    x5 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding='same'), name='roi_mask_afp5')(x5)
    x5 = tf.keras.layers.TimeDistributed(BatchNorm(), name='roi_mask_afp5_gn')(x5)
    x5 = tf.keras.layers.Activation('relu', name='roi_mask_afp5_gn_relu')(x5)

    x = tf.keras.layers.Maximum()([x2, x3, x4, x5])
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = tf.keras.layers.TimeDistributed(BatchNorm(), name='mrcnn_mask_gn1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = tf.keras.layers.TimeDistributed(BatchNorm(), name='mrcnn_mask_gn2')(x)
    shared = tf.keras.layers.Activation('relu')(x)

    x_fcn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(shared)
    x_fcn = tf.keras.layers.TimeDistributed(BatchNorm(), name='mrcnn_mask_gn3')(x_fcn)
    x_fcn = tf.keras.layers.Activation('relu')(x_fcn)
    x_fcn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(cfg.EMB_DIM, (2, 2), strides=(2, 2), activation="relu"),
                           name="mrcnn_mask_deconv")(x_fcn)

    x_ff = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(shared)
    x_ff = tf.keras.layers.TimeDistributed(BatchNorm(), name='mrcnn_mask_gn4')(x_ff)
    x_ff = tf.keras.layers.Activation('relu')(x_ff)
    x_ff = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM//2, (3, 3), padding="same"),
                              name="mrcnn_mask_conv5")(x_ff)
    x_ff = tf.keras.layers.TimeDistributed(BatchNorm(), name='mrcnn_mask_gn5')(x_ff)
    x_ff = tf.keras.layers.Activation('relu')(x_ff)
    x_ff_shape = tf.keras.backend.int_shape(x_ff)
    x_ff = tf.keras.layers.Reshape((x_ff_shape[1], x_ff_shape[2]*x_ff_shape[3]*x_ff_shape[4]))(x_ff)
    x_ff = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(pool_size*pool_size*2*2, activation='relu'), name='mrcnn_mask_fc')(x_ff)

    x_fcn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(num_classes, (1, 1), strides=1), name="mrcnn_mask_fcn")(x_fcn)
    x_ff = tf.keras.layers.Reshape((x_ff_shape[1], pool_size*2, pool_size*2, 1))(x_ff)
    x_ff = tf.keras.layers.Lambda(lambda x: tf.tile(x, (1, 1, 1, 1, num_classes)))(x_ff)
    x = tf.keras.layers.Add()([x_fcn, x_ff])
    x = tf.keras.layers.Activation('sigmoid', name='mrcnn_mask')(x)

    return x

