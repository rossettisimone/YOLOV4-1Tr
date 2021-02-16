#! /usr/bin/env python
# coding=utf-8
import gc
import tensorflow as tf
import tensorflow_addons as tfa
import config as cfg
from backbone import cspdarknet53
from layers import CustomUpsampleAndConcatAndShuffle, CustomDownsampleAndConcatAndShuffle, CustomDecode, CustomProposalLayer
from utils import entry_stop_gradients, preprocess_mrcnn, mrcnn_class_loss_graph, mrcnn_bbox_loss_graph, mrcnn_mask_loss_graph, draw_bbox, smooth_l1_loss, decode_delta, xyxy2xywh, xywh2xyxy, show_image, filter_inputs
from datetime import datetime
from compute_ap import compute_ap_range
from layers import build_fpn_mask_graph_AFP, fpn_classifier_graph_AFP

class fpn(tf.keras.Model):
    def __init__(self, name='fpn', **kwargs):
        super(fpn, self).__init__(name=name, **kwargs)
        
        self.up_1 = CustomUpsampleAndConcatAndShuffle(filters=256, n=1)
        self.up_2 = CustomUpsampleAndConcatAndShuffle(filters=128, n=2)
        self.up_3 = CustomUpsampleAndConcatAndShuffle(filters=64, n=3)
        self.down_3 = CustomDownsampleAndConcatAndShuffle(filters=128, n=3)
        self.down_2 = CustomDownsampleAndConcatAndShuffle(filters=256, n=2)
        self.down_1 = CustomDownsampleAndConcatAndShuffle(filters=512, n=1)

    def call(self, input_layers, training=False):
        b_2, b_3, b_4, b_5 = input_layers 
        p_5 = b_5
        p_4 = self.up_1(p_5,b_4, training)
        p_3 = self.up_2(p_4,b_3, training)
        p_2 = self.up_3(p_3,b_2, training)
        n_2 = p_2
        n_3 = self.down_3(n_2,p_3, training)
        n_4 = self.down_2(n_3,p_4, training)
        n_5 = self.down_1(n_4,p_5, training)
        return  n_2, n_3, n_4, n_5 
    
    def build_graph(self):
        inputs = self.get_input_shape()
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
    
    def get_input_shape(self):
        return [tf.keras.Input(shape=(cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], 
                                        128*2**(i-(0 if i<cfg.LEVELS-1 else 1)))) for i in range(cfg.LEVELS)]
    def get_output_shape(self):
        return self.call(self.get_input_shape())
    
    
class rpn(tf.keras.Model):
    def __init__(self, name='rpn', **kwargs):
        super(rpn, self).__init__(name=name, **kwargs)
        
        self.heads = [CustomDecode(level, n=level+1) for level in range(cfg.LEVELS)]
        
    def call(self, input_layers, training=False):
        preds = []
        embs = []
        for i in range(cfg.LEVELS):
            x = self.heads[i](input_layers[i], training)
            preds.append(x[0])
            embs.append(x[1])
        return preds, embs

    def get_input_shape(self):
        return [tf.keras.Input(shape=(cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], 
                                        64*2**i)) for i in range(cfg.LEVELS)]
    def get_output_shape(self):
        preds, embs = self.call(self.get_input_shape())
        return [out.shape for out in preds], [out.shape for out in embs]
    
    def build_graph(self):
        inputs = self.get_input_shape()
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
    
class MSDS(tf.keras.Model): #MSDS, multi subject detection and segmentation
    def __init__(self, emb = True, mask = True, freeze_bkbn = True, freeze_bn = False, data_loader = None, name='MSDS', **kwargs):
        super(MSDS, self).__init__(name=name, **kwargs)
        
        self.nID = 972
        self.mask = mask
        self.emb = emb
        if data_loader is not None:
            self.ds = data_loader
            self.nID =  self.ds.nID
            self.epochs = cfg.EPOCHS
            self.epoch = 1
            self.step_train = 1
            self.step_val = 1
            self.batch = cfg.BATCH
            self.steps_train = cfg.STEPS_PER_EPOCH_TRAIN
            self.steps_val = cfg.STEPS_PER_EPOCH_VAL
            self.folder = "{}".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            self.writer = tf.summary.create_file_writer("./{}/logdir".format(self.folder))
        self.freeze_bkbn = freeze_bkbn
        self.freeze_bn = freeze_bn
        self.LEVELS = cfg.LEVELS
        self.NUM_ANCHORS = cfg.NUM_ANCHORS
        self.ANCHORS = tf.reshape(tf.constant(cfg.ANCHORS,dtype=tf.float32),[self.LEVELS, self.NUM_ANCHORS, 2])
        self.STRIDES = tf.constant(cfg.STRIDES,dtype=tf.float32)
        self.emb_dim = cfg.EMB_DIM 
        self.emb_scale = (tf.math.sqrt(2.0) * tf.math.log(self.nID-1.0)) if self.nID>1.0 else 1.0
        self.optimizer = tfa.optimizers.SGDW( weight_decay = cfg.WD, learning_rate = cfg.LR, momentum = cfg.MOM, nesterov = False, clipnorm = cfg.GRADIENT_CLIP) #tf.keras.optimizers.Adam(learning_rate = cfg.LR)

        #CSPDARKNET53 
        self.bkbn = cspdarknet53(pretrained = self.freeze_bkbn) # if pretrained freeze backbone
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
            self.proposal = CustomProposalLayer() #proposal + embedding
            self.fpn_classifier = fpn_classifier_AFP()            
            self.fpn_mask = fpn_mask_AFP()
            self.s_mc = tf.Variable(initial_value=0.0, trainable=True) 
            self.s_mr = tf.Variable(initial_value=0.0, trainable=True)
            self.s_mm = tf.Variable(initial_value=0.0, trainable=True)
    
    def call(self, input_layers, training=False):
        features = self.bkbn(input_layers, training)
        pyramid = self.neck(features, training)
        rpn_pred, rpn_embeddings = self.head(pyramid, training) # (proposal, embedding),...(proposal_embedding) for each piramid level
        if self.mask:
            if self.emb:
                rpn_proposals = self.proposal(rpn_pred, rpn_embeddings) # p,e = proposals, embeddings [batch, rois, (x1, y1, x2, y2, embeddings...)]
            else:
                rpn_proposals = self.proposal(rpn_pred) # p,e = proposals, embeddings [batch, rois, (x1, y1, x2, y2, embeddings...)]
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
        image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
        labels = [label_2, labe_3, label_4, label_5]
        with tf.GradientTape() as tape:
            if self.mask:
                preds, embs, proposals, pred_class_logits, pred_class, pred_bbox, pred_mask = self(image, training=training)
                proposals = proposals[...,:4]
                target_class_ids, target_bbox, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks) # preprocess and tile labels according to IOU
                alb_total_loss, *loss_list = self.compute_loss(labels, preds, embs, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_mask, training)
            else:
                preds, embs = self(image, training=training)
                alb_total_loss, *loss_list = self.compute_loss_rpn(labels, preds, embs, training)
            gradients = tape.gradient(alb_total_loss, self.trainable_variables)
            self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.trainable_variables) if grad is not None)
        return [alb_total_loss] + loss_list
    
    def compute_loss(self, labels, preds, embs, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_masks, training):
        # rpn loss 
        alb_total_loss, *rpn_loss_list = self.compute_loss_rpn(labels, preds, embs, training)
        # mrcnn loss
        alb_loss, *mrcnn_loss_list = self.compute_loss_mrcnn(proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_masks)
        alb_total_loss += alb_loss
        alb_total_loss *= 0.5

        return [alb_total_loss] + rpn_loss_list + mrcnn_loss_list

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

    def compute_loss_rpn_level(self, label, pred, emb, training):
        pbox = pred[..., :4]
        pconf = pred[..., 4:6]
        tbox = label[...,:4]
        tconf = label[...,4]
        mask = tf.greater(tconf,0.0)
        if tf.greater(tf.reduce_sum(tf.cast(mask,tf.float32)),0.0):
            lbox = tf.reduce_mean(smooth_l1_loss(y_true = tf.boolean_mask(tbox,mask),y_pred = tf.boolean_mask(pbox,mask)))
        else:
            lbox = tf.constant(0.0)
        non_negative_entry = tf.cast(tf.greater_equal(tconf[...,tf.newaxis],0.0),tf.float32)
        pconf = entry_stop_gradients(pconf, non_negative_entry) # stop gradient for regions labeled -1 below CONF threshold, look dataloader
        tconf = tf.cast(tf.where(tf.less(tconf, 0.0), 0.0, tconf),tf.int32)
        lconf =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tconf,logits=pconf)) # apply softmax and do non negative log likelihood loss 
        if self.emb:
            tids = label[...,5:6]
            emb_mask = tf.cast(tf.math.reduce_max(tf.cast(mask, tf.float32), axis=1), tf.bool)
            tids = tf.math.reduce_max(tids, axis=1)
            tids = tids[emb_mask]
            embedding = tf.boolean_mask(emb,emb_mask)
            embedding = self.emb_scale * tf.math.l2_normalize(embedding,axis=1, epsilon=1e-12)
            if tf.greater(tf.shape(tids)[0],0.0):
                logits = self.classifier(embedding) 
                non_negative_entry = tf.greater_equal(tids,0.0)
                logits = entry_stop_gradients(logits,tf.cast(non_negative_entry,tf.float32))
                tids = tf.cast(tf.where(tf.logical_not(non_negative_entry), 0.0, tids),tf.int32) # stop gradient for regions labeled -1
                lid = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.squeeze(tids,axis=-1),logits))
            else:
                lid = tf.constant(0.0)
            self.classifier_summary(embedding, training)
            return lbox, lconf, lid
        else:
            return lbox, lconf

    def compute_loss_mrcnn(self, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_masks):
        # prepare the ground truth
        mrcnn_class_loss = mrcnn_class_loss_graph(target_class_ids, pred_class_logits)
        mrcnn_box_loss = mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox)
        mrcnn_mask_loss = mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks)
        if tf.greater(tf.reduce_sum(proposals),0.0) :
            alb_loss = tf.math.exp(-self.s_mc)*mrcnn_class_loss + tf.math.exp(-self.s_mr)*mrcnn_box_loss \
                    + tf.math.exp(-self.s_mm)*mrcnn_mask_loss + (self.s_mr + self.s_mc + self.s_mm) #Automatic Loss Balancing        
        else:
            alb_loss = mrcnn_class_loss + mrcnn_box_loss + mrcnn_mask_loss
        return alb_loss, mrcnn_class_loss, mrcnn_box_loss, mrcnn_mask_loss
   
    @tf.function
    def test_step(self, data):
        training = False
        image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
        labels = [label_2, labe_3, label_4, label_5]
        if self.mask:
            predictions = self(image, training=training)
            preds, embs, proposals, pred_class_logits, pred_class, pred_bbox, pred_mask = predictions
            proposals = proposals[...,:4]
            target_class_ids, target_bbox, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks) # preprocess and tile labels according to IOU
            alb_total_loss, *loss_list = self.compute_loss(labels, preds, embs, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_mask, training)
        else:
            predictions = self(image, training=training)
            preds, embs = predictions
            alb_total_loss, *loss_list = self.compute_loss_rpn(labels, preds, embs, training)
        
        return [alb_total_loss] + loss_list, predictions
        
    def fit(self, epochs = None, start_epoch = 0):
        assert self.ds is not None, 'DataLoader is required'
        if epochs is not None:
            self.epochs = epochs
        self.start_epoch = start_epoch
        # train_generator = self.ds.train_ds.repeat().filter(filter_inputs).batch(self.batch).prefetch(tf.data.experimental.AUTOTUNE).__iter__()
        # val_generator = self.ds.val_ds.repeat().filter(filter_inputs).batch(self.batch).prefetch(tf.data.experimental.AUTOTUNE)
        train_generator = self.ds.train_ds.repeat()\
                .filter(filter_inputs).batch(self.batch).apply(tf.data.experimental.copy_to_device("/gpu:0"))\
                .prefetch(tf.data.experimental.AUTOTUNE).__iter__()
        val_generator = self.ds.val_ds.repeat()\
                .filter(filter_inputs).batch(self.batch).apply(tf.data.experimental.copy_to_device("/gpu:0"))\
                .prefetch(tf.data.experimental.AUTOTUNE)
        
        while self.epoch < self.epochs:
            if self.freeze_bkbn and self.epoch < 2:
                self.bkbn.trainable = False
            else:
                self.bkbn.trainable = True
            if self.freeze_bn: # finetuning 
                self.freeze_batch_normalization() 
            # self.adapt_lr()
            self.set_trainable(trainable = True, include_backbone = False)
            while self.step_train < self.epoch * self.steps_train :
                data = train_generator.next()
                losses = self.train_step(data)
                self.loss_summary(losses, training=True)
                self.print_loss(losses, training=True)
                if self.mask:
                    self.denses_summary(training = True)
                self.step_train += 1
            self.set_trainable(trainable = False, include_backbone = True)
            mean_AP = []
            gc.collect()
            val = val_generator.__iter__() # use always same batchs
            while self.step_val < self.epoch * self.steps_val :
                data = val.next()
                losses, predictions = self.test_step(data) 
                mean_AP.append(show_mAP(data, predictions))
                self.loss_summary(losses, training=False)
                self.print_loss(losses, training=False)
                if self.mask:
                    self.denses_summary(training = False)
                    self.mAP_summary(tf.reduce_mean(mean_AP))
                self.step_val += 1
            path = "./{}/weights/{}_{}_{}_{}_{:0.5f}_{}.tf".format(self.folder,self.name,'emb' if self.emb else 'noemb','mask' if self.mask else 'nomask',self.epoch, tf.reduce_mean(mean_AP),datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            self.save(path)
            gc.collect()
            self.epoch += 1
    
    @tf.function
    def infer(self, image):
        if self.mask:
            preds, embs, proposals, logits, probs, bboxes, masks = self(image, training = False)
            return preds, embs, proposals, logits, probs, bboxes, masks
        else:
            preds, embs = self(image, training = False)
            proposals = CustomProposalLayer()(preds, embs, training = False)
            return preds, embs, proposals
        
    def print_loss(self, losses, training=True):
        if training:
            res = "=> EPOCH {}  TRAIN STEP {}/{}  lr: {:0.5f}  auto_loss_bal: "\
                "{:0.5f}  rpn_box_loss: {:0.5f}  rpn_class_loss: {:0.5f}  "\
                "".format(self.epoch, self.step_train % self.steps_train, self.steps_train, self.optimizer.lr.numpy(), 
                          losses[0], losses[1], losses[2])
            if self.emb:
                res += "rpn_id_loss: {:0.5f}  ".format(losses[3])
            if self.mask:
               res += "mrcnn_class_loss: {:0.5f}  mrcnn_box_loss: {:0.5f}"\
               "  mrcnn_mask_loss: {:0.5f}".format(losses[-3], losses[-2], losses[-1])
            tf.print(res)
        else:
            res = "=> EPOCH {}  VAL STEP {}/{}  auto_loss_bal: {:0.5f}  rpn_box_loss: "\
                "{:0.5f}   rpn_class_loss: {:0.5f} ".format(self.epoch, self.step_val % self.steps_val, self.steps_val,
                 losses[0], losses[1], losses[2])
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

    def denses_summary(self, training):
        with self.writer.as_default():
            scope = 'train' if training else 'val'
            step = self.step_train if training else self.step_val
            with tf.name_scope(scope):
                for layer in self.fpn_classifier.layers[0].layers:
                    if layer.name == 'mrcnn_class_logits' or layer.name == 'mrcnn_bbox_fc':
                        with tf.name_scope(layer.name):
                            with tf.name_scope('weights'):
                                self.variable_summaries(layer.layer.kernel,step)
                            with tf.name_scope('biases'):
                                self.variable_summaries(layer.layer.bias,step)
                for layer in self.fpn_mask.layers[0].layers:
                    if layer.name == 'mrcnn_mask_fc':
                        with tf.name_scope(layer.name):
                            with tf.name_scope('weights'):
                                self.variable_summaries(layer.layer.kernel,step)
                            with tf.name_scope('biases'):
                                self.variable_summaries(layer.layer.bias,step)
        self.writer.flush()
    
    def mAP_summary(self, mean_AP):
        with self.writer.as_default():
            with tf.name_scope('val'):
                with tf.name_scope('mAP_0.5:0.05:0.95'):
                    tf.summary.scalar("mean_AP", tf.squeeze(mean_AP), step=self.step_val)

    def set_trainable(self, trainable = True, include_backbone = False):
        start = 0 if include_backbone else 1
        for net in self.layers[start:]:
            net.trainable = trainable
        self.s_c._trainable = trainable
        self.s_r._trainable = trainable
        if self.emb:
            self.s_id._trainable = trainable
        if self.mask:
            self.s_mc._trainable = trainable
            self.s_mr._trainable = trainable
            self.s_mm._trainable = trainable
            
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
        tf.keras.utils.plot_model(self.neck.build_graph(), to_file='panet.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
        tf.keras.utils.plot_model(self.head.build_graph(), to_file='yolov4.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
        tf.keras.utils.plot_model(self.fpn_classifier.build_graph(), to_file='mrcnn_box.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
        tf.keras.utils.plot_model(self.fpn_mask.build_graph(), to_file='mrcnn_mask.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
        tf.keras.utils.plot_model(CustomProposalLayer().build_graph(), to_file='prop.png', show_shapes=False,  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)

    def custom_build(self):
        tf.summary.trace_on(graph=True, profiler=False)
        self.build((tf.newaxis, cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3))
        with self.writer.as_default():
            tf.summary.trace_export(
                    name="model",
                    step=self.step_train,
                    profiler_outdir="./{}/logdir".format(self.folder))
         
    def adapt_lr(self):
        if self.epoch < self.epochs * 0.5 :
            lr = cfg.LR
        elif self.epoch <= self.epochs * 0.75 and self.epoch > self.epochs * 0.5:
            lr = cfg.LR * 0.1
        else:
            lr = cfg.LR * 0.01
        self.optimizer.lr.assign(lr)
        
    def save(self, name):
        self.save_weights(name)

    def load(self, name):
        self.load_weights(name)

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


