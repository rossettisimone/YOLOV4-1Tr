#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
from backbone import cspdarknet53
from layers import CustomUpsampleAndConcatAndShuffle, CustomDownsampleAndConcatAndShuffle, CustomDecode,ProposalLayer, PyramidROIAlign, PyramidROIAlign_AFP
from PIL import Image, ImageDraw, ImageFont
import gc
from utils import nms_proposals, batch_bbox_iou, xywh2xyxy
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
        for f, n in zip(self.heads ,input_layers):
            x = f(n, training, inferring)
            preds.append(x[0])
            embs.append(x[1])
#        preds_embs = [(f(n, training, inferring)) for f, n in zip(self.heads ,input_layers)]
#        preds = [pred for pred, emb in preds_embs]
#        preds = [emb for pred, emb in preds_embs]
        return preds, embs

    def get_input_shape(self):
        return [tf.keras.Input(shape=(cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], 
                                        64*2**i)) for i in range(cfg.LEVELS)]
    def get_output_shape(self):
        preds, embs = self.call(self.get_input_shape())
        return [out.shape for out in preds], [out.shape for out in embs]
    
class tracker(tf.keras.Model):
    def __init__(self, freeze_bkbn = True, freeze_bn = False, data_loader = None, name='tracker', **kwargs):
        super(tracker, self).__init__(name=name, **kwargs)
        
        self.nID = 972
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
        
        self.proposal = ProposalLayer()
                
        self.fpn_classifier = fpn_classifier_AFP()
        
        self.fpn_mask = fpn_mask_AFP()
        
        self.classifier = tf.keras.layers.Dense(self.nID) if self.nID>0 else None
        self.classifier.build((tf.newaxis,self.emb_dim))
        self.s_c = tf.Variable(initial_value=[-4.15], trainable=True) 
        self.s_r = tf.Variable(initial_value=[-4.85], trainable=True)
        self.s_id = tf.Variable(initial_value=[-2.3], trainable=True)
        self.s_cc = tf.Variable(initial_value=[-4.15], trainable=True) 
        self.s_rr = tf.Variable(initial_value=[-4.85], trainable=True)
        self.s_mm = tf.Variable(initial_value=[-2.3], trainable=True)
        
    def call(self, input_layers, training=False, inferring=False):
        b = self.bkbn(input_layers, training)
        n = self.neck(b, training)
        p, e = self.head(n, training, inferring) # (proposal, embedding),...(proposal_embedding) for each piramid level
        r = self.proposal(p, training, inferring) # p,e = proposals, embeddings [batch, rois, (x1, y1, x2, y2, embeddings...)]
        bb = r[...,:4] # take only boxes, not confs
        logits, probs, bboxes = self.fpn_classifier([bb,e])
        masks = self.fpn_mask([bb,e])
        
        #compute_loss_rpn(*p,*e)
        # box prediction layer 
        # mask prediciton layer
#        if training and inferring:
#            # we havefor each level proposals for each image in the batch, thus to be more flexible we transpose the list of lists
#            # pythonic transpose list of list of irregular size: [[image1 - fpn1, image2 - fpn1, ..], [image1 - fpn2], [image2 - fpn2], ..]
#            l = [len(i) for i in h]
#            h = [[i[o] for ix, i in enumerate(h) if l[ix] > o] for o in range(max(l))] 
#            h = [tf.concat(p, axis=0) for p in h] # concat proposals from all levels for each image in batch
        return p, e, bb, logits, probs, bboxes, masks
    
    def get_input_shape(self):
        return tf.zeros((cfg.BATCH,cfg.TRAIN_SIZE,cfg.TRAIN_SIZE, 3))
    
    def get_output_shape(self):
        return [out.shape for out in self.call(self.get_input_shape())]
        
    def train_step(self, data):
        self.step_train += 1
        training = True
        inferring = False
        image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
        labels = [label_2, labe_3, label_4, label_5]
        with tf.GradientTape() as tape:
            preds, embs, proposals, logits, probs, bboxes, masks = self(image, training=training, inferring=inferring)
#            print(proposals)
            self.mean_total_loss = self.compute_loss(labels, preds, embs, proposals, logits, probs, bboxes, masks, gt_bboxes, gt_masks, training)
#            self.total_loss = [] 
#            self.box_loss = []
#            self.conf_loss = []
#            self.id_loss = []
#            for label, (pred, emb) in zip(labels, preds):
#                loss, lbox, lconf, lid = self.compute_loss(label, pred, emb, training)
#                self.total_loss.append(loss)
#                self.box_loss.append(lbox)
#                self.conf_loss.append(lconf)
#                self.id_loss.append(lid)
#            self.mean_total_loss = tf.reduce_mean(tf.concat(self.total_loss, axis=0),axis=0)
#            self.mean_box_loss = tf.reduce_mean(tf.concat(self.box_loss, axis=0),axis=0)
#            self.mean_conf_loss = tf.reduce_mean(tf.concat(self.conf_loss, axis=0),axis=0)
#            self.mean_id_loss = tf.reduce_mean(tf.concat(self.id_loss, axis=0),axis=0)
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(self.mean_total_loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            self.loss_summary(training)
        tf.print("=> STEP %4d/%4d   lr: %.6f   mean_box_loss: %4.2f   mean_conf_loss: %4.2f   "
                     "mean_id_loss: %4.2f   mean_total_loss: %4.2f" % (self.step_train, self.step_trains, \
                                                             self.optimizer.lr.numpy(), self.mean_box_loss, \
                                                             self.mean_conf_loss, self.mean_id_loss, self.mean_total_loss))
    def compute_loss(self, labels, preds, embs, proposals, logits, pred_class_logits, pred_bbox, pred_masks, gt_bboxes, gt_masks, training):
        self.total_loss = [] 
        self.box_loss = []
        self.conf_loss = []
        self.id_loss = []
        for label, pred, emb in zip(labels, preds, embs):
            lbox, lconf, lid = self.compute_loss_rpn(label, pred, emb, training)
            self.box_loss.append(lbox)
            self.conf_loss.append(lconf)
            self.id_loss.append(lid)
        lbox = self.mean_box_loss = tf.reduce_mean(self.box_loss,axis=0)
        lconf = self.mean_conf_loss = tf.reduce_mean(self.conf_loss,axis=0)
        lid = self.mean_id_loss = tf.reduce_mean(self.id_loss,axis=0)
        lmclass, lmbox, lmmask = self.compute_loss_mrcnn(proposals, logits, pred_class_logits, pred_bbox, pred_masks, gt_bboxes, gt_masks)
        loss = tf.math.exp(-self.s_r)*lbox + tf.math.exp(-self.s_c)*lconf \
                + tf.math.exp(-self.s_id)*lid + tf.math.exp(-self.s_cc)*lmclass \
                + tf.math.exp(-self.s_rr)*lmbox + tf.math.exp(-self.s_mm)*lmmask \
                + (self.s_r + self.s_c + self.s_id + self.s_rr + self.s_cc + self.s_mm) #Automatic Loss Balancing
        loss *= 0.5
        return loss
    
#    @tf.function
    def compute_loss_rpn(self, label, pred, emb, training):
        # 5 losses
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

        return lbox, lconf, lid
    
    def compute_loss_mrcnn(self, proposals, logits, pred_class_logits, pred_bbox, pred_masks, gt_bboxes, gt_masks):
#        gt_bboxes = tf.random.shuffle(tf.random.uniform((5,20,4)))*cfg.TRAIN_SIZE
#        gt_masks = tf.random.uniform((5,20,28,28))
#        proposals = tf.random.shuffle(tf.random.uniform((5,200,4)))
#        pred_bbox = tf.random.shuffle(tf.random.uniform((5,200,2,4)))
#        pred_masks = tf.random.shuffle(tf.random.uniform((5,200,28,28,2)))
#        pred_class_logits = tf.random.shuffle(tf.random.uniform((5,200,2)))
        valid = tf.reduce_sum(proposals,axis=-1)>0
        nB, nP, _ = proposals.shape
        target_class_ids = tf.where(valid, 1, 0)# exclude zero padded
        active_class_ids = tf.ones((nB,2))
        gt_bboxes /= cfg.TRAIN_SIZE
        gt_intersect = batch_bbox_iou(proposals, gt_bboxes, x1y1x2y2=True) # batch , proposals , gt_bboxes
        gt_indices =  tf.math.argmax(gt_intersect,axis=-1)
        target_bbox = []
        for i,j in zip(gt_bboxes, gt_indices):
            target_bbox.append(tf.gather(i,j,axis=0))
        target_bbox = tf.stack(target_bbox, axis=0)
        
        target_masks = []
        for i,j in zip(gt_masks, gt_indices):
            target_masks.append(tf.gather(i,j,axis=0))
        target_masks = tf.stack(target_masks, axis=0)
        
        loss_class = self.mrcnn_class_loss_graph(target_class_ids, pred_class_logits, active_class_ids)
        loss_bbox = self.mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox)
        loss_mask = self.mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks)
        
        return loss_class, loss_bbox, loss_mask
        
    def mrcnn_class_loss_graph(self,target_class_ids, pred_class_logits,
                               active_class_ids):
        """Loss for the classifier head of Mask RCNN.
        target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
            padding to fill in the array.
        pred_class_logits: [batch, num_rois, num_classes]
        active_class_ids: [batch, num_classes]. Has a value of 1 for
            classes that are in the dataset of the image, and 0
            for classes that are not in the dataset.
        """
        # During model building, Keras calls this function with
        # target_class_ids of type float32. Unclear why. Cast it
        # to int to get around it.
        target_class_ids = tf.cast(target_class_ids, 'int32')
    
        # Find predictions of classes that are not in the dataset.
        pred_class_ids = tf.argmax(pred_class_logits, axis=2)
        # TODO: Update this line to work with batch > 1. Right now it assumes all
        #       images in a batch have the same active_class_ids
        pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    
        # # Loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_class_ids, logits=pred_class_logits)
    
        # Erase losses of predictions of classes that are not in the active
        # classes of the image.
        loss = loss * pred_active
    
        # Computer loss mean. Use only predictions that contribute
        # to the loss to get a correct mean.
        loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
        return loss

    def smooth_l1_loss(self,y_true, y_pred):
        """Implements Smooth-L1 loss.
        y_true and y_pred are typically: [N, 4], but could be any shape.
        """
        diff = tf.keras.backend.abs(y_true - y_pred)
        less_than_one = tf.keras.backend.cast(tf.keras.backend.less(diff, 1.0), "float32")
        loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
        return loss
    
    def mrcnn_bbox_loss_graph(self,target_bbox, target_class_ids, pred_bbox):
        """Loss for Mask R-CNN bounding box refinement.
        target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
        target_class_ids: [batch, num_rois]. Integer class IDs.
        pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
        """
        # Reshape to merge batch and roi dimensions for simplicity.
        target_class_ids = tf.keras.backend.reshape(target_class_ids, (-1,))
        target_bbox = tf.keras.backend.reshape(target_bbox, (-1, 4))
        pred_bbox = tf.keras.backend.reshape(pred_bbox, (-1, tf.keras.backend.int_shape(pred_bbox)[2], 4))
    
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indices.
        positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = tf.cast(
            tf.gather(target_class_ids, positive_roi_ix), tf.int64)
        indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)
    
        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = tf.gather(target_bbox, positive_roi_ix)
        pred_bbox = tf.gather_nd(pred_bbox, indices)
    
        # Smooth-L1 Loss
        loss = tf.keras.backend.switch(tf.size(target_bbox) > 0,
                        self.smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                        tf.constant(0.0))
        loss = tf.keras.backend.mean(loss)
        return loss
    
    
    def mrcnn_mask_loss_graph(self,target_masks, target_class_ids, pred_masks):
        """Mask binary cross-entropy loss for the masks head.
        target_masks: [batch, num_rois, height, width].
            A float32 tensor of values 0 or 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                    with values from 0 to 1.
        """
        # Reshape for simplicity. Merge first two dimensions into one.
        target_class_ids = tf.keras.backend.reshape(target_class_ids, (-1,))
        mask_shape = tf.shape(target_masks)
        target_masks = tf.keras.backend.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
        pred_shape = tf.shape(pred_masks)
        pred_masks = tf.keras.backend.reshape(pred_masks,
                               (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
        # Permute predicted masks to [N, num_classes, height, width]
        pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])
    
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = tf.where(target_class_ids > 0)[:, 0]
        positive_class_ids = tf.cast(
            tf.gather(target_class_ids, positive_ix), tf.int64)
        indices = tf.stack([positive_ix, positive_class_ids], axis=1)
    
        # Gather the masks (predicted and true) that contribute to loss
        y_true = tf.gather(target_masks, positive_ix)
        y_pred = tf.gather_nd(pred_masks, indices)
    
        # Compute binary cross entropy. If no positive ROIs, then return 0.
        # shape: [batch, roi, num_classes]
        loss = tf.keras.backend.switch(tf.size(y_true) > 0,
                        tf.keras.backend.binary_crossentropy(target=y_true, output=y_pred),
                        tf.constant(0.0))
        loss = tf.keras.backend.mean(loss)
        return loss

        
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
            loss, lbox, lconf, lid = self.compute_loss_rpn(label, pred, emb, training)
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
            self.save('./weights/tracker_weights_'+str(self.epoch)+'.tf')
            gc.collect()
            for data in self.val_ds.take(100*self.batch).batch(self.batch):
                self.test_step(data)               
            gc.collect()
    
    def infer(self, image):
        training = True
        inferring = True
        proposals = self(image, training=training, inferring=inferring)
        proposals = nms_proposals(proposals)
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
            draw.text(xy, str(np.round(conf,3)), font=ImageFont.truetype("./other/arial.ttf"))
        img.show() 
#        plt.imshow(np.array(img))
#        plt.show()
        
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
    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
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
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
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
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
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
    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
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


#class fpn_classifier(tf.keras.Model):
#    def __init__(self, name='fpn_classifier', **kwargs):
#        super(fpn_classifier, self).__init__(name=name, **kwargs)
#        self.model = self.build_graph()
#        
#    def call(self, input_layers, training=False):
#        return self.model(input_layers)
#    
#    def build_graph(self):
#        input_shape = self.get_input_shape()
#        return tf.keras.Model(inputs=input_shape, outputs=fpn_classifier_graph(input_shape))
#    
#    def get_input_shape(self):
#        return tf.keras.layers.Input((cfg.MAX_PROP, cfg.ALIGN_H, cfg.ALIGN_W, cfg.EMB_DIM))
#        
#    def get_output_shape(self):
#        return [out.shape for out in self.call(self.get_input_shape())]
#        
##def fpn_classifier_graph(rois, feature_maps, image_meta,
##                         pool_size, num_classes, train_bn=True,
##                         fc_layers_size=1024):
#
#def fpn_classifier_graph(rois, pool_size=7, num_classes=1, train_bn=True,
#                         fc_layers_size=1024):
#    """Builds the computation graph of the feature pyramid network classifier
#    and regressor heads.
#    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
#          coordinates.
#    feature_maps: List of feature maps from different layers of the pyramid,
#                  [P2, P3, P4, P5]. Each has a different resolution.
#    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
#    pool_size: The width of the square feature map generated from ROI Pooling.
#    num_classes: number of classes, which determines the depth of the results
#    train_bn: Boolean. Train or freeze Batch Norm layers
#    fc_layers_size: Size of the 2 FC layers
#    Returns:
#        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
#        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
#        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
#                     proposal boxes
#    """
#    # ROI Pooling
#    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
##    x = PyramidROIAlign([pool_size, pool_size],
##                        name="roi_align_classifier")([rois, image_meta] + feature_maps)
#    # Two 1024 FC layers (implemented with Conv2D for consistency)
#    x=rois
#    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
#                           name="mrcnn_class_conv1")(x)
#    x = tf.keras.layers.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
#    x = tf.keras.layers.Activation('relu')(x)
#    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(fc_layers_size, (1, 1)),
#                           name="mrcnn_class_conv2")(x)
#    x = tf.keras.layers.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
#    x = tf.keras.layers.Activation('relu')(x)
#
#    shared = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(tf.keras.backend.squeeze(x, 3), 2),
#                       name="pool_squeeze")(x)
#
#    # Classifier head
#    mrcnn_class_logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes),
#                                            name='mrcnn_class_logits')(shared)
#    mrcnn_probs = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("softmax"),
#                                     name="mrcnn_class")(mrcnn_class_logits)
#
#    # BBox head
#    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
#    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes * 4, activation='linear'),
#                           name='mrcnn_bbox_fc')(shared)
#    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
#    s = tf.keras.backend.int_shape(x)
#    mrcnn_bbox = tf.keras.layers.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)
#
#    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox
#
#class fpn_mask(tf.keras.Model):
#    def __init__(self, name='fpn_mask', **kwargs):
#        super(fpn_mask, self).__init__(name=name, **kwargs)
#        self.model = self.build_graph()
#        
#    def call(self, input_layers, training=False):
#        return self.model(input_layers)
#    
#    def build_graph(self):
#        input_shape = self.get_input_shape()
#        return tf.keras.Model(inputs=input_shape, outputs=fpn_mask_graph(input_shape))
#    
#    def get_input_shape(self):
#        return tf.keras.layers.Input((cfg.MAX_PROP, cfg.ALIGN_H, cfg.ALIGN_W, cfg.EMB_DIM))
#        
#    def get_output_shape(self):
#        return self.call(self.get_input_shape()).shape
##
##def build_fpn_mask_graph(rois, feature_maps, image_meta,
##                         pool_size, num_classes, train_bn=True):
#def fpn_mask_graph(rois, pool_size=7, num_classes=1, train_bn=True):
#    """Builds the computation graph of the mask head of Feature Pyramid Network.
#    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
#          coordinates.
#    feature_maps: List of feature maps from different layers of the pyramid,
#                  [P2, P3, P4, P5]. Each has a different resolution.
#    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
#    pool_size: The width of the square feature map generated from ROI Pooling.
#    num_classes: number of classes, which determines the depth of the results
#    train_bn: Boolean. Train or freeze Batch Norm layers
#    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
#    """
#    # ROI Pooling
#    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
##    x = PyramidROIAlign([pool_size, pool_size],
##                        name="roi_align_mask")([rois, image_meta] + feature_maps)
#
#    # Conv layers
#    x=rois
#    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding="same"),
#                           name="mrcnn_mask_conv1")(x)
#    x = tf.keras.layers.TimeDistributed(BatchNorm(),
#                           name='mrcnn_mask_bn1')(x, training=train_bn)
#    x = tf.keras.layers.Activation('relu')(x)
#
#    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding="same"),
#                           name="mrcnn_mask_conv2")(x)
#    x = tf.keras.layers.TimeDistributed(BatchNorm(),
#                           name='mrcnn_mask_bn2')(x, training=train_bn)
#    x = tf.keras.layers.Activation('relu')(x)
#
#    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding="same"),
#                           name="mrcnn_mask_conv3")(x)
#    x = tf.keras.layers.TimeDistributed(BatchNorm(),
#                           name='mrcnn_mask_bn3')(x, training=train_bn)
#    x = tf.keras.layers.Activation('relu')(x)
#
#    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.EMB_DIM, (3, 3), padding="same"),
#                           name="mrcnn_mask_conv4")(x)
#    x = tf.keras.layers.TimeDistributed(BatchNorm(),
#                           name='mrcnn_mask_bn4')(x, training=train_bn)
#    x = tf.keras.layers.Activation('relu')(x)
#
#    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(cfg.EMB_DIM, (2, 2), strides=2, activation="relu"),
#                           name="mrcnn_mask_deconv")(x)
#    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
#                           name="mrcnn_mask")(x)
#    return x

