import tensorflow as tf
from backbone import cspdarknet53_graph, load_weights_cspdarknet53, freeze_weights_cspdarknet53
from layers import yolov4_plus1_graph, yolov4_plus1_decode_graph, yolov4_plus1_proposal_graph,\
     fpn_classifier_graph_AFP, build_fpn_mask_graph_AFP
import config as cfg
from new_utils import train_step, val_step, freeze_batch_norm, freeze_backbone, freeze_rpn
from utils import data_labels
import numpy as np

class Model(tf.keras.Model):
    
    def compile(self, optimizer):
        super(Model, self).compile()
        self.optimizer = optimizer
        
    def train_step(self, data):
        image, gt_mask, gt_masks, gt_bboxes = data
        label_2, label_3, label_4, label_5 = tf.map_fn(data_labels, (gt_bboxes, gt_mask), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
        data = image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes 
        alb_total_loss, rpn_box_loss, rpn_class_loss, mrcnn_class_loss, mrcnn_box_loss, mrcnn_mask_loss = train_step(self, data, self.optimizer)
        return {"alb_total_loss": alb_total_loss, "rpn_box_loss": rpn_box_loss, "rpn_class_loss": rpn_class_loss, \
                "mrcnn_class_loss":mrcnn_class_loss, "mrcnn_box_loss":mrcnn_box_loss, "mrcnn_mask_loss": mrcnn_mask_loss}
    
    def test_step(self, data):
        image, gt_mask, gt_masks, gt_bboxes = data
        label_2, label_3, label_4, label_5 = tf.map_fn(data_labels, (gt_bboxes, gt_mask), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
        data = image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes 
        alb_total_loss, rpn_box_loss, rpn_class_loss, mrcnn_class_loss, mrcnn_box_loss, mrcnn_mask_loss = val_step(self, data)
        return {"alb_total_loss": alb_total_loss, "rpn_box_loss": rpn_box_loss, "rpn_class_loss": rpn_class_loss, \
                "mrcnn_class_loss":mrcnn_class_loss, "mrcnn_box_loss":mrcnn_box_loss, "mrcnn_mask_loss": mrcnn_mask_loss}
    
    @tf.function
    def infer(self, inputs):
        return self(inputs, training=False)
    
def get_model(pretrained_backbone=True):
    
    input_layer = tf.keras.layers.Input((cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3))
    
    backbone = cspdarknet53_graph(input_layer) # may try a smaller backbone? backbone = cspdarknet53_tiny(input_layer)
    
    neck = yolov4_plus1_graph(backbone)
    
    rpn_predictions, rpn_embeddings = yolov4_plus1_decode_graph(neck)
    rpn_proposals = yolov4_plus1_proposal_graph(rpn_predictions)
    
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph_AFP([rpn_proposals[...,:4],rpn_embeddings])
    mrcnn_mask = build_fpn_mask_graph_AFP([rpn_proposals[...,:4],rpn_embeddings])
    
    model = Model(inputs=input_layer, outputs=[rpn_predictions, rpn_embeddings, \
                                                rpn_proposals, mrcnn_class_logits, \
                                                mrcnn_class, mrcnn_bbox, mrcnn_mask])
        
    model.s_c = tf.Variable(initial_value=0.0, trainable=True)
    model.s_r = tf.Variable(initial_value=0.0, trainable=True)
    model.s_mc = tf.Variable(initial_value=0.0, trainable=True)
    model.s_mr = tf.Variable(initial_value=0.0, trainable=True)
    model.s_mm = tf.Variable(initial_value=0.0, trainable=True)
    
    if pretrained_backbone:
        
        load_weights_cspdarknet53(model, cfg.CSP_DARKNET53) # load backbone weights and set to non trainable
        freeze_backbone(model)
    
    return model

class FreezeBackbone(tf.keras.callbacks.Callback):
    def __init__(self, n_epochs=2):
        super().__init__()
        self.n_epochs = n_epochs

    def on_epoch_start(self, epoch, logs=None):
        if epoch <= self.n_epochs:
            freeze_backbone(self.model, trainable=False)
        else:
            freeze_backbone(self.model, trainable=True)

class FreezeBatchNorm(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_start(self, epoch, logs=None):
        freeze_batch_norm(self.model)

class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("alb_total_loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class EarlyStoppingRPN(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience1=0,patience2=0):
        super(EarlyStoppingRPN, self).__init__()
        self.patience1 = patience1
        self.patience2 = patience2
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        self.freeze_rpn = False
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("alb_total_loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if not self.freeze_rpn:
                if self.wait >= self.patience1:
                    self.wait = 0
                    self.stopped_epoch = epoch
                    self.freeze_rpn = True
                    self.model.set_weights(self.best_weights) 
                    print("Restoring model weights from the end of the best epoch.")
            else:
                if self.wait >= self.patience2:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print("Restoring model weights from the end of the best epoch.")
                    self.model.set_weights(self.best_weights)


    def on_epoch_start(self, epoch, logs=None):
        if self.freeze_rpn:
            freeze_rpn(self.model,trainable=False)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

