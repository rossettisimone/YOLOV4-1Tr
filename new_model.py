import tensorflow as tf
from backbone import cspdarknet53_graph, load_weights_cspdarknet53, freeze_weights_cspdarknet53
from layers import yolov4_plus1_graph, yolov4_plus1_decode_graph, yolov4_plus1_proposal_graph,\
     fpn_classifier_graph_AFP, build_fpn_mask_graph_AFP
import config as cfg
from new_utils import train_step, val_step, freeze_batch_norm
from utils import data_labels

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.model = model_graph()
    
    def compile(self, optimizer):
        super(Model, self).compile()
        self.optimizer = optimizer
        
    def train_step(self, data):
        image, gt_mask, gt_masks, gt_bboxes = data
        label_2, label_3, label_4, label_5 = tf.map_fn(data_labels, (gt_bboxes, gt_mask), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
        data = image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes 
        alb_total_loss, rpn_box_loss, rpn_class_loss, mrcnn_class_loss, mrcnn_box_loss, mrcnn_mask_loss = train_step(self.model, data, self.optimizer)
        return {"alb_total_loss": alb_total_loss, "rpn_box_loss": rpn_box_loss, "rpn_class_loss": rpn_class_loss, \
                "mrcnn_class_loss":mrcnn_class_loss, "mrcnn_box_loss":mrcnn_box_loss, "mrcnn_mask_loss": mrcnn_mask_loss}
    
    def test_step(self, data):
        image, gt_mask, gt_masks, gt_bboxes = data
        label_2, label_3, label_4, label_5 = tf.map_fn(data_labels, (gt_bboxes, gt_mask), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
        data = image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes 
        alb_total_loss, rpn_box_loss, rpn_class_loss, mrcnn_class_loss, mrcnn_box_loss, mrcnn_mask_loss = val_step(self.model, data)
        return {"alb_total_loss": alb_total_loss, "rpn_box_loss": rpn_box_loss, "rpn_class_loss": rpn_class_loss, \
                "mrcnn_class_loss":mrcnn_class_loss, "mrcnn_box_loss":mrcnn_box_loss, "mrcnn_mask_loss": mrcnn_mask_loss}
        
    def predict_step(self, image):
        return self.model(image)
    
def model_graph(pretrained_backbone=True): 
     
    input_layer = tf.keras.layers.Input((cfg.TRAIN_SIZE, cfg.TRAIN_SIZE, 3))
    
    backbone = cspdarknet53_graph(input_layer) # may try a smaller backbone? backbone = cspdarknet53_tiny(input_layer)
    
    neck = yolov4_plus1_graph(backbone)
    
    rpn_predictions, rpn_embeddings = yolov4_plus1_decode_graph(neck)
    rpn_proposals = yolov4_plus1_proposal_graph(rpn_predictions)
    
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = fpn_classifier_graph_AFP([rpn_proposals[...,:4],rpn_embeddings])
    mrcnn_mask = build_fpn_mask_graph_AFP([rpn_proposals[...,:4],rpn_embeddings])
    
    model = tf.keras.Model(inputs=input_layer, outputs=[rpn_predictions, rpn_embeddings, \
                                                    rpn_proposals, mrcnn_class_logits, \
                                                    mrcnn_class, mrcnn_bbox, mrcnn_mask])
    model.s_c = tf.Variable(initial_value=0.0, trainable=True)
    model.s_r = tf.Variable(initial_value=0.0, trainable=True)
    model.s_mc = tf.Variable(initial_value=0.0, trainable=True)
    model.s_mr = tf.Variable(initial_value=0.0, trainable=True)
    model.s_mm = tf.Variable(initial_value=0.0, trainable=True)
    
    if pretrained_backbone:
        load_weights_cspdarknet53(model, cfg.CSP_DARKNET53) # load backbone weights and set to non trainable
        
        freeze_weights_cspdarknet53(model)
        
    return model

class FreezeBackbone(tf.keras.callbacks.Callback):
    def __init__(self, model, n_epochs=2):
        super().__init__()
        self.n_epochs = n_epochs
        self.model = model.model

    def on_epoch_start(self, epoch, logs=None):
        if epoch <= self.n_epochs:
            freeze_weights_cspdarknet53(self.model)

class FreezeBatchNorm(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model.model

    def on_epoch_start(self, epoch, logs=None):
        freeze_batch_norm(self.model)