import tensorflow as tf
from backbone import cspdarknet53_graph, load_weights_cspdarknet53, freeze_weights_cspdarknet53
from layers import yolov4_plus1_graph, yolov4_plus1_decode_graph, yolov4_plus1_proposal_graph,\
     fpn_classifier_graph_AFP, build_fpn_mask_graph_AFP
import config as cfg

def build_model(pretrained_backbone=True): 
     
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