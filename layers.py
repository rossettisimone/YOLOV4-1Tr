#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 18:37:18 2020

@author: Simone Rossetti
"""
import tensorflow as tf
import config as cfg
from utils import decode_delta_map, xywh2xyxy, entry_stop_gradients, check_proposals_tensor, nms_proposals_tensor, decode_prediction
from backbone import cspdarknet53_graph
import numpy as np
from group_norm import GroupNormalization as GroupNorm

############################################################
#  Support layers
############################################################

def Conv2D(x, kernel_size, filters, downsample=False, activate=True, bn=True, activate_type='leaky', name=None):
    """
    Support Conv2D layer
    """
    if downsample:
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size = kernel_size, strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.),name=name)(x)
    if bn: x = tf.keras.layers.BatchNormalization()(x)
    
    def mish(x):
        return x * tf.math.tanh(tf.math.softplus(x))

    if activate:
        if activate_type == "leaky":
            x = tf.nn.leaky_relu(x, alpha=0.1)
        elif activate_type == "relu":
            x = tf.nn.relu(x)
        elif activate_type == "mish":
            x = mish(x)
    return x

############################################################
#  YOLOV4 neck graph
############################################################

def yolov4_plus1_graph(input_layers):
    """
    YOLOv4 implements only 3 FPN levels and 3 bottom up path augmentation levels,
    here a fourth level is added in order to enhance detection of small targets
    Input:
        b_2: [batch, train_size/stride[0], train_size/stride[0], 128]
        b_3: [batch, train_size/stride[1], train_size/stride[1], 256]
        b_4: [batch, train_size/stride[2], train_size/stride[2], 512]
        b_5: [batch, train_size/stride[3], train_size/stride[3], 512]
    Output:
        n_2: [batch, train_size/stride[0], train_size/stride[0], 64]
        n_3: [batch, train_size/stride[1], train_size/stride[1], 128]
        n_4: [batch, train_size/stride[2], train_size/stride[2], 256]
        n_5: [batch, train_size/stride[3], train_size/stride[3], 512]
    """
    
    b_2, b_3, b_4, b_5 = input_layers 
    
    # Top - Down FPN
    p_5 = b_5
    x = Conv2D(p_5, kernel_size = 1, filters = 256)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    y = Conv2D(b_4, kernel_size = 1, filters = 256)
    x = tf.concat([y,x],axis=-1)
    
    x = Conv2D(x, kernel_size = 1, filters = 256)
    x = Conv2D(x, kernel_size = 3, filters = 512)
    x = Conv2D(x, kernel_size = 1, filters = 256)
    x = Conv2D(x, kernel_size = 3, filters = 512)
    x = Conv2D(x, kernel_size = 1, filters = 256)
    
    p_4 = x
    
    x = Conv2D(p_4, kernel_size = 1, filters = 128)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    y = Conv2D(b_3, kernel_size = 1, filters = 128)
    x = tf.concat([y,x],axis=-1)
    
    x = Conv2D(x, kernel_size = 1, filters = 128)
    x = Conv2D(x, kernel_size = 3, filters = 256)
    x = Conv2D(x, kernel_size = 1, filters = 128)
    x = Conv2D(x, kernel_size = 3, filters = 256)
    x = Conv2D(x, kernel_size = 1, filters = 128)
    
    p_3 = x
    
    x = Conv2D(p_3, kernel_size = 1, filters = 64)
    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    y = Conv2D(b_2, kernel_size = 1, filters = 64)
    x = tf.concat([y,x],axis=-1)
    
    x = Conv2D(x, kernel_size = 1, filters = 64)
    x = Conv2D(x, kernel_size = 3, filters = 128)
    x = Conv2D(x, kernel_size = 1, filters = 64)
    x = Conv2D(x, kernel_size = 3, filters = 128)
    x = Conv2D(x, kernel_size = 1, filters = 64)
    
    p_2 = x
    
    # Bottom - Up Augmentation 
    n_2 = p_2
    
    x = Conv2D(n_2, kernel_size = 3, filters = 128, downsample=True)
    x = tf.concat([x, p_3], axis=-1)
    x = Conv2D(x, kernel_size = 1, filters = 128)
    x = Conv2D(x, kernel_size = 3, filters = 256)
    x = Conv2D(x, kernel_size = 1, filters = 128)
    x = Conv2D(x, kernel_size = 3, filters = 256)
    x = Conv2D(x, kernel_size = 1, filters = 128)
    
    n_3 = x
    
    x = Conv2D(n_3, kernel_size = 3, filters = 256, downsample=True)
    x = tf.concat([x, p_4],axis=-1)
    x = Conv2D(x, kernel_size = 1, filters = 256)
    x = Conv2D(x, kernel_size = 3, filters = 512)
    x = Conv2D(x, kernel_size = 1, filters = 256)
    x = Conv2D(x, kernel_size = 3, filters = 512)
    x = Conv2D(x, kernel_size = 1, filters = 256)
    
    n_4 = x
    
    x = Conv2D(n_4, kernel_size = 3, filters = 512, downsample=True)
    x = tf.concat([x, p_5],axis=-1)
    x = Conv2D(x, kernel_size = 1, filters = 512)
    x = Conv2D(x, kernel_size = 3, filters = 1024)
    x = Conv2D(x, kernel_size = 1, filters = 512)
    x = Conv2D(x, kernel_size = 3, filters = 1024)
    x = Conv2D(x, kernel_size = 1, filters = 512)
    
    n_5 = x
    
    return n_2, n_3, n_4, n_5

############################################################
#  YOLOV4+1 predictions and features splitting graph
############################################################

def yolov4_plus1_decode_graph(input_layer):
    """
    This is a support graph which permits to split features and prediction data, 
    Differently from YOLOv4 we produce features tensors which are used in ROI Pooling
    to extract the mask of the targets.
    Input:
        n_2: [batch, train_size/stride[0], train_size/stride[0], 64]
        n_3: [batch, train_size/stride[1], train_size/stride[1], 128]
        n_4: [batch, train_size/stride[2], train_size/stride[2], 256]
        n_5: [batch, train_size/stride[3], train_size/stride[3], 512]
    Output:
        p_2: [batch, num_anchors, train_size/stride[0], train_size/stride[0], bbox_dim+conf_dim]
        p_3: [batch, num_anchors, train_size/stride[1], train_size/stride[1], bbox_dim+conf_dim]
        p_4: [batch, num_anchors, train_size/stride[2], train_size/stride[2], bbox_dim+conf_dim]
        p_5: [batch, num_anchors, train_size/stride[3], train_size/stride[3], bbox_dim+conf_dim]
        
        e_2: [batch, train_size/stride[0], train_size/stride[0], num_classes]
        e_3: [batch, train_size/stride[1], train_size/stride[1], num_classes]
        e_4: [batch, train_size/stride[2], train_size/stride[2], num_classes]
        e_5: [batch, train_size/stride[3], train_size/stride[3], num_classes]
        
        f_2: [batch, train_size/stride[0], train_size/stride[0], embedding_dim]
        f_3: [batch, train_size/stride[1], train_size/stride[1], embedding_dim]
        f_4: [batch, train_size/stride[2], train_size/stride[2], embedding_dim]
        f_5: [batch, train_size/stride[3], train_size/stride[3], embedding_dim]
        
    """
    n_2, n_3, n_4, n_5 = input_layer

    prediction_channels = cfg.BBOX_REG + cfg.BBOX_CONF #+ cfg.NUM_CLASSES
    prediction_filters = cfg.NUM_ANCHORS * prediction_channels
    classification_channels = cfg.NUM_ANCHORS * cfg.NUM_CLASSES
    
    # level 2
    f_2 = Conv2D(n_2, kernel_size = 3, filters = cfg.EMB_DIM, activate=False, bn=False)
    e_2 = tf.keras.layers.Dense(classification_channels)(n_2)
    x = Conv2D(n_2, kernel_size = 3, filters = 64)
    x = Conv2D(x, kernel_size = 1, filters = prediction_filters, activate=False, bn=False)#24
    x = tf.concat([x,e_2],axis=-1)
    p_2 = tf.transpose(tf.reshape(x, [tf.shape(x)[0], cfg.TRAIN_SIZE//cfg.STRIDES[0], \
                                      cfg.TRAIN_SIZE//cfg.STRIDES[0], cfg.NUM_ANCHORS, \
                                      prediction_channels+cfg.NUM_CLASSES]), perm = [0, 3, 1, 2, 4])
    # level 3
    f_3 = Conv2D(n_3, kernel_size = 3, filters = cfg.EMB_DIM, activate=False, bn=False)
    e_3 = tf.keras.layers.Dense(classification_channels)(n_3)
    x = Conv2D(n_3, kernel_size = 3, filters = 128)
    x = Conv2D(x, kernel_size = 1, filters = prediction_filters, activate=False, bn=False)#24
    x = tf.concat([x,e_3],axis=-1)
    p_3 = tf.transpose(tf.reshape(x, [tf.shape(x)[0], cfg.TRAIN_SIZE//cfg.STRIDES[1], \
                                      cfg.TRAIN_SIZE//cfg.STRIDES[1], cfg.NUM_ANCHORS, \
                                      prediction_channels+cfg.NUM_CLASSES]), perm = [0, 3, 1, 2, 4])
    # level 4
    f_4 = Conv2D(n_4, kernel_size = 3, filters = cfg.EMB_DIM, activate=False, bn=False)
    e_4 = tf.keras.layers.Dense(classification_channels)(n_4)
    x = Conv2D(n_4, kernel_size = 3, filters = 256)
    x = Conv2D(x, kernel_size = 1, filters = prediction_filters, activate=False, bn=False)#24
    x = tf.concat([x,e_4],axis=-1)
    p_4 = tf.transpose(tf.reshape(x, [tf.shape(x)[0], cfg.TRAIN_SIZE//cfg.STRIDES[2], \
                                      cfg.TRAIN_SIZE//cfg.STRIDES[2], cfg.NUM_ANCHORS, \
                                      prediction_channels+cfg.NUM_CLASSES]), perm = [0, 3, 1, 2, 4])
    # level 5
    f_5 = Conv2D(n_5, kernel_size = 3, filters = cfg.EMB_DIM, activate=False, bn=False)
    e_5 = tf.keras.layers.Dense(classification_channels)(n_5)  
    x = Conv2D(n_5, kernel_size = 3, filters = 512)
    x = Conv2D(x, kernel_size = 1, filters = prediction_filters, activate=False, bn=False)#24
    x = tf.concat([x,e_5],axis=-1)
    p_5 = tf.transpose(tf.reshape(x, [tf.shape(x)[0], cfg.TRAIN_SIZE//cfg.STRIDES[3], \
                                      cfg.TRAIN_SIZE//cfg.STRIDES[3], cfg.NUM_ANCHORS, \
                                      prediction_channels+cfg.NUM_CLASSES]), perm = [0, 3, 1, 2, 4])
    
    return [p_2,p_3,p_4,p_5], [f_2,f_3,f_4,f_5]

############################################################
#  YOLOV4+1 proposals decode graph
############################################################

def yolov4_plus1_proposal_graph(predictions):
    """
    YOLOv4 decoding is applied and shift wrt to anchors is parsed:
        ANCHORS: 4 anchors for 4 levels, each anchors is made by width W and height H
        STRIDES: are the strides resulting from each pyramid level, you can get dimension
                of the features per each level by applying TRAIN_SIZE/STRIDES
    In decode_prediction the delta mapping is inverted and for each layer 
    proposals are checked for consistency and ordered by prediction class confidence,
    each proposal vector is zero padded and after concatenation NMS is performed
    (overlapping proposals are removed by confidance score).
    Backpropagation is stopped for cutted and zeroed proposals
    Input:
        p_2: [batch, num_anchors, train_size/stride[0], train_size/stride[0], bbox_dim+conf_dim]
        p_3: [batch, num_anchors, train_size/stride[1], train_size/stride[1], bbox_dim+conf_dim]
        p_4: [batch, num_anchors, train_size/stride[2], train_size/stride[2], bbox_dim+conf_dim]
        p_5: [batch, num_anchors, train_size/stride[3], train_size/stride[3], bbox_dim+conf_dim]
    Output (foreground detection):
        proposals: [batch, max_proposals, bbox+conf] 
        
    """
    ANCHORS = tf.reshape(tf.constant(cfg.ANCHORS,dtype=tf.float32),[cfg.LEVELS, cfg.NUM_ANCHORS, 2])
    p_2, p_3, p_4, p_5 = predictions
    d_2 = decode_prediction(p_2, ANCHORS[0], cfg.STRIDES[0])
    d_3 = decode_prediction(p_3, ANCHORS[1], cfg.STRIDES[1])    
    d_4 = decode_prediction(p_4, ANCHORS[2], cfg.STRIDES[2])    
    d_5 = decode_prediction(p_5, ANCHORS[3], cfg.STRIDES[3])
    proposals = tf.concat([d_2,d_3,d_4,d_5],axis=1) #concat along levels
    proposals = nms_proposals_tensor(proposals)
    
    # stop backpropagation for all zero boxes, this leads to nan gradient due to log decoding
    mask_non_zero_entry = tf.cast(tf.not_equal(tf.reduce_sum(proposals[...,:4],axis=-1),0.0)[...,tf.newaxis],tf.float32)
    proposals = entry_stop_gradients(proposals, mask_non_zero_entry)
    
    return proposals

############################################################
#  PANet modified MASK-RCNN mask graph
############################################################

def mask_graph_AFP(inputs, pool_size =cfg.MASK_POOL_SIZE , mask_conf=cfg.MASK_CONF): # num classes + 1, 0 is background
    """Builds the computation graph of the mask head of Feature Pyramid Network.
    Params: 
        pool_size: The width of the square feature map generated from ROI Pooling.
        mask_conf: number of conf channels, which determines the depth of the results
    Input:
        rois: [batch, num_rois, (x1, y1, x2, y2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                      [e_2, e_3, e_4, e_5]. Each has a different resolution.
        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    Output: 
        Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, MASK_CONF]
    """
    # rois, feature_maps = inputs[0], inputs[1]
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x2, x3, x4, x5 = inputs
    x2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.MASK_LAYERS_SIZE, (3, 3), padding='same'), name='roi_mask_afp2')(x2)
    x2 = tf.keras.layers.TimeDistributed(GroupNorm(), name='roi_mask_afp2_gn')(x2)
    x2 = tf.keras.layers.Activation('relu', name='roi_mask_afp2_gn_relu')(x2)
    x3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.MASK_LAYERS_SIZE, (3, 3), padding='same'), name='roi_mask_afp3')(x3)
    x3 = tf.keras.layers.TimeDistributed(GroupNorm(), name='roi_mask_afp3_gn')(x3)
    x3 = tf.keras.layers.Activation('relu', name='roi_mask_afp3_gn_relu')(x3)
    x4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.MASK_LAYERS_SIZE, (3, 3), padding='same'), name='roi_mask_afp4')(x4)
    x4 = tf.keras.layers.TimeDistributed(GroupNorm(), name='roi_mask_afp4_gn')(x4)
    x4 = tf.keras.layers.Activation('relu', name='roi_mask_afp4_gn_relu')(x4)
    x5 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.MASK_LAYERS_SIZE, (3, 3), padding='same'), name='roi_mask_afp5')(x5)
    x5 = tf.keras.layers.TimeDistributed(GroupNorm(), name='roi_mask_afp5_gn')(x5)
    x5 = tf.keras.layers.Activation('relu', name='roi_mask_afp5_gn_relu')(x5)

    x = tf.keras.layers.Maximum()([x2, x3, x4, x5])
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.MASK_LAYERS_SIZE, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = tf.keras.layers.TimeDistributed(GroupNorm(), name='mrcnn_mask_gn1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.MASK_LAYERS_SIZE, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = tf.keras.layers.TimeDistributed(GroupNorm(), name='mrcnn_mask_gn2')(x)
    shared = tf.keras.layers.Activation('relu')(x)

    x_fcn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.MASK_LAYERS_SIZE, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(shared)
    x_fcn = tf.keras.layers.TimeDistributed(GroupNorm(), name='mrcnn_mask_gn3')(x_fcn)
    x_fcn = tf.keras.layers.Activation('relu')(x_fcn)
    x_fcn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(cfg.MASK_LAYERS_SIZE, (2, 2), strides=(2, 2), activation="relu"),
                           name="mrcnn_mask_deconv")(x_fcn)

    x_ff = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.MASK_LAYERS_SIZE, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(shared)
    x_ff = tf.keras.layers.TimeDistributed(GroupNorm(), name='mrcnn_mask_gn4')(x_ff)
    x_ff = tf.keras.layers.Activation('relu')(x_ff)
    x_ff = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(cfg.MASK_LAYERS_SIZE//2, (3, 3), padding="same"),
                              name="mrcnn_mask_conv5")(x_ff)
    x_ff = tf.keras.layers.TimeDistributed(GroupNorm(), name='mrcnn_mask_gn5')(x_ff)
    x_ff = tf.keras.layers.Activation('relu')(x_ff)
    x_ff_shape = x_ff.shape.as_list()
    x_ff = tf.keras.layers.Reshape((x_ff_shape[1], x_ff_shape[2]*x_ff_shape[3]*x_ff_shape[4]))(x_ff)
    x_ff = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(pool_size*2*pool_size*2, activation='relu'), name='mrcnn_mask_fc')(x_ff)

    x_fcn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(mask_conf, (1, 1), strides=1), name="mrcnn_mask_fcn")(x_fcn)
    x_ff = tf.keras.layers.Reshape((x_ff_shape[1], pool_size*2, pool_size*2, 1))(x_ff)
    x_ff = tf.keras.layers.Lambda(lambda x: tf.tile(x, (1, 1, 1, 1, mask_conf)))(x_ff)
    x = tf.concat([x_fcn, x_ff],axis=-1)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(mask_conf, (1, 1), strides=1), name="custom_out")(x)
#    x = tf.keras.layers.Activation('softmax', name='mrcnn_mask')(x)

    return x


############################################################
#  ROIAlign Layer
############################################################

class PyramidROIAlign_AFP(tf.keras.layers.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
    Params:
        pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
    Inputs:
        boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
        feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]
    Output:
        Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
        The width and height are those specific in the pool_shape in the layer
        constructor.
    """
    def __init__(self, pool_shape = (cfg.POOL_SIZE, cfg.POOL_SIZE), **kwargs):
        super(PyramidROIAlign_AFP, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def get_config(self):
        cfg = super().get_config()
        return cfg   
    
    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0] # is x1, y1, x2, y2
        x1, y1, x2, y2 = tf.unstack(boxes[...,:4], axis=-1)
        boxes = tf.concat([y1[...,tf.newaxis],x1[...,tf.newaxis],y2[...,tf.newaxis],x2[...,tf.newaxis]],axis=-1)
        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[1]
        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        for i, level in enumerate(range(2, 6)):
            box_indices = tf.range(tf.shape(boxes)[0])
            box_indices = tf.reshape(box_indices, [-1, 1])    
            box_indices = tf.tile(box_indices, [1, tf.shape(boxes)[1]])  
            box_indices = tf.reshape(box_indices, [-1])             
            level_boxes = tf.reshape(boxes, (-1, 4))
            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)
            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))
        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled[0])[1:]], axis=0)
        pooled = [tf.reshape(p, shape) for p in pooled]

        return pooled

    def get_input_shape(self):
        return tf.zeros((cfg.BATCH, cfg.MAX_PROP, 4)),  [tf.zeros(shape=(cfg.BATCH, cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], cfg.EMB_DIM)) for i in range(cfg.LEVELS)]
    
    def get_output_shape(self):
        return [out.shape for out in self.call(self.get_input_shape())]
