#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 18:37:18 2020

@author: fiorapirri
"""
import tensorflow as tf
import config as cfg
from utils import decode_delta_map, xywh2xyxy, nms_proposals, entry_stop_gradients, check_proposals, check_proposals_tensor, nms_proposals_tensor, decode_prediction
from backbone import cspdarknet53_graph

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def Conv2D(x, kernel_size, filters, downsample=False, activate=True, bn=True, activate_type='leaky', name=None):
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
    if activate:
        if activate_type == "leaky":
            x = tf.nn.leaky_relu(x, alpha=0.1)
        elif activate_type == "mish":
            x = mish(x)
    return x

def yolov4_plus1_graph(input_layers):
    
    b_2, b_3, b_4, b_5 = input_layers 
    
    # Top - Down FPN
    p_5 = b_5
    x = Conv2D(p_5, kernel_size = 1, filters = 256)
    x = tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='bilinear')
    y = Conv2D(b_4, kernel_size = 1, filters = 256)
    x = tf.concat([y,x], axis=-1)
    
    x = Conv2D(x, kernel_size = 1, filters = 256)
    x = Conv2D(x, kernel_size = 3, filters = 512)
    x = Conv2D(x, kernel_size = 1, filters = 256)
    x = Conv2D(x, kernel_size = 3, filters = 512)
    x = Conv2D(x, kernel_size = 1, filters = 256)
    
    p_4 = x
    
    x = Conv2D(p_4, kernel_size = 1, filters = 128)
    x = tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='bilinear')
    y = Conv2D(b_3, kernel_size = 1, filters = 128)
    x = tf.concat([y,x], axis=-1)
    
    x = Conv2D(x, kernel_size = 1, filters = 128)
    x = Conv2D(x, kernel_size = 3, filters = 256)
    x = Conv2D(x, kernel_size = 1, filters = 128)
    x = Conv2D(x, kernel_size = 3, filters = 256)
    x = Conv2D(x, kernel_size = 1, filters = 128)
    
    p_3 = x
    
    x = Conv2D(p_3, kernel_size = 1, filters = 64)
    x = tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method='bilinear')
    y = Conv2D(b_2, kernel_size = 1, filters = 64)
    x = tf.concat([y,x], axis=-1)
    
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
    x = tf.concat([x, p_4], axis=-1)
    x = Conv2D(x, kernel_size = 1, filters = 256)
    x = Conv2D(x, kernel_size = 3, filters = 512)
    x = Conv2D(x, kernel_size = 1, filters = 256)
    x = Conv2D(x, kernel_size = 3, filters = 512)
    x = Conv2D(x, kernel_size = 1, filters = 256)
    
    n_4 = x
    
    x = Conv2D(n_4, kernel_size = 3, filters = 512, downsample=True)
    x = tf.concat([x, p_5], axis=-1)
    x = Conv2D(x, kernel_size = 1, filters = 512)
    x = Conv2D(x, kernel_size = 3, filters = 1024)
    x = Conv2D(x, kernel_size = 1, filters = 512)
    x = Conv2D(x, kernel_size = 3, filters = 1024)
    x = Conv2D(x, kernel_size = 1, filters = 512)
    
    n_5 = x
    
    return n_2, n_3, n_4, n_5
    
def yolov4_plus1_decode_graph(input_layer):
    
    n_2, n_3, n_4, n_5 = input_layer
    
    prediction_channels = cfg.BBOX_REG + cfg.BBOX_CLASS + cfg.NUM_CLASS
    prediction_filters = cfg.NUM_ANCHORS * prediction_channels
    
    e_2 = Conv2D(n_2, kernel_size = 3, filters = cfg.EMB_DIM, activate=False, bn=False)
    x = Conv2D(n_2, kernel_size = 3, filters = 64)
    x = Conv2D(x, kernel_size = 1, filters = prediction_filters, activate=False, bn=False)#24
    p_2 = tf.transpose(tf.reshape(x, [tf.shape(x)[0], cfg.TRAIN_SIZE//cfg.STRIDES[0], \
                                      cfg.TRAIN_SIZE//cfg.STRIDES[0], cfg.NUM_ANCHORS, \
                                      prediction_channels]), perm = [0, 3, 1, 2, 4])
    
    e_3 = Conv2D(n_3, kernel_size = 3, filters = cfg.EMB_DIM, activate=False, bn=False)
    x = Conv2D(n_3, kernel_size = 3, filters = 128)
    x = Conv2D(x, kernel_size = 1, filters = prediction_filters, activate=False, bn=False)#24
    p_3 = tf.transpose(tf.reshape(x, [tf.shape(x)[0], cfg.TRAIN_SIZE//cfg.STRIDES[1], \
                                      cfg.TRAIN_SIZE//cfg.STRIDES[1], cfg.NUM_ANCHORS, \
                                      prediction_channels]), perm = [0, 3, 1, 2, 4])
    
    e_4 = Conv2D(n_4, kernel_size = 3, filters = cfg.EMB_DIM, activate=False, bn=False)
    x = Conv2D(n_4, kernel_size = 3, filters = 256)
    x = Conv2D(x, kernel_size = 1, filters = prediction_filters, activate=False, bn=False)#24
    p_4 = tf.transpose(tf.reshape(x, [tf.shape(x)[0], cfg.TRAIN_SIZE//cfg.STRIDES[2], \
                                      cfg.TRAIN_SIZE//cfg.STRIDES[2], cfg.NUM_ANCHORS, \
                                      prediction_channels]), perm = [0, 3, 1, 2, 4])
    
    e_5 = Conv2D(n_5, kernel_size = 3, filters = cfg.EMB_DIM, activate=False, bn=False)
    x = Conv2D(n_5, kernel_size = 3, filters = 512)
    x = Conv2D(x, kernel_size = 1, filters = prediction_filters, activate=False, bn=False)#24
    p_5 = tf.transpose(tf.reshape(x, [tf.shape(x)[0], cfg.TRAIN_SIZE//cfg.STRIDES[3], \
                                      cfg.TRAIN_SIZE//cfg.STRIDES[3], cfg.NUM_ANCHORS, \
                                      prediction_channels]), perm = [0, 3, 1, 2, 4])
    
    return [p_2,p_3,p_4,p_5], [e_2,e_3,e_4,e_5]

def yolov4_plus1_proposal_graph(predictions):
    
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

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

class CustomDummy(tf.keras.layers.Layer):
    def __init__(self, n=0, name='dummy', **kwargs):
        super(CustomDummy, self).__init__(name=name+'_'+str(n), **kwargs)

    def call(self, input_layer, training):
        return input_layer

class CustomConv2D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filters, downsample=False, activate=True, bn=True, activate_type='leaky',name='custom_conv2d', n = 0, **kwargs):
        super(CustomConv2D, self).__init__(name = name + '_' + str(n), **kwargs)
         
        self.activate = activate
        self.activate_type = activate_type
        if downsample:
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'
        self.zero_pad = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0))) if downsample else CustomDummy(n=n*10+1)
        self.conv_2d = tf.keras.layers.Conv2D(filters=filters, kernel_size = kernel_size, strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))
        self.batch_norm = BatchNormalization() if bn else CustomDummy(n=n*10+1)
        self.nn = n+1
        
    def call(self, input_layer, training=False):
        x = self.zero_pad(input_layer)
        x = self.conv_2d(x)
        x = self.batch_norm(x, training)
        if self.activate == True:
            if self.activate_type == "leaky":
                x = tf.nn.leaky_relu(x, alpha=0.1)
            elif self.activate_type == "mish":
                x = self.mish(x)
        return x
    
    def mish(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))

class CustomShuffleConv2D(tf.keras.layers.Layer):
    def __init__(self,  filters, n=0, name='shuffle', **kwargs):
        super(CustomShuffleConv2D, self).__init__(name=name+'_'+str(n), **kwargs)

        self.conv_1 = CustomConv2D(kernel_size = 1, filters = filters, n=n*10+1)
        self.conv_2 = CustomConv2D(kernel_size = 3, filters = filters*2, n=n*10+2)
        self.conv_3 = CustomConv2D(kernel_size = 1, filters = filters, n=n*10+3)
        self.conv_4 = CustomConv2D(kernel_size = 3, filters = filters*2, n=n*10+4)
        self.conv_5 = CustomConv2D(kernel_size = 1, filters = filters, n=n*10+5)

    def call(self, input_layer, training=False):
        x = self.conv_1(input_layer, training)
        x = self.conv_2(x, training)
        x = self.conv_3(x, training)
        x = self.conv_4(x, training)
        x = self.conv_5(x, training)
        return x
    
class CustomUpsampleAndConcat(tf.keras.layers.Layer):
    def __init__(self, filters, n=0, name='upsample_concat', **kwargs):
        super(CustomUpsampleAndConcat, self).__init__(name=name+'_'+str(n), **kwargs)
        
        self.conv_1 = CustomConv2D(kernel_size = 1, filters = filters, n=n*10+1)
        self.conv_2 = CustomConv2D(kernel_size = 1, filters = filters, n=n*10+2)
        
    def call(self, up_layer, concat_layer, training=False):
        y = self.conv_1(up_layer, training)
        y = self.upsample(y)
        x = self.conv_2(concat_layer, training)
        x = self.concat(x,y)
        return x
    
    def upsample(self, input_layer):
        return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')
    
    def concat(self, input_layer1, input_layer2):
        return tf.concat([input_layer1, input_layer2], axis=-1)

class CustomUpsampleAndConcatAndShuffle(tf.keras.layers.Layer):
    def __init__(self, filters, n=0, name='upsample_concat_shuffle', **kwargs):
        super(CustomUpsampleAndConcatAndShuffle, self).__init__(name=name+'_'+str(n), **kwargs)
        
        self.up_concat = CustomUpsampleAndConcat(filters, n=n*10+1)
        self.shuffle = CustomShuffleConv2D(filters, n=n*10+1)
        
    def call(self, up_layer, concat_layer, training=False):
        x = self.up_concat(up_layer, concat_layer, training)
        x = self.shuffle(x, training)
        return x
    
class CustomDownsampleAndConcatAndShuffle(tf.keras.layers.Layer):
    def __init__(self, filters, n=0, name='downsample_concat_shuffle', **kwargs):
        super(CustomDownsampleAndConcatAndShuffle, self).__init__(name=name+'_'+str(n), **kwargs)
        
        self.conv_down = CustomConv2D(kernel_size = 3, filters = filters, downsample=True, n=n*10+1)
        self.shuffle = CustomShuffleConv2D(filters, n=n*10+1)
        
    def call(self, down_layer, concat_layer, training=False):
        y = self.conv_down(down_layer, training)
        y = self.concat(y, concat_layer)
        x = self.shuffle(y, training)
        return x

    def concat(self, input_layer1, input_layer2):
        return tf.concat([input_layer1, input_layer2], axis=-1)

class CustomDecode(tf.keras.layers.Layer):
    def __init__(self, level, n=0, name='decode', **kwargs):
        super(CustomDecode, self).__init__(name=name+'_'+str(n), **kwargs)
        
        self.NUM_ANCHORS = cfg.NUM_ANCHORS
        self.NUM_CLASS = cfg.NUM_CLASS
        self.BBOX_CLASS = cfg.BBOX_CLASS
        self.BBOX_REG = cfg.BBOX_REG
        self.MASK = cfg.MASK
        self.LEVEL = level
        self.STRIDES = tf.constant(cfg.STRIDES,dtype=tf.float32)
        self.LEVELS = cfg.LEVELS
        self.STRIDE = self.STRIDES[level]
#        self.XYSCALE = tf.constant(cfg.XYSCALE,dtype=tf.float32)
        self.filters = 2**(6+level)
        self.bbox_filters = self.NUM_ANCHORS * (self.NUM_CLASS + self.BBOX_CLASS + self.BBOX_REG)
        self.ANCHORS = tf.reshape(tf.constant(cfg.ANCHORS,dtype=tf.float32),[self.LEVELS, self.NUM_ANCHORS, 2])[level]
        self.emb_dim = cfg.EMB_DIM 
        
        self.conv_1 = CustomConv2D(kernel_size = 3, filters = self.emb_dim, activate=False, bn=False, n=1) #512
        self.conv_2 = CustomConv2D(kernel_size = 3, filters = self.filters, n=2) #64/128/..
        self.conv_3 = CustomConv2D(kernel_size = 1, filters = self.bbox_filters, activate=False, bn=False, n=3)#24
        
    def call(self, input_layer, training = False): # align = False
        emb = self.conv_1(input_layer, training)
        pred = self.conv_2(input_layer, training)
        pred = self.conv_3(pred, training)
        pred = self.decode(pred) #align
        return pred, emb
    
    def decode(self, pred):#align
#        if training and not inferring: # b x 104 x 104 x 24 -> b x 104 x 104 x 4 x 6
        return tf.transpose(tf.reshape(pred, [tf.shape(pred)[0], cfg.TRAIN_SIZE//cfg.STRIDES[self.LEVEL], cfg.TRAIN_SIZE//cfg.STRIDES[self.LEVEL], cfg.NUM_ANCHORS, cfg.NUM_CLASS + 5]), perm = [0, 3, 1, 2, 4])#, pemb  # prediction        

class CustomProposalLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomProposalLayer, self).__init__(**kwargs)
        self.STRIDES = tf.constant(cfg.STRIDES,dtype=tf.float32)
        self.ANCHORS = tf.reshape(tf.constant(cfg.ANCHORS,dtype=tf.float32),[cfg.LEVELS, cfg.NUM_ANCHORS, 2])
        self.TRAIN_SIZE = tf.constant(cfg.TRAIN_SIZE,dtype=tf.float32)

    def call(self, predictions, embeddings = None):
        """ predictions: b x 4 x h x w x 6; embeddings: b x h x w x emb (=208) """
        """ proposals: b x MAX_PROP x (4+1+emb) where bb is in xyxy form"""
        
        return  proposal_graph(predictions, embeddings)
    
    def get_input_shape(self):
#        return [tf.random.uniform((cfg.BATCH, cfg.NUM_ANCHORS, cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], cfg.NUM_CLASS + 5)) for i in range(cfg.LEVELS)], [tf.random.uniform((cfg.BATCH, cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], cfg.EMB_DIM)) for i in range(cfg.LEVELS)]
        return [tf.random.uniform((cfg.BATCH, cfg.NUM_ANCHORS, cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], cfg.NUM_CLASS + 5)) for i in range(cfg.LEVELS)], None
    
    def get_output_shape(self):
        pred, emb = self.get_input_shape()
        return self.call(pred, emb).shape

def proposal_graph(predictions, embeddings = None):
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
#  ROIAlign Layer
############################################################

#def log2_graph(x):
#    """Implementation of Log2. TF doesn't have a native implementation."""
#    return tf.math.log(x) / tf.math.log(2.0)

class PyramidROIAlign_AFP(tf.keras.layers.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
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
#        boxes = inputs[0]
        boxes, feature_maps = inputs[0], inputs[1]
        # Image meta
        # Holds details about the image. See compose_image_meta()
#        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
#        feature_maps = inputs[2:]

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
#        box_to_level = []
        for i, level in enumerate(range(2, 6)):

            box_indices = tf.range(tf.shape(boxes)[0])
            
            box_indices = tf.reshape(box_indices, [-1, 1])    
            box_indices = tf.tile(box_indices, [1, tf.shape(boxes)[1]])  
            box_indices = tf.reshape(box_indices, [-1]) 
#            box_indices = tf.keras.backend.repeat_elements(box_indices, tf.keras.backend.int_shape(boxes)[1], axis=0)
            
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
        # Pack pooled features into one tensor
        # pooled = tf.concat(pooled, axis=0)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled[0])[1:]], axis=0)
        # pooled = tf.reshape(pooled, shape)
        pooled = [tf.reshape(p, shape) for p in pooled]

        return pooled

    def get_input_shape(self):
        return tf.zeros((cfg.BATCH, cfg.MAX_PROP, 4)),  [tf.zeros(shape=(cfg.BATCH, cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], cfg.EMB_DIM)) for i in range(cfg.LEVELS)]
    
    def get_output_shape(self):
        return [out.shape for out in self.call(self.get_input_shape())]

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

def fpn_classifier_graph_AFP(inputs, pool_size=cfg.POOL_SIZE, num_classes=2, fc_layers_size=cfg.FC_LAYER_SIZE):
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
    shared = tf.keras.layers.Lambda(lambda x: tf.squeeze(tf.squeeze(x, 3), 2), name="pool_squeeze")(x)
    # Classifier head
    mrcnn_class_logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    # kernel_regularizer=tf.keras.regularizers.L1(0.01),
    # activity_regularizer=tf.keras.regularizers.L2(0.01)
    mrcnn_probs = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dx, dy, log(dw), log(dh))]
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # kernel_regularizer=tf.keras.regularizers.L1(0.01),
    # activity_regularizer=tf.keras.regularizers.L2(0.01)
    mrcnn_bbox = tf.keras.layers.Reshape((x.shape.as_list()[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

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
    x_ff_shape = x_ff.shape.as_list()
    x_ff = tf.keras.layers.Reshape((x_ff_shape[1], x_ff_shape[2]*x_ff_shape[3]*x_ff_shape[4]))(x_ff)
    x_ff = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(pool_size*pool_size*2*2, activation='relu'), name='mrcnn_mask_fc')(x_ff)

    x_fcn = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(num_classes, (1, 1), strides=1), name="mrcnn_mask_fcn")(x_fcn)
    x_ff = tf.keras.layers.Reshape((x_ff_shape[1], pool_size*2, pool_size*2, 1))(x_ff)
    x_ff = tf.keras.layers.Lambda(lambda x: tf.tile(x, (1, 1, 1, 1, num_classes)))(x_ff)
    x = tf.keras.layers.Add()([x_fcn, x_ff])
    x = tf.keras.layers.Activation('sigmoid', name='mrcnn_mask')(x)

    return x