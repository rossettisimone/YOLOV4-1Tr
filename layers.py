#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 18:37:18 2020

@author: fiorapirri
"""


import tensorflow as tf
import config as cfg
from utils import decode_delta_map, xywh2xyxy


class BatchNormalization(tf.keras.layers.BatchNormalization):
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
        self.bbox_filters = self.NUM_ANCHORS * (self.NUM_CLASS + self.BBOX_CLASS + self.BBOX_REG + self.MASK)
        self.ANCHORS = tf.reshape(tf.constant(cfg.ANCHORS,dtype=tf.float32),[self.LEVELS, self.NUM_ANCHORS, 2])[level]
        self.emb_dim = cfg.EMB_DIM 
        
        self.conv_1 = CustomConv2D(kernel_size = 3, filters = self.emb_dim, activate=False, bn=False, n=1) #512
        self.conv_2 = CustomConv2D(kernel_size = 3, filters = self.filters, n=2) #64/128/..
        self.conv_3 = CustomConv2D(kernel_size = 1, filters = self.bbox_filters, activate=False, bn=False, n=3)#24
#        self.align = ROIAlignLayer(cfg.ALIGN_H, cfg.ALIGN_W)
        
    def call(self, input_layer, training = False, inferring = False): # align = False
        pemb = self.conv_1(input_layer, training)
        pred = self.conv_2(input_layer, training)
        pred = self.conv_3(pred, training)
        x = self.decode(pred, pemb, training, inferring) #align
        return x
    
    def decode(self, pred, pemb, training, inferring):#align
        if training and not inferring: # b x 104 x 104 x 4 x 6 - b x 104 x 104 x 64
            return tf.transpose(tf.reshape(pred, [tf.shape(pred)[0], tf.shape(pred)[1], tf.shape(pred)[2], cfg.NUM_ANCHORS, cfg.NUM_CLASS + 5]), perm = [0, 3, 1, 2, 4]), pemb  # prediction        
        
        if training and inferring:
            pred = tf.transpose(tf.reshape(pred, [tf.shape(pred)[0], tf.shape(pred)[1], tf.shape(pred)[2], cfg.NUM_ANCHORS, cfg.NUM_CLASS + 5]), perm = [0, 3, 1, 2, 4])
            pbox = pred[..., :4]
            pconf = pred[..., 4:6]  # Conf
            
            pconf = tf.nn.softmax(pconf, axis=-1)[...,1][...,tf.newaxis]
            pemb = tf.math.l2_normalize(tf.tile(pemb[:,tf.newaxis],[1,cfg.NUM_ANCHORS,1,1,1]),axis=-1, epsilon=1e-12)
#            pcls = tf.zeros((tf.shape(pred)[0], tf.shape(pred)[1], tf.shape(pred)[2], tf.shape(pred)[3],1)) # useless
            pbox = decode_delta_map(pbox, self.ANCHORS/self.STRIDE)
            pbox *= self.STRIDE
            preds = tf.concat([pbox, pconf, pemb], axis=-1)
            preds = tf.reshape(preds, [preds.shape[0], -1, preds.shape[-1]]) # b x nBB x (4 + 1 + 1 + 208) rois
#            pred = tf.concat(preds, axis=1)
            proposals = []
            for ii in range(preds.shape[0]): # batch images
                pred = preds[ii]
                pred = pred[pred[..., 4] > cfg.CONF_THRESH]
                pred = tf.concat([xywh2xyxy(pred),pred[...,4:]],axis=-1) # to bbox
                # pred now has lesser number of proposals. Proposals rejected on basis of object confidence score
                if len(pred) > 0:    
                    boxes = pred[...,:4]
                    scores = pred[...,4]
                    selected_indices = tf.image.non_max_suppression(
                                        boxes, scores, max_output_size=20, iou_threshold=cfg.NMS_THRESH,
                                        score_threshold=cfg.CONF_THRESH
                                    )
                    pred_emb = tf.gather(pred, selected_indices) #b x n rois x (4+1+1+208)
                    proposals.append(pred_emb)
            return proposals
    
#        if align:
#            pred = tf.transpose(tf.reshape(pred, [tf.shape(pred)[0], tf.shape(pred)[1], tf.shape(pred)[2], cfg.NUM_ANCHORS, cfg.NUM_CLASS + 5]), perm = [0, 3, 1, 2, 4])
#            pbox = pred[..., :4]
#            pconf = pred[..., 4:6]  # Conf
#            pconf = tf.keras.activations.softmax(pconf, axis=-1)[...,1][...,tf.newaxis]
#            pemb = tf.math.l2_normalize(tf.tile(pemb[:,tf.newaxis],[1,cfg.NUM_ANCHORS,1,1,1]),axis=-1, epsilon=1e-12)
#            pcls = tf.zeros((tf.shape(pred)[0], tf.shape(pred)[1], tf.shape(pred)[2], tf.shape(pred)[3],1)) # useless
#            pbox = decode_delta_map(pbox, self.ANCHORS/self.STRIDE)
#            pred = tf.concat([pbox, pconf, pcls, pemb], axis=-1)
#            pred = pred[pred[..., 4] > cfg.CONF_THRESH]
#            feature_map = pred[...,5:]
#            rois = pred[...,:4]
#            rois = tf.reshape(rois, (rois.shape[0],-1,rois.shape[-1]))
#            return self.align(feature_map, rois), rois
        
class ROIAlignLayer(tf.keras.layers.Layer):
    """ Implements Region Of Interest Max Pooling 
        for channel-first images and relative bounding box coordinates
        
        # Constructor parameters
            align_height, align_width (int) -- 
              specify height and width of layer outputs
        
        Shape of inputs
            [(batch_size, align_height, align_width, n_channels),
             (batch_size, num_rois, 4)]
           
        Shape of output
            (batch_size * num_rois, align_height, align_width, n_channels)
    
    """
    def __init__(self, align_height, align_width, **kwargs):
        super(ROIAlignLayer, self).__init__(**kwargs)
        self.align_height = align_height
        self.align_width = align_width
        
    def call(self, feature_map, rois):
        """ Maps the input tensor of the ROI layer to its output
        
            # Parameters
                feature_map -- Convolutional feature map tensor,
                        shape (batch_size, align_height, align_width, n_channels)
                rois -- Tensor of region of interests from candidate bounding boxes,
                        shape (batch_size, num_rois, 4)
                        Each region of interest is defined by four relative 
                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1
            # Output
                aligned features -- Tensor with the pooled region of interest, shape
                    (batch_size * num_rois, align_height, align_width, n_channels)
        """
        batch_size, num_rois, _ = rois.shape
        box_indices = tf.repeat(tf.range(0,batch_size,1, tf.int32), num_rois)
        rois = tf.reshape(rois, (-1,4))
        crop_size = (self.align_height, self.align_width)
        return tf.image.crop_and_resize(feature_map, rois, box_indices=box_indices,
                                        crop_size=crop_size, method='bilinear')
