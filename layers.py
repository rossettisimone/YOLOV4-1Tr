#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 18:37:18 2020

@author: fiorapirri
"""
import tensorflow as tf
import config as cfg
from utils import decode_delta_map, xywh2xyxy, nms_proposals, entry_stop_gradients, conf_proposals


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
        proposals = []
        for i in range(cfg.LEVELS):
            pred = predictions[i]
            pconf = pred[..., 4:6]  # Conf
            pconf = tf.nn.softmax(pconf, axis=-1)[...,1][...,tf.newaxis] # 1 is foreground
            pbox = pred[..., :4]
            pbox = decode_delta_map(pbox, self.ANCHORS[i]/self.STRIDES[i])
            pbox *= self.STRIDES[i] # now in range [0, .... cfg.TRAIN_SIZE]
            pbox /= self.TRAIN_SIZE #now normalized in [0...1]
            pbox = xywh2xyxy(pbox) # to bbox
            pbox = tf.clip_by_value(pbox,0.0,1.0) # clip to avoid nan
            preds = tf.concat([pbox, pconf], axis=-1)
            
            if embeddings:
                pemb = embeddings[i]
                pemb = tf.math.l2_normalize(tf.tile(pemb[:,tf.newaxis],(1,cfg.NUM_ANCHORS,1,1,1)), axis=-1, epsilon=1e-12)
                preds = tf.concat([preds, pemb], axis=-1)     
                
            proposal = tf.reshape(preds, [tf.shape(preds)[0], -1, tf.shape(preds)[-1]]) # b x nBB x (4 + 1 + 1 + 208) rois
            proposal = tf.map_fn(conf_proposals, proposal, fn_output_signature=tf.float32)
            proposals.append(proposal)
            
        proposals = tf.concat(proposals,axis=1) #concat along levels
        proposals = tf.map_fn(nms_proposals, proposals, fn_output_signature=tf.float32)
        
        mask_non_zero_entry = tf.cast(tf.not_equal(tf.reduce_sum(proposals[...,:4],axis=-1),0.0)[...,tf.newaxis],tf.float32)
        proposals = entry_stop_gradients(proposals, mask_non_zero_entry)
        return proposals
    
    def get_input_shape(self):
#        return [tf.random.uniform((cfg.BATCH, cfg.NUM_ANCHORS, cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], cfg.NUM_CLASS + 5)) for i in range(cfg.LEVELS)], [tf.random.uniform((cfg.BATCH, cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], cfg.EMB_DIM)) for i in range(cfg.LEVELS)]
        return [tf.random.uniform((cfg.BATCH, cfg.NUM_ANCHORS, cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], cfg.NUM_CLASS + 5)) for i in range(cfg.LEVELS)], None
    
    def get_output_shape(self):
        pred, emb = self.get_input_shape()
        return self.call(pred, emb).shape

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

#class PyramidROIAlign(tf.keras.layers.Layer):
#    """Implements ROI Pooling on multiple levels of the feature pyramid.
#    Params:
#    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
#    Inputs:
#    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
#             coordinates. Possibly padded with zeros if not enough
#             boxes to fill the array.
#    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
#    - feature_maps: List of feature maps from different levels of the pyramid.
#                    Each is [batch, height, width, channels]
#    Output:
#    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
#    The width and height are those specific in the pool_shape in the layer
#    constructor.
#    """
#
#    def __init__(self, pool_shape, **kwargs):
#        super(PyramidROIAlign, self).__init__(**kwargs)
#        self.pool_shape = tuple(pool_shape)
#
#    def call(self, boxes, feature_maps):
#        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
##        boxes = inputs[0]
#
#        # Image meta
#        # Holds details about the image. See compose_image_meta()
#        image_shape = (cfg.TRAIN_SIZE,cfg.TRAIN_SIZE)
#
#        # Feature Maps. List of feature maps from different level of the
#        # feature pyramid. Each is [batch, height, width, channels]
##        feature_maps = inputs[2:]
#
#        # Assign each ROI to a level in the pyramid based on the ROI area.
#        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
#        h = y2 - y1
#        w = x2 - x1
#        # Use shape of first image. Images in a batch must have the same size.
##        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
#        # Equation 1 in the Feature Pyramid Networks paper. Account for
#        # the fact that our coordinatimage_metaes are normalized here.
#        # e.g. a 224x224 ROI (in pixels) maps to P4
#        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
#        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
#        roi_level = tf.minimum(5, tf.maximum(
#            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
#        roi_level = tf.squeeze(roi_level, 2)
#
#        # Loop through levels and apply ROI pooling to each. P2 to P5.
#        pooled = []
#        box_to_level = []
#        for i, level in enumerate(range(2, 6)):
#            ix = tf.where(tf.equal(roi_level, level))
#            level_boxes = tf.gather_nd(boxes, ix)
#
#            # Box indices for crop_and_resize.
#            box_indices = tf.cast(ix[:, 0], tf.int32)
#
#            # Keep track of which box is mapped to which level
#            box_to_level.append(ix)
#
#            # Stop gradient propogation to ROI proposals
#            level_boxes = tf.stop_gradient(level_boxes)
#            box_indices = tf.stop_gradient(box_indices)
#
#            # Crop and Resize
#            # From Mask R-CNN paper: "We sample four regular locations, so
#            # that we can evaluate either max or average pooling. In fact,
#            # interpolating only a single value at each bin center (without
#            # pooling) is nearly as effective."
#            #
#            # Here we use the simplified approach of a single value per bin,
#            # which is how it's done in tf.crop_and_resize()
#            # Result: [batch * num_boxes, pool_height, pool_width, channels]
#            pooled.append(tf.image.crop_and_resize(
#                feature_maps[i], level_boxes, box_indices, self.pool_shape,
#                method="bilinear"))
#
#        # Pack pooled features into one tensor
#        pooled = tf.concat(pooled, axis=0)
#
#        # Pack box_to_level mapping into one array and add another
#        # column representing the order of pooled boxes
#        box_to_level = tf.concat(box_to_level, axis=0)
#        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
#        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
#                                 axis=1)
#
#        # Rearrange pooled features to match the order of the original boxes
#        # Sort box_to_level by batch then box index
#        # TF doesn't have a way to sort by two columns, so merge them and sort.
#        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
#        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
#            box_to_level)[0]).indices[::-1]
#        ix = tf.gather(box_to_level[:, 2], ix)
#        pooled = tf.gather(pooled, ix)
#
#        # Re-add the batch dimension
#        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
#        pooled = tf.reshape(pooled, shape)
#        return pooled
#
#    def get_input_boxes_shape(self):
#        return tf.zeros((cfg.BATCH, cfg.MAX_PROP, 4))
#    
#    def get_input_features_shape(self):
#        return [tf.zeros(shape=(cfg.BATCH, cfg.TRAIN_SIZE//cfg.STRIDES[i],cfg.TRAIN_SIZE//cfg.STRIDES[i], 
#                                        cfg.EMB_DIM)) for i in range(cfg.LEVELS)]
#    def get_output_shape(self):
#        return self.call(self.get_input_boxes_shape(), self.get_input_features_shape()).shape

#        
#class ROIAlignLayer(tf.keras.layers.Layer):
#    """ Implements Region Of Interest Max Pooling 
#        for channel-first images and relative bounding box coordinates
#        
#        # Constructor parameters
#            align_height, align_width (int) -- 
#              specify height and width of layer outputs
#        
#        Shape of inputs
#            [(batch_size, align_height, align_width, n_channels),
#             (batch_size, num_rois, 4)]
#           
#        Shape of output
#            (batch_size * num_rois, align_height, align_width, n_channels)
#    
#    """
#    def __init__(self, align_height, align_width, **kwargs):
#        super(ROIAlignLayer, self).__init__(**kwargs)
#        self.align_height = align_height
#        self.align_width = align_width
#        
#    def call(self, feature_map, rois):
#        """ Maps the input tensor of the ROI layer to its output
#        
#            # Parameters
#                feature_map -- Convolutional feature map tensor,
#                        shape (batch_size, align_height, align_width, n_channels)
#                rois -- Tensor of region of interests from candidate bounding boxes,
#                        shape (batch_size, num_rois, 4)
#                        Each region of interest is defined by four relative 
#                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1
#            # Output
#                aligned features -- Tensor with the pooled region of interest, shape
#                    (batch_size * num_rois, align_height, align_width, n_channels)
#        """
#        batch_size, num_rois, _ = rois.shape
#        box_indices = tf.repeat(tf.range(0,batch_size,1, tf.int32), num_rois)
#        rois = tf.reshape(rois, (-1,4))
#        crop_size = (self.align_height, self.align_width)
#        return tf.image.crop_and_resize(feature_map, rois, box_indices=box_indices,
#                                        crop_size=crop_size, method='bilinear')

