#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 19:44:41 2020

@author: fiorapirri
"""


import tensorflow as tf

############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
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

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

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
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


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

align = ROIAlignLayer(7,7)
#
x = align(tf.random.uniform((5,13,13,512),0,255,dtype=tf.int32), tf.random.uniform((5,200,4),0,1,dtype=tf.float32))

class ROIPoolingLayer(tf.keras.layers.Layer):
    """ Implements Region Of Interest Max Pooling 
        for channel-first images and relative bounding box coordinates
        
        # Constructor parameters
            pooled_height, pooled_width (int) -- 
              specify height and width of layer outputs
        
        Shape of inputs
            [(batch_size, pooled_height, pooled_width, n_channels),
             (batch_size, num_rois, 4)]
           
        Shape of output
            (batch_size, num_rois, pooled_height, pooled_width, n_channels)
    
    """
    def __init__(self, pooled_height, pooled_width, **kwargs):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        
        super(ROIPoolingLayer, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shape):
        """ Returns the shape of the ROI Layer output
        """
        feature_map_shape, rois_shape = input_shape
        assert feature_map_shape[0] == rois_shape[0]
        batch_size = feature_map_shape[0]
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooled_height, 
                self.pooled_width, n_channels)

    def call(self, x):
        """ Maps the input tensor of the ROI layer to its output
        
            # Parameters
                x[0] -- Convolutional feature map tensor,
                        shape (batch_size, pooled_height, pooled_width, n_channels)
                x[1] -- Tensor of region of interests from candidate bounding boxes,
                        shape (batch_size, num_rois, 4)
                        Each region of interest is defined by four relative 
                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1
            # Output
                pooled_areas -- Tensor with the pooled region of interest, shape
                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)
        """
        def curried_pool_rois(x): 
          return ROIPoolingLayer._pool_rois(x[0], x[1], 
                                            self.pooled_height, 
                                            self.pooled_width)
        
        pooled_areas = tf.map_fn(curried_pool_rois, x, dtype=tf.float32)

        return pooled_areas
    
    @staticmethod
    def _pool_rois(feature_map, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single image and varios ROIs
        """
        def curried_pool_roi(roi): 
          return ROIPoolingLayer._pool_roi(feature_map, roi, 
                                           pooled_height, pooled_width)
        
        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas
    
    @staticmethod
    def _pool_roi(feature_map, roi, pooled_height, pooled_width):
        """ Applies ROI pooling to a single image and a single region of interest
        """

        # Compute the region of interest        
        feature_map_height = int(feature_map.shape[0])
        feature_map_width  = int(feature_map.shape[1])
        
        h_start = tf.cast(feature_map_height * roi[0], 'int32')
        w_start = tf.cast(feature_map_width  * roi[1], 'int32')
        h_end   = tf.cast(feature_map_height * roi[2], 'int32')
        w_end   = tf.cast(feature_map_width  * roi[3], 'int32')
        
        region = feature_map[h_start:h_end, w_start:w_end, :]
        
        # Divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width  = w_end - w_start
        h_step = tf.cast( region_height / pooled_height, 'int32')
        w_step = tf.cast( region_width  / pooled_width , 'int32')
        
        areas = [[(
                    i*h_step, 
                    j*w_step, 
                    (i+1)*h_step if i+1 < pooled_height else region_height, 
                    (j+1)*w_step if j+1 < pooled_width else region_width
                   ) 
                   for j in range(pooled_width)] 
                  for i in range(pooled_height)]
        
        # take the maximum of each area and stack the result
        def pool_area(x): 
          return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])
        
        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
        return pooled_features
    
    
#
#import numpy as np
#import tensorflow as tf
# 
# 
#def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
#    """
#    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.
#    Args:
#        image: NCHW
#        boxes: nx4, x1y1x2y2
#        box_ind: (n,)
#        crop_size (int):
#    Returns:
#        n,C,size,size
#    """
#    assert isinstance(crop_size, int), crop_size
#    boxes = tf.stop_gradient(boxes)
# 
#    # TF's crop_and_resize produces zeros on border
#    if pad_border:
#        # this can be quite slow
#        image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
#        boxes = boxes + 1
# 
#    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
#        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)
# 
#        spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
#        spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)
# 
#        nx0 = (x0 + spacing_w / 2 - 0.5) / tf.cast(image_shape[1] - 1, tf.float32)
#        ny0 = (y0 + spacing_h / 2 - 0.5) / tf.cast(image_shape[0] - 1, tf.float32)
# 
#        nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / tf.cast(image_shape[1] - 1, tf.float32)
#        nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / tf.cast(image_shape[0] - 1, tf.float32)
# 
#        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)
# 
#    image_shape = tf.shape(image)[2:]
#    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
#    image = tf.transpose(image, [0, 2, 3, 1])   # nhwc
#    ret = tf.image.crop_and_resize(
#        image, boxes, tf.cast(box_ind, tf.int32),
#        crop_size=[crop_size, crop_size])
#    ret = tf.transpose(ret, [0, 3, 1, 2])   # ncss
#    return ret
# 
# 
#def roi_align(featuremap, boxes, resolution):
#    """
#    Args:
#        featuremap: 1xCxHxW
#        boxes: Nx4 floatbox
#        resolution: output spatial resolution
#    Returns:
#        NxCx res x res
#    """
#    # sample 4 locations per roi bin
#    ret = crop_and_resize(
#        featuremap, boxes,
#        tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32),
#        resolution * 2)
#    ret = tf.nn.avg_pool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format='NCHW')
#    return ret
# 
# 
#if __name__ == '__main__':
#    
#    # want to crop 2x2 out of a 5x5 image, and resize to 4x4
#    image = np.arange(25).astype('float32').reshape(5, 5)
#    boxes = np.asarray([[1, 1, 3, 3]], dtype='float32')
#    target = 4
# 
#    print(crop_and_resize(
#        image[None, None, :, :], boxes, [0], target)[0][0])
#    
#    
#def roi_align(feature_maps, boxes, box_indices, output_size, sample_ratio):
#    """Implement ROI align.
#    Args:
#        feature_maps: tensor, shape[batch_size, height, width, channels]
#        boxes: shape [num_boxes, (y1, x1, y2, x2)] y and x normalized by (height -1) and
#            (width -1) respectively.
#        box_indices: indicate which feature_map to crop.
#        output_size: roi_align output size.
#        sample_ratio: sample ratio in each row and col in each bin. If sample 4 regular locations,
#            sample_ratio is 2.
#    """
#
#    def _roi_align_core():
#        """
#        Return:
#            cropped image sampled by bilinear interpolation with shape [num_boxes, crop_height,
#            crop_width, depth], sampled (output_size * sample_ratio)**2 points.
#        """
#        y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
#        bin_height = (y2 - y1) / output_size[0]
#        bin_width = (x2 - x1) / output_size[1]
#
#        grid_center_y1 = (y1 + 0.5 * bin_height / sample_ratio)
#        grid_center_x1 = (x1 + 0.5 * bin_width / sample_ratio)
#
#        grid_center_y2 = (y2 - 0.5 * bin_height / sample_ratio)
#        grid_center_x2 = (x2 - 0.5 * bin_width / sample_ratio)
#
#        # Constructed by top-left sample point and bottom-right
#        # sample point. shape [num_boxes, (y1, x1, y2, x2)]
#        new_boxes = tf.concat([grid_center_y1, grid_center_x1, grid_center_y2, grid_center_x2],
#                              axis=1)
#
#        crop_size = tf.constant([output_size[0] * sample_ratio, output_size[1] * sample_ratio])
#        return tf.image.crop_and_resize(feature_maps, new_boxes, box_indices=box_indices,
#                                        crop_size=crop_size, method='bilinear')
#
#    sampled = _roi_align_core()
#    aligned = tf.nn.avg_pool2d(sampled, sample_ratio, sample_ratio, padding='VALID')
#    return aligned
#
#import tensorflow as tf
#feature_maps = tf.random.uniform((5,104,104,64), 0, 255,tf.int32)
#boxes = tf.random.uniform((200,4),0,104, tf.float32)/104
#box_indices = tf.repeat(tf.range(0,5,1, tf.int32), 40)
#output_size = (3,3)
#sample_ratio = 3
#align = roi_align(feature_maps, boxes, box_indices, output_size, sample_ratio)
#
#
#
#import numpy as np
#
#def roi_align(image, box, height, width):
#  """
#  `image` is a 2-D array, representing the input feature map
#  `box` is a list of four numbers
#  `height` and `width` are the desired spatial size of output feature map
#  """
#  y_min, x_min, y_max, x_max = box
#
#  img_height, img_width = image.shape
#
#  feature_map = []
#
#  for y in np.linspace(y_min, y_max, height) * (img_height - 1):
#    for x in np.linspace(x_min, x_max, width) * (img_width - 1):
#
#      y_l, y_h = np.floor(y).astype('int32'), np.ceil(y).astype('int32')
#      x_l, x_h = np.floor(x).astype('int32'), np.ceil(x).astype('int32')
#
#      a = image[y_l, x_l]
#      b = image[y_l, x_h]
#      c = image[y_h, x_l]
#      d = image[y_h, x_h]
#
#      y_weight = y - y_l
#      x_weight = x - x_l
#
#      val = a * (1 - x_weight) * (1 - y_weight) + \
#            b * x_weight * (1 - y_weight) + \
#            c * y_weight * (1 - x_weight) + \
#            d * x_weight * y_weight
#
#      feature_map.append(val)
#
#  return np.array(feature_map).reshape(height, width)
#
#import numpy as np
#
#def roi_align_vectorized(image, box, height, width):
#  """
#  `image` is a 2-D array, representing the input feature map
#  `box` is a list of four numbers
#  `height` and `width` are the desired spatial size of output feature map
#  """
#  y_min, x_min, y_max, x_max = box
#
#  img_height, img_width = image.shape
#
#  y, x = np.meshgrid(
#      np.linspace(y_min, y_max, height) * (img_height - 1),
#      np.linspace(x_min, x_max, width) * (img_width - 1))
#
#  y = y.transpose().ravel()
#  x = x.transpose().ravel()
#
#  image = image.ravel()
#
#  y_l, y_h = np.floor(y).astype('int32'), np.ceil(y).astype('int32')
#  x_l, x_h = np.floor(x).astype('int32'), np.ceil(x).astype('int32')
#
#  a = image[y_l * img_width + x_l]
#  b = image[y_l * img_width + x_h]
#  c = image[y_h * img_width + x_l]
#  d = image[y_h * img_width + x_h]
#
#  y_weight = y - y_l
#  x_weight = x - x_l
#
#  feature_map = a * (1 - x_weight) * (1 - y_weight) + \
#                b * x_weight * (1 - y_weight) + \
#                c * y_weight * (1 - x_weight) + \
#                d * x_weight * y_weight
#
#  return feature_map.reshape(height, width)
#
#def roi_align_vectorized_backward(image, box, height, width, grad):
#  """
#  `image` is a 2-D array, representing the input feature map
#  `box` is a list of four numbers
#  `height` and `width` are the desired spatial size of output feature map
#  `grad` is a 2-D array of shape [height, width], holding gradient backpropped
#    from downstream layer. 
#  """
#  y_min, x_min, y_max, x_max = box
#  
#  img_height, img_width = image.shape
#
#  y, x = np.meshgrid(
#      np.linspace(y_min, y_max, height) * (img_height - 1),
#      np.linspace(x_min, x_max, width) * (img_width - 1))
#
#  y = y.transpose().ravel()
#  x = x.transpose().ravel()
#
#  image = image.ravel()
#
#  y_l, y_h = np.floor(y).astype('int32'), np.ceil(y).astype('int32')
#  x_l, x_h = np.floor(x).astype('int32'), np.ceil(x).astype('int32')
#
#  y_weight = y - y_l
#  x_weight = x - x_l
#
#  grad = grad.ravel()
#
#  # gradient wrt `a`, `b`, `c`, `d`
#  d_a = (1 - x_weight) * (1 - y_weight) * grad
#  d_b = x_weight * (1 - y_weight) * grad
#  d_c = y_weight * (1 - x_weight) * grad
#  d_d = x_weight * y_weight * grad
#
#  # [4 * height * width]
#  grad = np.concatenate([d_a, d_b, d_c, d_d])
#  # [4 * height * width]
#  indices = np.concatenate([y_l * img_width + x_l,
#                            y_l * img_width + x_h,
#                            y_h * img_width + x_l,
#                            y_h * img_width + x_h])
#
#  # we must route gradients in `grad` to the correct indices of `image` in 
#  # `indices`
#
#  # use numpy's broadcasting rule to generate 2-D array of shape
#  # [4 * height * width, img_height * img_width] 
#  indices = (indices.reshape((-1, 1)) ==
#              np.arange(img_height * img_width).reshape((1, -1)))
#  d_image = np.apply_along_axis(lambda col: grad[col].sum(), 0, indices)
#
#  return d_image.reshape(img_height, img_width)
#
#import tensorflow as tf
#import numpy as np
#
##tf.enable_eager_execution()
#
#with tf.GradientTape() as g:
#  image = tf.convert_to_tensor(
#      np.random.randint(0, 255, size=(1, 3, 8, 1)).astype('float32'))
#  boxes = np.array([[0.32, 0.05, 0.43, 0.54]])
#  g.watch(image)
#  output = tf.image.crop_and_resize(image, boxes, [0], [6, 6])
#
#grad_val = np.random.randint(-10, 10, size=(6, 6)).astype('float32')
#grad_tf = g.gradient(output, image, grad_val.reshape(1, 6, 6, 1))
#
#grad = roi_align_vectorized_backward(
#    image[0, :, :, 0].numpy(), boxes[0], 6, 6, grad_val)
#
## compare if `grad_tf` and `grad` are equal.
#_pool_roi
#roi_align_vectorized(image[0,:,:,0].numpy(), [0.32, 0.05, 0.43, 0.54], 6, 6)


