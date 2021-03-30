import config as cfg
import tensorflow as tf
from utils import encode_labels, preprocess_mrcnn, entry_stop_gradients, freeze_backbone, xyxy2xywh, decode_delta, xywh2xyxy
from backbone import cspdarknet53_graph, load_weights_cspdarknet53
from layers import yolov4_plus1_graph, yolov4_plus1_decode_graph, yolov4_plus1_proposal_graph,\
     mask_graph_AFP, PyramidROIAlign_AFP, box_classifier_graph_AFP

class Model(tf.keras.Model):
    
    @tf.function
    def infer(self, inputs):
        return self(inputs, training=False)

    def compile(self, optimizer):
        super(Model, self).compile()
        self.optimizer = optimizer
        
    def train_step(self, data):
        image, gt_masks, gt_bboxes = data
        label_2, label_3, label_4, label_5 = tf.map_fn(encode_labels, gt_bboxes, fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
        data = image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes 
        alb_loss, box_loss, conf_loss, class_loss, rbox_loss, mask_loss = train_step(self, data, self.optimizer)
        return {"alb_loss": alb_loss, "box_loss": box_loss, "conf_loss": conf_loss, "class_loss": class_loss, "rbox_loss": rbox_loss, \
                "mask_loss": mask_loss, "s_r":self.s_r, "s_c":self.s_c, "s_mc":self.s_mc, "s_mr":self.s_mr, "s_mm":self.s_mm }
    
    def test_step(self, data):
        image, gt_masks, gt_bboxes = data
        label_2, label_3, label_4, label_5 = tf.map_fn(encode_labels, gt_bboxes, fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
        data = image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes 
        alb_loss, box_loss, conf_loss, class_loss, rbox_loss, mask_loss = val_step(self, data)
        return {"alb_loss": alb_loss, "box_loss": box_loss, "conf_loss": conf_loss, "class_loss": class_loss, "rbox_loss": rbox_loss,  \
                "mask_loss": mask_loss, "s_r":self.s_r, "s_c":self.s_c, "s_mc":self.s_mc, "s_mr":self.s_mr, "s_mm":self.s_mm }
    
def get_model(pretrained_backbone=True, infer=False):
    input_layer = tf.keras.layers.Input(cfg.INPUT_SHAPE)
    backbone = cspdarknet53_graph(input_layer) # may try a smaller backbone? backbone = cspdarknet53_tiny(input_layer)
    neck = yolov4_plus1_graph(backbone)
    rpn_predictions, rpn_embeddings = yolov4_plus1_decode_graph(neck)
    rpn_proposals = yolov4_plus1_proposal_graph(rpn_predictions,rpn_embeddings)
    pooled_rois_classifier = PyramidROIAlign_AFP((cfg.POOL_SIZE, cfg.POOL_SIZE),name="roi_align_classifier")([rpn_proposals[...,:4],rpn_embeddings])
    pooled_rois_mask = PyramidROIAlign_AFP((cfg.MASK_POOL_SIZE, cfg.MASK_POOL_SIZE),name="roi_align_mask")([rpn_proposals[...,:4],rpn_embeddings])
    mrcnn_class_logits, mrcnn_bbox = box_classifier_graph_AFP(pooled_rois_classifier)
    mrcnn_mask = mask_graph_AFP(pooled_rois_mask)
    #backbone, neck,pooled_rois_classifier, pooled_rois_mask,
    if infer:
        bbox = decode_delta(mrcnn_bbox, xyxy2xywh(rpn_proposals[...,:4]))
        bbox = xywh2xyxy(bbox)
        bbox = tf.clip_by_value(bbox,0.0,1.0)
        bbox = tf.round(bbox*cfg.TRAIN_SIZE)
        conf = tf.nn.softmax(mrcnn_class_logits,axis=-1)
        class_id = tf.add(tf.argmax(conf,axis=-1),tf.constant(1,dtype=tf.int64))
        conf = tf.reduce_max(conf,axis=-1)
        mask = tf.nn.softmax(mrcnn_mask,axis=-1)[...,1]
        model = Model(inputs=input_layer, outputs=[bbox, conf, class_id, mask])
    else:
        model = Model(inputs=input_layer, outputs=[rpn_predictions, rpn_proposals, mrcnn_class_logits, \
                                                   mrcnn_bbox, mrcnn_mask])
        if pretrained_backbone:
            load_weights_cspdarknet53(model, cfg.CSP_DARKNET53) # load backbone weights and set to non trainable
            freeze_backbone(model)
        model.s_c = tf.Variable(initial_value=0.0, trainable=True, name = 's_c')
        model.s_r = tf.Variable(initial_value=0.0, trainable=True, name = 's_r')
        model.s_mc = tf.Variable(initial_value=0.0, trainable=True, name = 's_mc')
        model.s_mr = tf.Variable(initial_value=0.0, trainable=True, name = 's_mr')
        model.s_mm = tf.Variable(initial_value=0.0, trainable=True, name = 's_mm')
        if pretrained_backbone:
            load_weights_cspdarknet53(model, cfg.CSP_DARKNET53) # load backbone weights and set to non trainable
            freeze_backbone(model)
    return model

def train_step(model, data, optimizer):
    training = True
    image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
    labels = [label_2, labe_3, label_4, label_5]
    with tf.GradientTape() as tape:
        preds, proposals, pred_class, pred_bbox, pred_mask = model(image, training=training)
        proposals = proposals[...,:4]
        target_class_ids, target_bboxes, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks) # preprocess and tile labels according to IOU
        alb_loss, *loss_list = compute_loss(model, labels, preds, proposals, target_class_ids, target_bboxes, target_masks, pred_class, pred_bbox, pred_mask)
    gradients = tape.gradient(alb_loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, model.trainable_variables) if grad is not None)
    return [alb_loss] + loss_list

def val_step(model, data):
    training = False
    image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
    labels = [label_2, labe_3, label_4, label_5]
    preds, proposals, pred_class, pred_bbox, pred_mask = model(image, training=training)
    proposals = proposals[...,:4]
    target_class_ids, target_bboxes, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks) # preprocess and tile labels according to IOU
    alb_loss, *loss_list = compute_loss(model, labels, preds, proposals, target_class_ids, target_bboxes, target_masks, pred_class, pred_bbox, pred_mask)
    return [alb_loss] + loss_list

def compute_loss(model, labels, preds, proposals, target_class_ids, target_bboxes, target_masks, pred_class, pred_bbox, pred_mask):
    # rpn loss 
    alb_loss, *rpn_loss_list = compute_loss_rpn(model, labels, preds)
    # mrcnn loss
    alb_total_loss, *mrcnn_loss_list = compute_loss_mrcnn(model, proposals, target_class_ids, target_bboxes, target_masks, pred_class, pred_bbox, pred_mask)
    alb_total_loss += alb_loss
    alb_total_loss *= 0.5

    return [alb_total_loss] + rpn_loss_list + mrcnn_loss_list

def compute_loss_rpn(model, labels, preds):
    box_loss = []
    conf_loss = []
    for label, pred in zip(labels, preds):
        lbox, lconf = compute_loss_rpn_level(label, pred)
        box_loss.append(lbox)
        conf_loss.append(lconf)
    box_loss, conf_loss = tf.reduce_mean(box_loss,axis=0), tf.reduce_mean(conf_loss,axis=0)
    alb_loss = tf.math.exp(-model.s_r)*box_loss + tf.math.exp(-model.s_c)*conf_loss \
                + (model.s_r + model.s_c ) #Automatic Loss Balancing        
    return alb_loss, box_loss, conf_loss

def compute_loss_rpn_level(label, pred):
    pbox, pconf = pred[..., :4], pred[..., 4:6]
    tbox, tconf = label[...,:4], label[...,4]
    # regression loss
    # stop gradient for regions labeled -1 or 0 below CONF threshold
    positive_entry = tf.tile(tf.cast(tf.greater(tconf,0.0)[...,tf.newaxis], tf.float32),(1,1,1,1,4))
    positive_entry = tf.stop_gradient(positive_entry)
    tbox = tf.stop_gradient(tbox)
    lbox = tf.cond(tf.greater(tf.reduce_sum(positive_entry),0.0), lambda: \
                   tf.reduce_mean(smooth_l1_loss(y_true = tbox * positive_entry ,y_pred = pbox * positive_entry)), \
                   lambda: tf.constant(0.0))
    # conf loss
    # stop gradient for regions labeled -1 below CONF threshold
    non_negative_entry = tf.greater_equal(tconf,0.0)
    pconf = entry_stop_gradients(pconf, tf.cast(non_negative_entry[...,tf.newaxis],tf.float32)) 
    tconf = tf.cast(tf.where(non_negative_entry, tconf, 0.0),tf.int32)
    non_negative_entry = tf.stop_gradient(non_negative_entry)
    tconf = tf.stop_gradient(tconf)
    lconf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tconf,logits=pconf))

    return lbox, lconf

def compute_loss_mrcnn(model, proposals, target_class_ids, target_bboxes, target_masks, pred_class, pred_bbox, pred_mask):
    # prepare the ground truth
    mrcnn_class_loss = class_loss_graph(target_class_ids, pred_class)
    mrcnn_box_loss = bbox_loss_graph(target_bboxes, target_class_ids, pred_bbox)
    mrcnn_mask_loss = mask_loss_graph(target_masks, target_class_ids, pred_mask)
    alb_loss = tf.cond(tf.greater(tf.reduce_sum(proposals),0.0), lambda: \
                tf.math.exp(-model.s_mc)*mrcnn_class_loss + tf.math.exp(-model.s_mr)*mrcnn_box_loss \
                + tf.math.exp(-model.s_mm)*mrcnn_mask_loss + (model.s_mr + model.s_mc + model.s_mm), \
                lambda: mrcnn_class_loss + mrcnn_box_loss + mrcnn_mask_loss)
    return alb_loss, mrcnn_class_loss, mrcnn_box_loss, mrcnn_mask_loss  #Automatic Loss Balancing   
   

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def class_loss_graph(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.
    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
        active_class_ids = tf.concat([tf.zeros((nB,1)),tf.ones((nB,1))],axis=-1)
    """

    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
#    target_class_ids = tf.cast(target_class_ids, 'int32')

    # Find predictions of classes that are not in the dataset.
#    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    
#    pred_active = tf.gather(active_class_ids[0], pred_class_ids)
    target_class_ids = tf.cast(target_class_ids, tf.int32)
    
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    
    pred_class_logits = tf.reshape(pred_class_logits, (-1,tf.shape(pred_class_logits)[-1]))
    
    # # Loss
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    
    target_class_ids = tf.gather(target_class_ids, positive_roi_ix, axis=0)
    
    target_class_ids = tf.subtract(target_class_ids, tf.constant(1,tf.int32))
    
    pred_class_logits = tf.gather(pred_class_logits, positive_roi_ix, axis=0)
    
    loss = tf.cond(tf.greater(tf.size(target_class_ids), 0),\
                   lambda: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=pred_class_logits),\
                   lambda: tf.constant(0.0))
    loss = tf.reduce_mean(loss)
    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
#    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    return loss


def bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.
    target_bbox: [batch, num_rois, (dx, dy, log(dw), log(dh))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dx, dy, log(dw), log(dh))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))
    pred_bbox = tf.reshape(pred_bbox, (-1, 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
#    positive_roi_class_ids = tf.cast(
#        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
#    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix, axis=0)
    pred_bbox = tf.gather(pred_bbox, positive_roi_ix, axis=0)
    # Smooth-L1 Loss
    loss = tf.cond(tf.greater(tf.size(target_bbox), 0),\
                   lambda: smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),\
                   lambda: tf.constant(0.0))
    loss = tf.reduce_mean(loss)
    return loss

def mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.
    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    
#    target_class_ids = tf.where(target_class_ids>0,1,0) # added to train with 1 class for mask
    
    mask_shape = tf.shape(target_masks)
    target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = tf.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]

#    positive_class_ids = tf.cast(
#        tf.gather(target_class_ids, positive_ix), tf.int64)
#    positive_class_ids = positive_class_ids - 1# classes starts by 1..41 thus indices 0 to 40
#    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix,axis=0)
#    y_pred = tf.gather_nd(pred_masks, indices)
    y_pred = tf.gather(pred_masks, positive_ix,axis=0)
    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
#    loss = tf.cond(tf.greater(tf.size(y_true), 0),\
#                   lambda: tf.keras.losses.binary_crossentropy(y_true, y_pred),\
#                   lambda: tf.constant(0.0))
    # Permute again masks to [N, height, width, num_classes]
    y_pred = tf.transpose(y_pred, [0, 2, 3, 1])

    # cast to correct label type
    y_true = tf.cast(y_true,tf.int32)
    y_true = tf.stop_gradient(y_true)

    loss = tf.cond(tf.greater(tf.size(y_true), 0),\
                   lambda: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred),\
                   lambda: tf.constant(0.0))
    loss = tf.reduce_mean(loss)
    return loss
#
#def log_compute_mAP(epoch, logs):
#    # Use the model to predict the values from the validation dataset.
#    _, _, rpn_proposals, mrcnn_mask = model.predict(test_images)
#    box, conf, class_id = rpn_proposals[...,:4], rpn_proposals[...,4], rpn_proposals[...,5]
#    box = tf.round(box*cfg.TRAIN_SIZE)
#    mask = tf.nn.softmax(mrcnn_mask,axis=-1)[...,1]
#  
#    image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
#    mean_AP = []
#    boxes, confs, class_ids, masks = prediction
#    for gt_mask, gt_bbox, box, conf, class_id, mask in zip(gt_masks, gt_bboxes, boxes, confs, class_ids, masks):
#        gt_bbox, gt_class_id, gt_mask = decode_ground_truth(gt_mask, gt_bbox)
#        pred_box, pred_score, pred_class_id, pred_mask = decode_mask(box, conf, class_id, mask,'cut')
#        gt_mask = np.transpose(gt_mask,(1,2,0))
#        pred_mask = np.transpose(pred_mask,(1,2,0))
#        if len(gt_bbox)>0: # this is never the case but better to put
#            AP = compute_ap_range(gt_bbox, gt_class_id, gt_mask,
#                     pred_box, pred_class_id, pred_score, pred_mask,
#                     iou_thresholds=iou_thresholds, verbose=verbose)
#            mean_AP.append(AP)
#
#    # Log the confusion matrix as an image summary.
#    with file_writer_cm.as_default():
#        tf.summary.image("Confusion Matrix", cm_image, step=epoch)
#
## Define the per-epoch callback.
#cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
