import config as cfg
import tensorflow as tf
from utils import encode_labels, preprocess_mrcnn, entry_stop_gradients, freeze_backbone
from backbone import cspdarknet53_graph, load_weights_cspdarknet53
from layers import yolov4_plus1_graph, yolov4_plus1_decode_graph, yolov4_plus1_proposal_graph,\
     mask_graph_AFP, PyramidROIAlign_AFP

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
        alb_loss, box_loss, conf_loss, class_loss, mask_loss = train_step(self, data, self.optimizer)
        return {"alb_loss": alb_loss, "box_loss": box_loss, "conf_loss": conf_loss, "class_loss": class_loss, \
                "mask_loss": mask_loss, "s_r":self.s_r, "s_c":self.s_c, "s_d":self.s_d, "s_m":self.s_m }
    
    def test_step(self, data):
        image, gt_masks, gt_bboxes = data
        label_2, label_3, label_4, label_5 = tf.map_fn(encode_labels, gt_bboxes, fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
        data = image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes 
        alb_loss, box_loss, conf_loss, class_loss, mask_loss = val_step(self, data)
        return {"alb_loss": alb_loss, "box_loss": box_loss, "conf_loss": conf_loss," class_loss": class_loss,  \
                "mask_loss": mask_loss, "s_r":self.s_r, "s_c":self.s_c, "s_d":self.s_d, "s_m":self.s_m }
    
def get_model(pretrained_backbone=True):
    input_layer = tf.keras.layers.Input(cfg.INPUT_SHAPE)
    backbone = cspdarknet53_graph(input_layer) # may try a smaller backbone? backbone = cspdarknet53_tiny(input_layer)
    neck = yolov4_plus1_graph(backbone)
    rpn_predictions, rpn_embeddings = yolov4_plus1_decode_graph(neck)
    rpn_proposals = yolov4_plus1_proposal_graph(rpn_predictions)
    pooled_rois_mask = PyramidROIAlign_AFP((cfg.MASK_POOL_SIZE, cfg.MASK_POOL_SIZE),name="roi_align_mask")([rpn_proposals[...,:4],rpn_embeddings])
    mrcnn_mask = mask_graph_AFP(pooled_rois_mask)
    #backbone, neck,pooled_rois_classifier, pooled_rois_mask,
    model = Model(inputs=input_layer, outputs=[rpn_predictions, rpn_proposals, mrcnn_mask])
    model.s_c = tf.Variable(initial_value=0.0, trainable=True, name = 's_c')
    model.s_r = tf.Variable(initial_value=0.0, trainable=True, name = 's_r')
    model.s_d = tf.Variable(initial_value=0.0, trainable=True, name = 's_d')
    model.s_m = tf.Variable(initial_value=0.0, trainable=True, name = 's_m')
    if pretrained_backbone:
        load_weights_cspdarknet53(model, cfg.CSP_DARKNET53) # load backbone weights and set to non trainable
        freeze_backbone(model)
    return model

def train_step(model, data, optimizer):
    training = True
    image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
    labels = [label_2, labe_3, label_4, label_5]
    with tf.GradientTape() as tape:
        preds, proposals, pred_mask = model(image, training=training)
        proposals = proposals[...,:4]
        target_class_ids, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks) # preprocess and tile labels according to IOU
        alb_loss, *loss_list = compute_loss(model, labels, preds, proposals, target_class_ids, target_masks, pred_mask, training)
    gradients = tape.gradient(alb_loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, model.trainable_variables) if grad is not None)
    return [alb_loss] + loss_list

def val_step(model, data):
    training = False
    image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
    labels = [label_2, labe_3, label_4, label_5]
    preds, proposals, pred_mask = model(image, training=training)
    proposals = proposals[...,:4]
    target_class_ids, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks) # preprocess and tile labels according to IOU
    alb_loss, *loss_list = compute_loss(model, labels, preds, proposals, target_class_ids, target_masks, pred_mask, training)
    return [alb_loss] + loss_list

def compute_loss(model, labels, preds, proposals, target_class_ids, target_masks, pred_masks, training):
    # rpn loss 
    alb_loss, *rpn_loss_list = compute_loss_rpn(model, labels, preds)
    # mrcnn loss
    alb_total_loss, *mrcnn_loss_list = compute_loss_mrcnn(model, proposals, target_class_ids, target_masks, pred_masks)
    alb_total_loss += alb_loss
    alb_total_loss *= 0.5

    return [alb_total_loss] + rpn_loss_list + mrcnn_loss_list

def compute_loss_rpn(model, labels, preds):
    box_loss = []
    conf_loss = []
    class_loss = []
    for label, pred in zip(labels, preds):
        lbox, lconf, lclass = compute_loss_rpn_level(label, pred)
        box_loss.append(lbox)
        conf_loss.append(lconf)
        class_loss.append(lclass)
    box_loss, conf_loss, class_loss = tf.reduce_mean(box_loss,axis=0), tf.reduce_mean(conf_loss,axis=0), tf.reduce_mean(class_loss,axis=0)
    alb_loss = tf.math.exp(-model.s_r)*box_loss + tf.math.exp(-model.s_c)*conf_loss + tf.math.exp(-model.s_d)*class_loss \
                + (model.s_r + model.s_c + model.s_d) #Automatic Loss Balancing        
    return alb_loss, box_loss, conf_loss, class_loss

def compute_loss_rpn_level(label, pred):
    pbox, pconf, pclass = pred[..., :4], pred[..., 4:6], pred[..., 6:]
    tbox, tconf, tid = label[...,:4], label[...,4:5], label[..., 5:6]
    mask = tf.cast(tf.greater(tconf,0.0), tf.float32)
#    class_mask = tf.cast(tf.math.reduce_max(mask, axis=1), tf.bool)
    mask = tf.tile(mask,(1,1,1,1,4))
    lbox = tf.cond(tf.greater(tf.reduce_sum(mask),0.0), lambda: \
                   tf.reduce_mean(smooth_l1_loss(y_true = tbox * mask,y_pred = pbox * mask)), lambda: tf.constant(0.0))
#    labels_one_hot = tf.one_hot(labels, n_classes)
#    loss = tf.nn.softmax_cross_entropy_with_logits(
#          labels=labels_one_hot,
#          logits=logits)
#    pclass = entry_stop_gradients(pclass, tf.cast(tf.greater(tconf,0.0),tf.float32))
    non_negative_entry = tf.cast(tf.greater_equal(tconf,0.0),tf.float32)
    pconf = entry_stop_gradients(pconf, non_negative_entry) # stop gradient for regions labeled -1 below CONF threshold, look dataloader
    tconf = tf.cast(tf.where(tf.less(tconf, 0.0), 0.0, tconf),tf.int32)
    tconf = tf.squeeze(tconf,axis=-1)
    lconf =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tconf,logits=pconf))
#    tid = tf.math.reduce_max(tid, axis=1)
#    tid = tf.squeeze(tid[class_mask],axis=-1)
#    pclass = tf.boolean_mask(pclass, class_mask)    
    non_zero_entry = tf.cast(tf.greater_equal(tid,0.0),tf.float32)
    pclass = entry_stop_gradients(pclass, non_zero_entry)
    tid = tf.squeeze(tid,axis=-1)
    mask = tf.greater_equal(tid,0.0)
    tid = tf.cast(tf.where(tf.less(tid, 0.0), 0.0, tid),tf.int32)
    lclass = tf.cond(tf.greater(tf.shape(tid[mask])[0],0), \
                     lambda: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tid[mask],logits=pclass[mask])),\
                     lambda: tf.constant(0.0))
    return lbox, lconf, lclass

def compute_loss_mrcnn(model, proposals, target_class_ids, target_masks, pred_masks):
    # prepare the ground truth
    mask_loss = mask_loss_graph(target_masks, target_class_ids, pred_masks)
    alb_loss = tf.cond(tf.greater(tf.reduce_sum(proposals),0.0), lambda: tf.math.exp(-model.s_m)*mask_loss + (model.s_m), \
                lambda: mask_loss)
    return alb_loss, mask_loss  #Automatic Loss Balancing   

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
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
    
    target_class_ids = tf.where(target_class_ids>0,1,0) # added to train with 1 class for mask
    
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
    y_true = tf.gather(target_masks, positive_ix)
#    y_pred = tf.gather_nd(pred_masks, indices)
    y_pred = tf.gather(pred_masks, positive_ix)
    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
#    loss = tf.cond(tf.greater(tf.size(y_true), 0),\
#                   lambda: tf.keras.losses.binary_crossentropy(y_true, y_pred),\
#                   lambda: tf.constant(0.0))
    # Permute again masks to [N, height, width, num_classes]
    y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
    # cast to correct label type
    y_true = tf.cast(y_true,tf.int32)
    loss = tf.cond(tf.greater(tf.size(y_true), 0),\
                   lambda: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred),\
                   lambda: tf.constant(0.0))
    
    loss = tf.reduce_mean(loss)
    return loss
