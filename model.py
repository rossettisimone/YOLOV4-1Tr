import config as cfg
import tensorflow as tf
from utils import encode_labels, preprocess_mrcnn, entry_stop_gradients
from backbone import cspdarknet53_graph, load_weights_cspdarknet53
from layers import yolov4_plus1_graph, yolov4_plus1_decode_graph, yolov4_plus1_proposal_graph,\
     fpn_classifier_graph_AFP, build_fpn_mask_graph_AFP

class Model(tf.keras.Model):
    
    @tf.function
    def infer(self, inputs):
        return self(inputs, training=False)
    
    def compile(self, optimizer):
        super(Model, self).compile()
        self.optimizer = optimizer
        
    def train_step(self, data):
        image, gt_mask, gt_masks, gt_bboxes = data
        label_2, label_3, label_4, label_5 = tf.map_fn(encode_labels, (gt_bboxes, gt_mask), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
        data = image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes 
        alb_total_loss, rpn_box_loss, rpn_class_loss, mrcnn_class_loss, mrcnn_box_loss, mrcnn_mask_loss = train_step(self, data, self.optimizer)
        return {"alb_total_loss": alb_total_loss, "rpn_box_loss": rpn_box_loss, "rpn_class_loss": rpn_class_loss, \
                "mrcnn_class_loss":mrcnn_class_loss, "mrcnn_box_loss":mrcnn_box_loss, "mrcnn_mask_loss": mrcnn_mask_loss,
                "s_r":self.s_r,"s_c":self.s_c, "s_mr":self.s_mr, "s_mc":self.s_mc, "s_mm":self.s_mm }
    
    def test_step(self, data):
        image, gt_mask, gt_masks, gt_bboxes = data
        label_2, label_3, label_4, label_5 = tf.map_fn(encode_labels, (gt_bboxes, gt_mask), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32))
        data = image, label_2, label_3, label_4, label_5, gt_masks, gt_bboxes 
        alb_total_loss, rpn_box_loss, rpn_class_loss, mrcnn_class_loss, mrcnn_box_loss, mrcnn_mask_loss = val_step(self, data)
        return {"alb_total_loss": alb_total_loss, "rpn_box_loss": rpn_box_loss, "rpn_class_loss": rpn_class_loss, \
                "mrcnn_class_loss":mrcnn_class_loss, "mrcnn_box_loss":mrcnn_box_loss, "mrcnn_mask_loss": mrcnn_mask_loss,
                "s_r":self.s_r,"s_c":self.s_c, "s_mr":self.s_mr, "s_mc":self.s_mc, "s_mm":self.s_mm }
    
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

def train_step(model, data, optimizer):
    training = True
    image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
    labels = [label_2, labe_3, label_4, label_5]
    with tf.GradientTape() as tape:
        preds, embs, proposals, pred_class_logits, pred_class, pred_bbox, pred_mask = model(image, training=training)
        proposals = proposals[...,:4]
        target_class_ids, target_bbox, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks) # preprocess and tile labels according to IOU
        alb_total_loss, *loss_list = compute_loss(model, labels, preds, embs, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_mask, training)
    gradients = tape.gradient(alb_total_loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, model.trainable_variables) if grad is not None)
    return [alb_total_loss] + loss_list

def val_step(model, data):
    training = False
    image, label_2, labe_3, label_4, label_5, gt_masks, gt_bboxes = data
    labels = [label_2, labe_3, label_4, label_5]
    preds, embs, proposals, pred_class_logits, pred_class, pred_bbox, pred_mask = model(image, training=training)
    proposals = proposals[...,:4]
    target_class_ids, target_bbox, target_masks = preprocess_mrcnn(proposals, gt_bboxes, gt_masks) # preprocess and tile labels according to IOU
    alb_total_loss, *loss_list = compute_loss(model, labels, preds, embs, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_mask, training)
    return [alb_total_loss] + loss_list

def compute_loss(model, labels, preds, embs, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_masks, training):
    # rpn loss 
    alb_total_loss, *rpn_loss_list = compute_loss_rpn(model, labels, preds, embs)
    # mrcnn loss
    alb_loss, *mrcnn_loss_list = compute_loss_mrcnn(model, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_masks)
    alb_total_loss += alb_loss
    alb_total_loss *= 0.5

    return [alb_total_loss] + rpn_loss_list + mrcnn_loss_list

def compute_loss_rpn(model, labels, preds, embs):
    rpn_box_loss = []
    rpn_class_loss = []
    for label, pred, emb in zip(labels, preds, embs):
        lbox, lconf = compute_loss_rpn_level(label, pred, emb)
        rpn_box_loss.append(lbox)
        rpn_class_loss.append(lconf)
    rpn_box_loss, rpn_class_loss = tf.reduce_mean(rpn_box_loss,axis=0), tf.reduce_mean(rpn_class_loss,axis=0)
    alb_loss = tf.math.exp(-model.s_r)*rpn_box_loss + tf.math.exp(-model.s_c)*rpn_class_loss \
                + (model.s_r + model.s_c) #Automatic Loss Balancing        
    return alb_loss, rpn_box_loss, rpn_class_loss

def compute_loss_rpn_level(label, pred, emb):
    pbox = pred[..., :4]
    pconf = pred[..., 4:6]
    tbox = label[...,:4]
    tconf = label[...,4]
    mask = tf.tile(tf.cast(tf.greater(tconf,0.0), tf.float32)[...,tf.newaxis],(1,1,1,1,4))
    lbox = tf.cond(tf.greater(tf.reduce_sum(mask),0.0), lambda: \
                   tf.reduce_mean(smooth_l1_loss(y_true = tbox * mask,y_pred = pbox * mask)),\
                   lambda: tf.constant(0.0))
    non_negative_entry = tf.cast(tf.greater_equal(tconf[...,tf.newaxis],0.0),tf.float32)
    pconf = entry_stop_gradients(pconf, non_negative_entry) # stop gradient for regions labeled -1 below CONF threshold, look dataloader
    tconf = tf.cast(tf.where(tf.less(tconf, 0.0), 0.0, tconf),tf.int32)
    lconf =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tconf,logits=pconf)) # apply softmax and do non negative log likelihood loss 
    return lbox, lconf

def compute_loss_mrcnn(model, proposals, target_class_ids, target_bbox, target_masks, pred_class_logits, pred_bbox, pred_masks):
    # prepare the ground truth
    mrcnn_class_loss = mrcnn_class_loss_graph(target_class_ids, pred_class_logits)
    mrcnn_box_loss = mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox)
    mrcnn_mask_loss = mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks)
    alb_loss = tf.cond(tf.greater(tf.reduce_sum(proposals),0.0), lambda: \
                tf.math.exp(-model.s_mc)*mrcnn_class_loss + tf.math.exp(-model.s_mr)*mrcnn_box_loss \
                + tf.math.exp(-model.s_mm)*mrcnn_mask_loss + (model.s_mr + model.s_mc + model.s_mm), \
                lambda: mrcnn_class_loss + mrcnn_box_loss + mrcnn_mask_loss)
    return alb_loss, mrcnn_class_loss, mrcnn_box_loss, mrcnn_mask_loss  #Automatic Loss Balancing   


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits):
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

    # # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
#    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_mean(loss) #/ tf.reduce_sum(pred_active)
    return loss


def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.
    target_bbox: [batch, num_rois, (dx, dy, log(dw), log(dh))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dx, dy, log(dw), log(dh))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))
    pred_bbox = tf.reshape(pred_bbox, (-1, tf.shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)
    # Smooth-L1 Loss
    loss = tf.cond(tf.greater(tf.size(target_bbox), 0),\
                   lambda: smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),\
                   lambda: tf.constant(0.0))
    loss = tf.reduce_mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.
    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
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
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = tf.cond(tf.greater(tf.size(y_true), 0),\
                   lambda: tf.keras.losses.binary_crossentropy(y_true, y_pred),\
                   lambda: tf.constant(0.0))

    loss = tf.reduce_mean(loss)
    return loss

def freeze_batch_norm(model, trainable = False):  
    for layer in model.layers:
        bn_layer_name = layer.name
        if bn_layer_name[:10] == 'batch_norm':
            bn_layer = model.get_layer(bn_layer_name)
            bn_layer._trainable = trainable
        else:
            try:
                bn_layer_name = layer.layer.name # TimeDistributed hides the name
                if bn_layer_name[:10] == 'batch_norm':
                    bn_layer = model.get_layer(bn_layer_name)
                    bn_layer._trainable = trainable
            except:
                continue

def freeze_backbone(model, trainable = False):
    cutoff = 78 # 77 convolutions and batch normalizations
    conv_0 = int(model.layers[1].name.split('_')[-1]) if not model.layers[1].name == 'conv2d' else 0
    batch_0 = int (model.layers[2].name.split('_')[-1]) if not model.layers[2].name == 'batch_normalization' else 0
    for i in range(0,cutoff):
        k = i + conv_0
        j = i + batch_0
        conv_layer_name = 'conv2d_%d' %k if k > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'
        conv_layer = model.get_layer(conv_layer_name)
        bn_layer = model.get_layer(bn_layer_name)
        conv_layer._trainable = trainable
        bn_layer._trainable = trainable
            
def freeze_rpn(model, trainable = False):
    cutoff = 561
    for layer in model.layers[:cutoff]:
        layer._trainable = trainable
