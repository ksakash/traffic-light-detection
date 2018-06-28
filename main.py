from tensorflow.python.tools import inspect_checkpoint as chkp
import tensorflow as tf

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from nets import inception_utils

import faster_rcnn_meta_arch 
import rfcn_meta_arch
import faster_rcnn_inception_resnet_v2_feature_extractor

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import box_predictor
from object_detection.core import post_processing
from object_detection.core import preprocessor
from object_detection.core import losses
from object_detection.utils import context_manager

import numpy as np

slim = tf.contrib.slim

anchor_generator = grid_anchor_generator.GridAnchorGenerator(scales=(0.5, 1.0, 2.0), 
                                                            aspect_ratios=(0.5, 1.0, 2.0),
                                                            base_anchor_size=[3, 3],
                                                            anchor_stride=[3, 3])

rfcn_box_predictor = box_predictor.RfcnBoxPredictor(is_training=True, 
                                                        num_classes=12, 
                                                        conv_hyperparams_fn=None,
                                                        num_spatial_bins=[3, 3],
                                                        depth=3,
                                                        crop_size=9,
                                                        box_code_size=9)

def logits_to_probabilities(logits):
    ar = np.array(logits)
    scores = tf.nn.softmax(ar)
    return scores

def scope_fn():
    batch_norm = slim.batch_norm
    affected_ops = [slim.conv2d, slim.separable_conv2d, slim.fully_connected]
    batch_norm_params = {
        'decay': batch_norm.decay,
        'center': batch_norm.center,
        'scale': batch_norm.scale,
        'epsilon': batch_norm.epsilon,
        # Remove is_training parameter from here and deprecate it in the proto
        # once we refactor Faster RCNN models to set is_training through an outer
        # arg_scope in the meta architecture.
        'is_training': True and batch_norm.train
    }
    with (slim.arg_scope([slim.batch_norm], **batch_norm_params)
            if batch_norm_params is not None else
            context_manager.IdentityContextManager()):
        with slim.arg_scope(
            affected_ops,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            weights_regularizer=slim.l2_regularizer(0.0005),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm) as sc:
            return sc

def resizer_fn(image):
    return preprocessor.random_resize_method(image, [256, 256])

FasterRCNNArch = rfcn_meta_arch.RFCNMetaArch(is_training=True,
                                            num_classes=10,
                                            image_resizer_fn=resizer_fn,
                                            feature_extractor=faster_rcnn_inception_resnet_v2_feature_extractor.FasterRCNNInceptionResnetV2FeatureExtractor,
                                            number_of_stages=2,
                                            first_stage_anchor_generator=anchor_generator,
                                            first_stage_atrous_rate=1,
                                            first_stage_box_predictor_arg_scope_fn=scope_fn,
                                            first_stage_box_predictor_kernel_size=3,
                                            first_stage_box_predictor_depth=3,
                                            first_stage_minibatch_size=10,
                                            first_stage_positive_balance_fraction=0.5,
                                            first_stage_nms_score_threshold=0,
                                            first_stage_nms_iou_threshold=0.5,
                                            first_stage_max_proposals=50,
                                            first_stage_localization_loss_weight=0.5,
                                            first_stage_objectness_loss_weight=0.5,
                                            second_stage_mask_rfcn_box_predictor=rfcn_box_predictor,
                                            second_stage_batch_size=10,
                                            second_stage_balance_fraction=0.25,
                                            second_stage_non_max_suppression_fn=post_processing.batch_multiclass_non_max_suppression,
                                            second_stage_score_conversion_fn=logits_to_probabilities,
                                            second_stage_localization_loss_weight=0.5,
                                            second_stage_classification_loss_weight=0.5,
                                            second_stage_classification_loss=losses.WeightedSoftmaxClassificationLoss,
                                            second_stage_mask_prediction_loss_weight=1.0
                                            hard_example_miner=None)


