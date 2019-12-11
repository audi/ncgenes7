# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""ModelMetric for image related tasks
"""

import nucleus7 as nc7
import tensorflow as tf

from ncgenes7.data_fields.semantic_segmentation import GroundtruthDataFields
from ncgenes7.data_fields.semantic_segmentation import PredictionDataFields


class SemanticSegmentationMetrics(nc7.model.ModelMetric):
    """
    Calculate IOU and confusion matrix for semantic segmentation

    Parameters
    ----------
    num_classes
        number of classes
    resize_labels_to_predictions
        if the labels should be resized with nearest_neighbors to match
        predictions spatial dimension

    Attributes
    ----------
    incoming_keys
        * predicted_classes : predicted classes, tf.int, [bs, h, w]
        * labels :  ground truth labels, tf.int, [bs, h, w]
        * mask : (optional) batch mask

    generated_keys
        * scalar_iou : intersection over union
        * image_confusion_matrix : confusion matrix as image,
          [num_classes, num_classes]
        * scalar_iou_for_class_{:02d} : iou for each class,
          This key will not be generated!
    """
    incoming_keys = [
        PredictionDataFields.prediction_segmentation_classes,
        GroundtruthDataFields.groundtruth_segmentation_classes,
        "_mask",
    ]
    generated_keys = [
        "scalar_iou",
        "image_confusion_matrix",
    ]

    def __init__(self, *,
                 num_classes: int,
                 resize_labels_to_predictions=True,
                 **metrics_kwargs):
        super(SemanticSegmentationMetrics, self).__init__(**metrics_kwargs)
        self.num_classes = num_classes
        self.resize_labels_to_predictions = resize_labels_to_predictions

    def process(self,
                prediction_segmentation_classes,
                groundtruth_segmentation_classes,
                mask=None):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        result = {}
        if self.resize_labels_to_predictions:
            logits_size = tf.shape(prediction_segmentation_classes)[1:3]
            groundtruth_segmentation_classes = tf.image.resize_nearest_neighbor(
                groundtruth_segmentation_classes, logits_size)
        if mask is not None:
            mask = tf.reshape(mask, [-1] + [1] * (len(
                groundtruth_segmentation_classes.get_shape()) - 1))
            tile_multiplier = tf.concat([[1], tf.shape(
                groundtruth_segmentation_classes)[1:]], 0)
            mask = tf.to_float(tf.tile(mask, tile_multiplier))
        iou, confusion = tf.metrics.mean_iou(
            labels=groundtruth_segmentation_classes,
            predictions=prediction_segmentation_classes,
            num_classes=self.num_classes,
            weights=mask)
        confusion_image = tf.reshape(
            tf.cast(confusion, tf.float32),
            [1, self.num_classes, self.num_classes, 1])
        result['image_confusion_matrix'] = confusion_image
        result['scalar_iou'] = iou

        iou_classes = _iou_for_each_class(confusion)
        for each_class_id in range(self.num_classes):
            name = 'scalar_iou_for_class_{:02d}'.format(each_class_id)
            result[name] = tf.gather(iou_classes, each_class_id)

        return result


def _iou_for_each_class(confusion_matrix):
    cm_tp = tf.diag_part(confusion_matrix)
    cm_add_rows = tf.reduce_sum(confusion_matrix, 0)
    cm_add_cols = tf.reduce_sum(confusion_matrix, 1)
    cm_tp_fp_fn = cm_add_rows + cm_add_cols - cm_tp

    cm_tp_fp_fn = tf.where(tf.equal(cm_tp_fp_fn, 0),
                           tf.ones_like(cm_tp_fp_fn), cm_tp_fp_fn)
    iou = tf.div(cm_tp, cm_tp_fp_fn)

    return iou
