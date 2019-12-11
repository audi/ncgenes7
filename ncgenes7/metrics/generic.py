# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Generic ModelMetrics
"""
import nucleus7 as nc7
import tensorflow as tf

from ncgenes7.utils.object_detection_io_utils import maybe_create_category_index


class AccuracyMetric(nc7.model.ModelMetric):
    """
    Calculate the accuracy

    Attributes
    ----------
    incoming_keys
        * labels : labels, int32
        * predictions : predictions, int32
    generated_keys
        * scalar_accuracy : accuracy, float32
    """
    incoming_keys = [
        "labels",
        "predictions",
    ]
    generated_keys = [
        "scalar_accuracy",
    ]

    def process(self, labels, predictions):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        _, accuracy_with_update_op = tf.metrics.accuracy(labels, predictions)
        return {"scalar_accuracy": accuracy_with_update_op}


class PerClassAccuracyMetric(nc7.model.ModelMetric):
    """
    Calculates classification accuracy per class with mean accuracy.

    Parameters
    ----------
    num_classes
        number of classes
    class_names_to_labels_mapping
        either str with file name to mapping or dict with mapping itself;
        mapping should be in format {"class name": {"class_id": 1}, },
        where class_id is an unique integer

    Attributes
    ----------
    incoming_keys
        * groundtruth_classes : groundtruth class indices of shape [bs, 1],
          tf.int32
        * predictions : predicted class indices of shape [bs], tf.int32
    generated_keys
        * scalars :  it generates scalars per class accuracy depending on
          number of classes
    """
    incoming_keys = ['groundtruth_classes',
                     'predictions']
    dynamic_generated_keys = True

    def __init__(self, *,
                 num_classes=None,
                 class_names_to_labels_mapping=None,
                 **summary_kwargs):
        super().__init__(**summary_kwargs)
        self.num_classes = num_classes
        self.class_names_to_labels_mapping = class_names_to_labels_mapping
        self.category_index = None

    def build(self):
        super(PerClassAccuracyMetric, self).build()
        self.num_classes, self.category_index = maybe_create_category_index(
            self.num_classes, self.class_names_to_labels_mapping,
            class_offset=0)
        return self

    def process(self, labels, predictions):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        mean_accuracy, accuracy = tf.metrics.mean_per_class_accuracy(
            labels=tf.squeeze(labels, axis=-1), predictions=predictions,
            num_classes=self.num_classes)
        result = {}
        for class_id in self.category_index.keys():
            result['scalar_accuracy_for_class_' +
                   self.category_index[class_id]['name']] = accuracy[class_id]
        result['scalar_mean_accuracy'] = mean_accuracy
        return result
