# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
General plugins for object detection
"""

from functools import partial
from typing import Optional
from typing import Tuple
from typing import Union

import nucleus7 as nc7
import tensorflow as tf

from ncgenes7.data_fields.object_detection import DetectionDataFields
from ncgenes7.data_fields.object_detection import ObjectDataFields
from ncgenes7.utils import object_detection_utils
from ncgenes7.utils.general_utils import broadcast_with_expand_to


class DetectionsClassSelectorPlugin(nc7.model.ModelPlugin):
    """
    Plugin that selects only particular classes and masks all other inputs
    accordingly

    Parameters
    ----------
    classes_to_select
        single class or a list of classes to select

    Attributes
    ----------
    incoming_keys
        * detection_object_classes : classes for detections, 1-based,
          == 0 if no class was found;
          [bs, num_detections]; tf.int
    generated_keys
        * detection_object_classes : classes for detections, 1-based,
          == 0 if no class was found;
          [bs, num_detections]; tf.int same as incoming
        * num_object_detections : number of objects, which is inferred from
          the shape of num_detections; [bs], tf.int32
    """
    incoming_keys = [
        DetectionDataFields.detection_object_classes,
        "_" + DetectionDataFields.num_object_detections,
    ]
    generated_keys = [
        DetectionDataFields.detection_object_classes,
        DetectionDataFields.num_object_detections,
    ]
    dynamic_incoming_keys = True
    dynamic_generated_keys = True

    def __init__(self, *,
                 classes_to_select: Union[int, list],
                 **encoder_kwargs):
        super().__init__(**encoder_kwargs)
        if isinstance(classes_to_select, int):
            classes_to_select = [classes_to_select]
        if not all((isinstance(each_class, int) and each_class >= 0
                    for each_class in classes_to_select)):
            msg = ("{}: provided classes_to_select is invalid! "
                   "It must be either single int or a list of ints "
                   "(provided: {})").format(self.name, classes_to_select)
            raise ValueError(msg)
        self.classes_to_select = classes_to_select

    def predict(self, detection_object_classes: tf.Tensor,
                num_object_detections: Optional[tf.Tensor] = None,
                **dynamic_inputs):
        # pylint: disable=arguments-differ
        # base method has more generic signature
        if 0 in self.classes_to_select and num_object_detections is None:
            msg = "{}: Provide num_object_detections to select 0 class".format(
                self.name)
            raise ValueError(msg)

        classes_mask, num_detections = self._create_classes_mask(
            detection_object_classes, num_object_detections)
        rearranged_indices = tf.contrib.framework.argsort(
            tf.cast(classes_mask, tf.int32), axis=-1,
            direction="DESCENDING")
        result = object_detection_utils.mask_inputs_to_classes(
            detection_object_classes, dynamic_inputs,
            classes_mask,
            partial(_mask_single_batch_input_to_classes,
                    rearranged_indices=rearranged_indices),
        )
        detections_classes_masked = result.pop(ObjectDataFields.object_classes)
        result[DetectionDataFields.detection_object_classes] = (
            detections_classes_masked)
        result[DetectionDataFields.num_object_detections] = num_detections
        return result

    def _create_classes_mask(self, detection_object_classes: tf.Tensor,
                             num_object_detections: Optional[tf.Tensor] = None
                             ) -> Tuple[tf.Tensor, tf.Tensor]:
        classes_mask, num_objects = object_detection_utils.create_classes_mask(
            detection_object_classes, self.classes_to_select,
            num_objects=num_object_detections)
        return classes_mask, num_objects


def _mask_single_batch_input_to_classes(
        item: tf.Tensor, classes_mask: tf.Tensor,
        rearranged_indices: tf.Tensor) -> tf.Tensor:
    mask_broadcasted = broadcast_with_expand_to(classes_mask, item)
    item_masked = tf.where(mask_broadcasted,
                           item,
                           tf.zeros_like(item))
    item_masked_rearranged = tf.batch_gather(item_masked,
                                             rearranged_indices)
    return item_masked_rearranged
