# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""Summaries for object detection
"""

import nucleus7 as nc7
from nucleus7.utils import deprecated
from object_detection.utils.visualization_utils import (
    draw_bounding_boxes_on_image_tensors)
import tensorflow as tf

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.data_fields.object_detection import ObjectDataFields
from ncgenes7.utils.object_detection_io_utils import maybe_create_category_index
from ncgenes7.utils.object_detection_utils import get_true_image_shapes


class ObjectDrawerSummary(nc7.model.ModelSummary):
    """
    Summary to draw the detections on images.

    Parameters
    ----------
    num_classes
        number of classes
    class_names_to_labels_mapping
        either str with file name to mapping or dict with mapping itself;
        mapping should be in format {"class name": {"class_id": 1}, },
        where class_id is an unique integer
    max_boxes_to_draw
        maximum number of boxes to draw on an image
    use_normalized_coordinates
        if the boxes (and keypoints if provided) are in normalized to image
        coordinates
    score_threshold
        score threshold to use for boxes

    Attributes
    ----------
    incoming_keys
        * images : batch of images; [bs, height, width, num_channels];
          tf.float32 in range [0, 1]
        * object_boxes : object boxes either in normalized or absolute
          coordinates
        * object_scores : (optional) object scores; if not defined,
          will be ones like with shape of boxes
        * object_classes : (optional) object classes; if not defined
          assumed to have only one class and it will be calculated from
          scores
        * object_instance_ids : (optional) instance ids of objects;
          if provided, colors of the drawn boxes will be calculated according
          to ids, not classes; [bs, num_objects], tf.int32
        * object_keypoints : detection keypoints in format [y, x] normalized
          to image coordinates if use_normalized_coordinates==True or in
          absolute coordinates otherwise; tf.float32,
          [bs, num_detections, num_keypoints, 2]
        * object_instance_masks_on_image : instance masks reframed to image;
          [bs, num_objects image_height, image_width]; tf.uint8
    generated_keys
        * images_with_objects : images with bounding boxes
    """
    incoming_keys = [
        ImageDataFields.images,
        ObjectDataFields.object_boxes,
        "_" + ObjectDataFields.object_scores,
        "_" + ObjectDataFields.object_classes,
        "_" + ObjectDataFields.object_instance_ids,
        "_" + ObjectDataFields.object_keypoints,
        "_" + ObjectDataFields.object_instance_masks_on_image,
    ]
    generated_keys = [
        ObjectDataFields.images_with_objects,
    ]

    @deprecated.replace_deprecated_parameter('boxes_in_normalized_coordinates',
                                             'use_normalized_coordinates',
                                             required=False)
    def __init__(self, *,
                 num_classes=None,
                 max_boxes_to_draw=20,
                 class_names_to_labels_mapping=None,
                 use_normalized_coordinates=True,
                 score_threshold: float = 0.1,
                 **summary_kwargs):
        super().__init__(**summary_kwargs)
        self.class_names_to_labels_mapping = class_names_to_labels_mapping
        self.num_classes = num_classes
        self.category_index = None
        self.max_boxes_to_draw = max_boxes_to_draw
        self.use_normalized_coordinates = use_normalized_coordinates
        self.score_threshold = score_threshold

    def build(self):
        super(ObjectDrawerSummary, self).build()
        self.num_classes, self.category_index = maybe_create_category_index(
            self.num_classes, self.class_names_to_labels_mapping)
        return self

    def process(self, images,
                object_boxes,
                object_scores=None, object_classes=None,
                object_instance_ids=None,
                object_keypoints=None,
                object_instance_masks_on_image=None,
                ):
        # pylint: disable=arguments-differ,too-many-arguments
        # base class has more generic signature
        images = tf.cast(images * 256.0, tf.uint8)
        if object_scores is None:
            object_scores = tf.ones_like(object_boxes, tf.float32)[..., 0]
        if object_classes is None:
            ones = tf.ones_like(object_scores, tf.int64)
            zeros = tf.zeros_like(object_scores, tf.int64)
            object_classes = tf.where(object_scores > 0, ones, zeros)
        object_classes = tf.cast(object_classes, tf.int64)
        if len(images.get_shape()) == 3:
            images = tf.expand_dims(images, -1)
        if images.get_shape().as_list()[-1] == 1:
            images = tf.tile(images, [1, 1, 1, 3])

        true_image_shapes = get_true_image_shapes(images)
        image_with_detections = draw_bounding_boxes_on_image_tensors(
            images=images,
            boxes=object_boxes,
            classes=object_classes,
            scores=object_scores,
            keypoints=object_keypoints,
            instance_masks=object_instance_masks_on_image,
            category_index=self.category_index,
            track_ids=object_instance_ids,
            max_boxes_to_draw=self.max_boxes_to_draw,
            true_image_shape=true_image_shapes,
            original_image_spatial_shape=true_image_shapes[:, :2],
            min_score_thresh=self.score_threshold,
            use_normalized_coordinates=self.use_normalized_coordinates,
        )
        image_with_detections = tf.reshape(
            image_with_detections, tf.shape(images))
        result = {
            ObjectDataFields.images_with_objects: image_with_detections
        }
        return result
