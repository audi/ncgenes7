# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Image augmentations
"""
from typing import Dict
from typing import Optional
import warnings

import nucleus7 as nc7
from object_detection.core import box_list
from object_detection.core import box_list_ops
import tensorflow as tf

from ncgenes7.augmentations.image import _ImageRandomCrop
from ncgenes7.augmentations.image import _ImageRandomCutout
from ncgenes7.augmentations.image import _ImageRandomRotation
from ncgenes7.data_fields.object_detection import ObjectDataFields
from ncgenes7.utils import object_detection_utils as od_utils
from ncgenes7.utils.general_utils import broadcast_with_expand_to


class ObjectsHorizontalFlip(nc7.data.RandomAugmentationTf):
    """
    Random horizontal object flipping

    Attributes
    ----------
    incoming_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [num_detections, 4]
          and with values in [0, 1]; tf.float32
        * object_keypoints : object keypoints normalized
          to image coordinates in format [y, x]; tf.float32,
          [num_detections, num_keypoints, 2]
    generated_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [num_detections, 4]
          and with values in [0, 1]; tf.float32
        * object_keypoints : object keypoints normalized
          to image coordinates in format [y, x]; tf.float32,
          [num_detections, num_keypoints, 2]
    """
    incoming_keys = [
        "_" + ObjectDataFields.object_boxes,
        "_" + ObjectDataFields.object_keypoints,
    ]
    generated_keys = [
        "_" + ObjectDataFields.object_boxes,
        "_" + ObjectDataFields.object_keypoints,
    ]
    dynamic_incoming_keys = True
    dynamic_generated_keys = True

    def augment(self, *, object_boxes: Optional[tf.Tensor] = None,
                object_keypoints: Optional[tf.Tensor] = None,
                **dynamic_inputs) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        result = {each_key: each_value
                  for each_key, each_value in dynamic_inputs.items()}
        if object_boxes is not None:
            boxes_flipped = _flip_boxes_left_right(object_boxes)
            result[ObjectDataFields.object_boxes] = boxes_flipped
        if object_keypoints is not None:
            keypoints_flipped = _flip_keypoints_left_right(object_keypoints)
            result[ObjectDataFields.object_keypoints] = keypoints_flipped
        return result


class ObjectsFlipUpDown(nc7.data.RandomAugmentationTf):
    """
    Random up down object flipping

    Attributes
    ----------
    incoming_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [num_detections, 4]
          and with values in [0, 1]; tf.float32
        * object_keypoints : object keypoints normalized
          to image coordinates in format [y, x]; tf.float32,
          [num_detections, num_keypoints, 2]
    generated_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [num_detections, 4]
          and with values in [0, 1]; tf.float32
        * object_keypoints : object keypoints normalized
          to image coordinates in format [y, x]; tf.float32,
          [num_detections, num_keypoints, 2]
    """
    incoming_keys = [
        "_" + ObjectDataFields.object_boxes,
        "_" + ObjectDataFields.object_keypoints,
    ]
    generated_keys = [
        "_" + ObjectDataFields.object_boxes,
        "_" + ObjectDataFields.object_keypoints,
    ]
    dynamic_incoming_keys = True
    dynamic_generated_keys = True

    def augment(self, *, object_boxes: Optional[tf.Tensor] = None,
                object_keypoints: Optional[tf.Tensor] = None,
                **dynamic_inputs) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        result = {each_key: each_value
                  for each_key, each_value in dynamic_inputs.items()}
        if object_boxes is not None:
            boxes_flipped = _flip_boxes_up_down(object_boxes)
            result[ObjectDataFields.object_boxes] = boxes_flipped
        if object_keypoints is not None:
            keypoints_flipped = _flip_keypoints_up_down(object_keypoints)
            result[ObjectDataFields.object_keypoints] = keypoints_flipped
        return result


class ObjectsRandomRotation(_ImageRandomRotation):
    """
    Objects rotation. It will rotate the objects and keypoints using center on
    [0.5, 0.5]. All the objects and keypoints, which coordinates will go out of
    [0, 1] will be removed.

    Parameters
    ----------
    max_angle
        defines boundaries for random angle generation in grads

    Attributes
    ----------
    incoming_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [num_detections, 4]
          and with values in [0, 1]; tf.float32
        * object_keypoints : object keypoints normalized
          to image coordinates in format [y, x]; tf.float32,
          [num_detections, num_keypoints, 2]
    generated_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [num_detections, 4]
          and with values in [0, 1]; tf.float32
        * object_keypoints : object keypoints normalized
          to image coordinates in format [y, x]; tf.float32,
          [num_detections, num_keypoints, 2]

    Notes
    -----
    Bounding boxes will be rotated in the way that resulted
    horizontal / vertical aligned box includes the rotated box. It means that
    it may be much larger as original one, so it is better to perform only small
    rotations.
    """
    incoming_keys = [
        "_" + ObjectDataFields.object_boxes,
        "_" + ObjectDataFields.object_keypoints,
    ]
    generated_keys = [
        "_" + ObjectDataFields.object_boxes,
        "_" + ObjectDataFields.object_keypoints,
    ]
    dynamic_incoming_keys = True
    dynamic_generated_keys = True

    def augment(self, *, object_boxes: Optional[tf.Tensor] = None,
                object_keypoints: Optional[tf.Tensor] = None,
                **dynamic_inputs
                ) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        if object_keypoints is not None:
            object_keypoints = self._rotate_keypoints(object_keypoints)
        if object_boxes is not None:
            warnings.warn("Bounding boxes are rotated and then result box is "
                          "placed over rotated ones, which means that "
                          "they may not be aligned with the object!!!")
            object_boxes = self._rotate_boxes(object_boxes)
            valid_boxes_mask = od_utils.get_valid_objects_boxes_mask(
                object_boxes)
            object_boxes, object_keypoints, dynamic_inputs = _filter_objects(
                object_boxes, object_keypoints, valid_boxes_mask,
                dynamic_inputs)
        result = {}
        if object_keypoints is not None:
            result[ObjectDataFields.object_keypoints] = object_keypoints
        if object_boxes is not None:
            result[ObjectDataFields.object_boxes] = object_boxes
        result.update(dynamic_inputs)
        return result

    def _rotate_boxes(self, object_boxes: tf.Tensor) -> tf.Tensor:
        rotation_angle = self.random_variables["rotation_angle"]
        boxes_points = _get_points_from_bbox(object_boxes)
        boxes_points_rotated = _rotate_points(boxes_points, rotation_angle)
        bboxes_rotated = _get_bbox_from_points(boxes_points_rotated)
        bboxes_rotated = tf.clip_by_value(bboxes_rotated, 0, 1)
        return bboxes_rotated

    def _rotate_keypoints(self, object_keypoints: tf.Tensor) -> tf.Tensor:
        rotation_angle = self.random_variables["rotation_angle"]
        keypoints_rotated = _rotate_points(object_keypoints,
                                           rotation_angle)
        valid_keypoints_mask_first = tf.reduce_all(tf.logical_and(
            tf.greater(object_keypoints, 0),
            tf.less_equal(object_keypoints, 1.0)), -1, keepdims=True)
        valid_keypoints_mask_after_rotation = tf.reduce_all(tf.logical_and(
            tf.greater(keypoints_rotated, 0),
            tf.less_equal(keypoints_rotated, 1.0)), -1, keepdims=True)
        valid_keypoints_mask = tf.logical_and(
            valid_keypoints_mask_first,
            valid_keypoints_mask_after_rotation)
        keypoints_rotated = tf.where(
            broadcast_with_expand_to(valid_keypoints_mask, object_keypoints),
            keypoints_rotated,
            tf.zeros_like(object_keypoints))
        return keypoints_rotated


class ObjectsRandomCrop(_ImageRandomCrop):
    """
    Crop images

    One of scale or size parameter should be provided

    Parameters
    ----------
    scale
        scale for cropping; if the scale is provided as list, then it will be
        uniform sampled from scale = [scale[0] : scale[1]]*2 otherwise
        constant scale will be used
    size
        size of cropped image; if provided, then scale is calculated
        scale = [size[0]/image_orig_size[0], size[1]/image_orig_size[1]]
    offset
        offset of image before cropping; if not provided it will be randomly
        sampled out of interval [image_orig_size-image_cropped_size]

    Attributes
    ----------
    incoming_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [num_detections, 4]
          and with values in [0, 1]; tf.float32
        * object_keypoints : object keypoints normalized
          to image coordinates in format [y, x]; tf.float32,
          [num_detections, num_keypoints, 2]
    generated_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [num_detections, 4]
          and with values in [0, 1]; tf.float32
        * object_keypoints : object keypoints normalized
          to image coordinates in format [y, x]; tf.float32,
          [num_detections, num_keypoints, 2]

    Raises
    ------
    AssertionError if both or none of size and scale are provided
    """
    incoming_keys = [
        "_" + ObjectDataFields.object_boxes,
        "_" + ObjectDataFields.object_keypoints,
    ]
    generated_keys = [
        "_" + ObjectDataFields.object_boxes,
        "_" + ObjectDataFields.object_keypoints,
    ]
    dynamic_incoming_keys = True
    dynamic_generated_keys = True

    def augment(self, *, object_boxes: Optional[tf.Tensor] = None,
                object_keypoints: Optional[tf.Tensor] = None,
                **dynamic_inputs) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        if object_keypoints is not None:
            object_keypoints = self._crop_keypoints(object_keypoints)
        if object_boxes is not None:
            object_boxes = self._crop_boxes(object_boxes)
            valid_boxes_mask = od_utils.get_valid_objects_boxes_mask(
                object_boxes)
            object_boxes, object_keypoints, dynamic_inputs = _filter_objects(
                object_boxes, object_keypoints, valid_boxes_mask,
                dynamic_inputs)
        result = {}
        if object_keypoints is not None:
            result[ObjectDataFields.object_keypoints] = object_keypoints
        if object_boxes is not None:
            result[ObjectDataFields.object_boxes] = object_boxes
        result.update(dynamic_inputs)
        return result

    def _crop_boxes(self, object_boxes: tf.Tensor) -> tf.Tensor:
        crop_offset = self.random_variables["crop_offset"]
        crop_scale = self.random_variables["crop_scale"]
        crop_offset = tf.tile(crop_offset, [2])

        boxes_with_offset = object_boxes - crop_offset
        boxes_cropped = boxes_with_offset / crop_scale
        boxes_cropped = tf.clip_by_value(boxes_cropped, 0, 1)
        return boxes_cropped

    def _crop_keypoints(self, object_keypoints: tf.Tensor) -> tf.Tensor:
        crop_offset = self.random_variables["crop_offset"]
        crop_scale = self.random_variables["crop_scale"]
        keypoints_with_offset = object_keypoints - crop_offset
        keypoints_with_offset_scaled = keypoints_with_offset / crop_scale
        valid_keypoints_mask_first = tf.reduce_all(tf.logical_and(
            tf.greater(object_keypoints, 0),
            tf.less_equal(object_keypoints, 1.0)), -1, keepdims=True)
        valid_keypoints_mask_after_crop = tf.reduce_all(tf.logical_and(
            tf.greater(keypoints_with_offset_scaled, 0),
            tf.less_equal(keypoints_with_offset_scaled, 1.0)
        ), -1, keepdims=True)
        valid_keypoints_mask = tf.logical_and(
            valid_keypoints_mask_first,
            valid_keypoints_mask_after_crop)
        keypoints_cropped = tf.where(
            broadcast_with_expand_to(valid_keypoints_mask, object_keypoints),
            keypoints_with_offset_scaled,
            tf.zeros_like(object_keypoints))
        return keypoints_cropped


class ObjectsRandomCutout(_ImageRandomCutout):
    """
    Set a random rectangle in the image to zero

    Parameters
    ----------
    min_cut_length
        Minimum rectangle length
    max_cut_length
        Maximum rectangle_length
    max_occlusion
        max occlusion of the objects in cut out area; if the occlusion is more
        than this value, object will be deleted

    Attributes
    ----------
    incoming_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [num_detections, 4]
          and with values in [0, 1]; tf.float32
        * object_keypoints : object keypoints normalized
          to image coordinates in format [y, x]; tf.float32,
          [num_detections, num_keypoints, 2]
    generated_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [num_detections, 4]
          and with values in [0, 1]; tf.float32
        * object_keypoints : object keypoints normalized
          to image coordinates in format [y, x]; tf.float32,
          [num_detections, num_keypoints, 2]

    References
    ----------
    DeVries, T., & Taylor, G. W. (2017). Improved regularization of
    convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552.
    """
    incoming_keys = [
        "_" + ObjectDataFields.object_boxes,
        "_" + ObjectDataFields.object_keypoints,
    ]
    generated_keys = [
        "_" + ObjectDataFields.object_boxes,
        "_" + ObjectDataFields.object_keypoints,
    ]
    dynamic_incoming_keys = True
    dynamic_generated_keys = True

    def __init__(self, *,
                 max_occlusion: float = 0.8,
                 **augmentation_kwargs):
        super().__init__(**augmentation_kwargs)
        self.max_occlusion = max_occlusion

    def augment(self, *, object_boxes: Optional[tf.Tensor] = None,
                object_keypoints: Optional[tf.Tensor] = None,
                **dynamic_inputs) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        cutout_box = self._get_cutout_box()
        if object_keypoints is not None:
            object_keypoints = self._cutout_keypoints(
                object_keypoints, cutout_box)
        if object_boxes is not None:
            valid_boxes_mask = self._get_cutout_boxes_mask(
                object_boxes, cutout_box)
            object_boxes, object_keypoints, dynamic_inputs = _filter_objects(
                object_boxes, object_keypoints, valid_boxes_mask,
                dynamic_inputs)
        result = {}
        if object_keypoints is not None:
            result[ObjectDataFields.object_keypoints] = object_keypoints
        if object_boxes is not None:
            result[ObjectDataFields.object_boxes] = object_boxes
        result.update(dynamic_inputs)
        return result

    def _get_cutout_boxes_mask(self, object_boxes, cutout_box):
        cutout_box_list = box_list.BoxList(cutout_box[tf.newaxis, ...])

        object_boxes_list = box_list.BoxList(object_boxes)
        object_boxes_occlusions = box_list_ops.ioa(
            cutout_box_list, object_boxes_list)[0]

        boxes_mask = tf.less_equal(object_boxes_occlusions, self.max_occlusion)
        return boxes_mask

    @staticmethod
    def _cutout_keypoints(object_keypoints, cutout_box):
        (cutout_box_ymin, cutout_box_xmin, cutout_box_ymax, cutout_box_xmax
         ) = tf.split(cutout_box, 4, -1)
        object_keypoints_y = object_keypoints[..., 0]
        object_keypoints_x = object_keypoints[..., 1]

        valid_keypoints_mask_first = tf.reduce_all(tf.logical_and(
            tf.greater(object_keypoints, 0),
            tf.less_equal(object_keypoints, 1.0)), -1, keepdims=True)
        valid_keypoints_mask_cutout = tf.reduce_any(
            tf.stack([tf.greater_equal(object_keypoints_y, cutout_box_ymax),
                      tf.less_equal(object_keypoints_y, cutout_box_ymin),
                      tf.greater_equal(object_keypoints_x, cutout_box_xmax),
                      tf.less_equal(object_keypoints_x, cutout_box_xmin),
                      ], -1),
            -1, keepdims=True)
        valid_keypoints_mask = tf.logical_and(
            valid_keypoints_mask_first,
            valid_keypoints_mask_cutout)
        keypoints_cutout = tf.where(
            broadcast_with_expand_to(valid_keypoints_mask, object_keypoints),
            object_keypoints,
            tf.zeros_like(object_keypoints))
        return keypoints_cutout

    def _get_cutout_box(self):
        cut_lengths = self.random_variables["cut_lengths"]
        cut_offset = self.random_variables["cut_offset"]
        cutout_box_min_coord = tf.concat(cut_offset, axis=0)
        cutout_box_max_coord = (cutout_box_min_coord
                                + tf.concat(cut_lengths, axis=0))
        cutout_box = tf.concat([cutout_box_min_coord, cutout_box_max_coord],
                               axis=0)
        return cutout_box


def _flip_boxes_left_right(boxes):
    ymin, xmin, ymax, xmax = tf.split(boxes, 4, -1)
    width = xmax - xmin
    xmin = 1.0 - width - xmin
    xmax = xmin + width
    boxes_flipped = tf.concat([ymin, xmin, ymax, xmax], -1)
    valid_boxes = tf.greater(width, 0)
    boxes_flipped = tf.where(broadcast_with_expand_to(valid_boxes, boxes),
                             boxes_flipped,
                             tf.zeros_like(boxes_flipped))
    return boxes_flipped


def _flip_boxes_up_down(boxes):
    ymin, xmin, ymax, xmax = tf.split(boxes, 4, -1)
    height = ymax - ymin
    ymin = 1.0 - height - ymin
    ymax = ymin + height
    boxes_flipped = tf.concat([ymin, xmin, ymax, xmax], -1)
    valid_boxes = tf.greater(height, 0)
    boxes_flipped = tf.where(broadcast_with_expand_to(valid_boxes, boxes),
                             boxes_flipped,
                             tf.zeros_like(boxes_flipped))
    return boxes_flipped


def _flip_keypoints_left_right(keypoints):
    keypoints_y, keypoints_x = tf.split(keypoints, 2, -1)
    keypoints_x_flipped = 1 - keypoints_x
    keypoints_flipped = tf.concat([keypoints_y, keypoints_x_flipped], -1)
    valid_keypoints = tf.reduce_any(tf.greater(keypoints, 0), -1)
    keypoints_flipped = tf.where(
        broadcast_with_expand_to(valid_keypoints, keypoints),
        keypoints_flipped,
        tf.zeros_like(keypoints_flipped))
    return keypoints_flipped


def _flip_keypoints_up_down(keypoints):
    keypoints_y, keypoints_x = tf.split(keypoints, 2, -1)
    keypoints_y_flipped = 1 - keypoints_y
    keypoints_flipped = tf.concat([keypoints_y_flipped, keypoints_x], -1)
    valid_keypoints = tf.reduce_any(tf.greater(keypoints, 0), -1)
    keypoints_flipped = tf.where(
        broadcast_with_expand_to(valid_keypoints, keypoints),
        keypoints_flipped,
        tf.zeros_like(keypoints_flipped))
    return keypoints_flipped


def _get_bbox_from_points(points):
    points_y = points[..., 0]
    points_x = points[..., 1]
    xmax = tf.reduce_max(points_x, 1)
    xmin = tf.reduce_min(points_x, 1)
    ymax = tf.reduce_max(points_y, 1)
    ymin = tf.reduce_min(points_y, 1)
    return tf.stack([ymin, xmin, ymax, xmax], -1)


def _rotate_points(points, angle):
    points_shifted = points - 0.5
    alpha_cos = tf.cos(angle)
    alpha_sin = tf.sin(angle)
    rot_matrix = tf.reshape(
        tf.stack([alpha_cos, alpha_sin, -alpha_sin, alpha_cos], 0),
        [2, 2])
    points_rotated_shifted = tf.tensordot(points_shifted, rot_matrix, [2, 1])
    points_rotated = points_rotated_shifted + 0.5
    return points_rotated


def _get_points_from_bbox(bbox):
    ymin, xmin, ymax, xmax = tf.split(bbox, 4, -1)
    point1 = tf.concat([ymin, xmin], -1)
    point2 = tf.concat([ymax, xmin], -1)
    point3 = tf.concat([ymax, xmax], -1)
    point4 = tf.concat([ymin, xmax], -1)
    points = tf.stack([point1, point2, point3, point4], 1)
    return points


def _filter_objects(object_boxes, object_keypoints, valid_boxes_mask,
                    dynamic_inputs):
    additional_inputs = ([] if object_keypoints is None
                         else [object_keypoints])
    (object_boxes, _, *additional_inputs, dynamic_inputs
     ) = od_utils.filter_objects_over_mask(valid_boxes_mask, object_boxes,
                                           *additional_inputs, **dynamic_inputs)
    if object_keypoints is not None:
        object_keypoints = additional_inputs[0]
    return object_boxes, object_keypoints, dynamic_inputs
