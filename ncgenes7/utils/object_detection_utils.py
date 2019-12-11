# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for bounding box operations

All of operations work on bounding boxes in format [ymin, xmin, ymax, xmax]
"""
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from nucleus7.utils import nest_utils
from nucleus7.utils import tf_varscopes_utils
import numpy as np
from object_detection.anchor_generators.grid_anchor_generator import (
    GridAnchorGenerator)
from object_detection.builders.hyperparams_builder import KerasLayerHyperparams
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import standard_fields as box_standard_fields
from object_detection.utils import np_box_list
from object_detection.utils import ops as od_utils_ops
import tensorflow as tf

from ncgenes7.data_fields.object_detection import ObjectDataFields
from ncgenes7.third_party.object_detection.np_box_list_ops import (
    multi_class_non_max_suppression_with_extra_fields)
from ncgenes7.utils.general_utils import broadcast_with_expand_to


class GridAnchorGeneratorRelative(GridAnchorGenerator):
    """
    Grid generator with possibly relative anchor base
    """

    def rescale(self, image_size: Optional[tf.Tensor] = None,
                feature_maps_size: Optional[tf.Tensor] = None,
                rescale_aspect_ratios=False):
        """
        Set the anchor base to be image size, strides to the ratio between
        feature maps of RPN and image size and also the aspect rations will
        be relative to aspect ratio of image

        Aspect ratio are the width / height ratio

        Parameters
        ----------
        image_size
            image size in format [height, width]
        feature_maps_size
            size of rpn feature maps in format [height, width]
        rescale_aspect_ratios
            if the aspect ratios should ber also rescaled relative to image
            aspect ratio
        """
        if image_size is not None:
            self._base_anchor_size = tf.cast(image_size, tf.float32)
        if feature_maps_size is not None:
            self._anchor_stride = (tf.cast(image_size, tf.float32)
                                   / tf.cast(feature_maps_size, tf.float32))
        # as the image ratio may be not 1 and desired aspect ratios are
        # absolute aspect ratios, rescale them according to image aspect ratio
        if rescale_aspect_ratios:
            image_aspect_ratio = (tf.cast(image_size[0], tf.float32)
                                  / tf.cast(image_size[1], tf.float32))
            self._aspect_ratios = ([tf.cast(v, tf.float32) * image_aspect_ratio
                                    for v in self._aspect_ratios])


# pylint: disable=super-init-not-called
# it must inherit from class but have different constructor
class KerasLayerHyperparamsFromData(KerasLayerHyperparams):
    """
    Since ncgenes7 does not use protobuf as configs, this class is using the
    built objects and have the same interfaces as original KerasLayerHyperparams

    Parameters
    ----------
    initializer
        kernel_initializer
    activation
        activation function to use
    batch_norm_params
        keras batch normalization parameters, which may have following keys:
        momentum, center, scale, epsilon

    """

    def __init__(
            self,
            activation: Callable[[tf.Tensor], tf.Tensor],
            initializer: Optional[tf.keras.initializers.Initializer] = None,
            batch_norm_params: Optional[dict] = None):
        self._activation_fn = activation
        self._op_params = {
            "kernel_regularizer": None,
            "kernel_initializer": initializer,
            "activation": self._activation_fn,
        }
        self._batch_norm_params = batch_norm_params


# pylint: enable=super-init-not-called

def normalize_bbox(bboxes: tf.Tensor,
                   image_size: Optional[Union[list, tf.Tensor]] = None,
                   *, true_image_shapes: Optional[tf.Tensor] = None
                   ) -> tf.Tensor:
    """
    Convert bounding boxes from image space to normalized coordinates, e.g.
    divide x coordinates over width and y coordinates over height and so
    resulted coordinates are in [0, 1]

    Parameters
    ----------
    bboxes
        bounding boxes in format [ymin, xmin, ymax, xmax] in image coordinates
    image_size
        image size in format [height, width]
    true_image_shapes
        true image shapes in format
        [bs, [image_height, image_width, num_channels]]; one of image_size
        or true_image_shapes must be provided

    Returns
    -------
    normalized_bboxes
        bboxes in format [ymin, xmin, ymax, xmax] where all coordinates are in
        [0, 1]
    """
    assert (true_image_shapes is None) != (image_size is None), (
        "Either image_size or true_image_shapes must be provided!")
    multiplier = _get_shape_multiplier(image_size, true_image_shapes, bboxes)
    bboxes = bboxes / multiplier
    return bboxes


def normalize_bbox_np(bboxes: np.ndarray, image_size: list) -> np.ndarray:
    """
    Convert bounding boxes from image space to normalized coordinates, e.g.
    divide x coordinates over width and y coordinates over height and so
    resulted coordinates are in [0, 1]

    Parameters
    ----------
    bboxes
        bounding boxes in format [ymin, xmin, ymax, xmax] in image coordinates
    image_size
        image size in format [height, width]

    Returns
    -------
    normalized_bboxes
        bboxes in format [ymin, xmin, ymax, xmax] where all coordinates are in
        [0, 1]
    """
    height, width = image_size
    # pylint: disable=unbalanced-tuple-unpacking
    ymin, xmin, ymax, xmax = np.split(bboxes, 4, -1)
    # pylint: enable=unbalanced-tuple-unpacking
    ymin = ymin / height
    ymax = ymax / height
    xmin = xmin / width
    xmax = xmax / width
    bboxes = np.concatenate([ymin, xmin, ymax, xmax], -1)
    return bboxes


def local_to_image_coordinates(bboxes: tf.Tensor,
                               image_size: Union[tf.Tensor, list] = None,
                               *, true_image_shapes: Optional[tf.Tensor] = None
                               ) -> tf.Tensor:
    """
    Convert bounding boxes from normalized coordinates in [0, 1] to image space
    e.g. multiply x coordinates with width and y coordinates with height and so
    resulted x coordinates [0, width] and y in [0, height]

    Parameters
    ----------
    bboxes
        normalized bounding boxes in format [ymin, xmin, ymax, xmax]
    image_size
        images size in like [height, width]
    true_image_shapes
        true image shapes in format
        [bs, [image_height, image_width, num_channels]]; one of image_size
        or true_image_shapes must be provided

    Returns
    -------
    bboxes_in_image_frame
        bboxes in format [ymin, xmin, ymax, xmax] in image coordinates
    """
    assert (true_image_shapes is None) != (image_size is None), (
        "Either image_size or true_image_shapes must be provided!")
    multiplier = _get_shape_multiplier(image_size, true_image_shapes, bboxes)
    bboxes = bboxes * multiplier
    return bboxes


def keypoints_local_to_image_coordinates(keypoints: tf.Tensor,
                                         image_size: Union[tf.Tensor, list]
                                         ) -> tf.Tensor:
    """
    Convert bounding boxes from normalized coordinates in [0, 1] to image space
    e.g. multiply x coordinates with width and y coordinates with height and so
    resulted x coordinates [0, width] and y in [0, height]

    Parameters
    ----------
    keypoints
        normalized keypoints in format [y, x]
    image_size
        images size in like [height, width]

    Returns
    -------
    keypoints_in_image_frame
        keypoints in format [y, x] in image coordinates
    """
    height, width = image_size[0], image_size[1]
    keypoints_ndim = len(keypoints.get_shape().as_list())
    multiplier = tf.cast(tf.stack([height, width]), tf.float32)
    keypoints = keypoints * tf.reshape(
        multiplier, [1] * (keypoints_ndim - 1) + [-1])
    return keypoints


def local_to_image_coordinates_np(bboxes: np.ndarray, image_size: list
                                  ) -> np.ndarray:
    """
    Convert bounding boxes from normalized coordinates in [0, 1] to image space
    e.g. multiply x coordinates with width and y coordinates with height and so
    resulted x coordinates [0, width] and y in [0, height]

    Parameters
    ----------
    bboxes
        normalized bounding boxes in format [ymin, xmin, ymax, xmax]
    image_size
        images size in like [height, width]

    Returns
    -------
    normalized_bboxes
        bboxes in format [ymin, xmin, ymax, xmax] in image coordinates
    """
    height, width = image_size
    multiplier = np.stack([height, width, height, width])
    bboxes = bboxes * np.reshape(multiplier, [1] * (bboxes.ndim - 1) + [-1])
    return bboxes


def get_category_index(num_classes, class_offset: int = 1):
    """
    Get category index in format {"class name": {"id": , {"name": ,}}, ...}.
    This index is used by object_detection API

    Parameters
    ----------
    num_classes
        number of classes
    class_offset
        class id to start from 0 (for classification) or 1 (for object
        detection)

    Returns
    -------
    category_index
        category index in format {"class name": {"id": , {"name": ,}}, ...}
    """

    return {i: {'id': i, 'name': 'class_{}'.format(i)}
            for i in range(class_offset, num_classes + class_offset)}


def get_filter_mask_by_width(object_boxes: tf.Tensor,
                             min_object_width: Union[float, tf.Tensor],
                             max_object_width: Union[float, tf.Tensor]
                             ) -> tf.Tensor:
    """
    Get the mask of objects with min_object_width <= width <= max_object_width

    Parameters
    ----------
    object_boxes
        boxes of objects
    min_object_width
        min width of filtered objects
    max_object_width
        max width of filtered objects; if equal to 0 then no filter over max
        width will be applied

    Returns
    -------
    mask
        mask of objects which have min_object_width <= width <= max_object_width
    """
    _, xmin, _, xmax = tf.split(object_boxes, 4, -1)
    width = xmax - xmin
    width = tf.squeeze(width, -1)
    mask = _get_interval_mask(width, min_object_width, max_object_width)
    return mask


def get_filter_mask_by_height(object_boxes: tf.Tensor,
                              min_object_height: Union[float, tf.Tensor],
                              max_object_height: Union[float, tf.Tensor]
                              ) -> tf.Tensor:
    """
    Get the mask of objects with
    min_object_height <= height <= max_object_height

    Parameters
    ----------
    object_boxes
        boxes of objects
    min_object_height
        min height of filtered objects
    max_object_height
        max height of filtered objects; if equal to 0 then no filter over max
        height will be applied

    Returns
    -------
    mask
        mask of objects which have
        min_object_height <= height <= max_object_height
    """
    ymin, _, ymax, _ = tf.split(object_boxes, 4, -1)
    height = ymax - ymin
    height = tf.squeeze(height, -1)
    mask = _get_interval_mask(height, min_object_height, max_object_height)
    return mask


def stack_and_pad_object_data(list_of_object_data: List[np.ndarray],
                              pad_value=0) -> np.ndarray:
    """
    Stack the object data and pad it with pad_value over the first dimension

    Parameters
    ----------
    list_of_object_data
        list with sample data, which can have different dimensions over first
        dimension
    pad_value
        value to use as padding

    Returns
    -------
    stacked_and_padded_data
        stack and padded batch object data
    """
    max_length = max(
        [each_sample.shape[0] for each_sample in list_of_object_data])
    pad_lengths = [max_length - each_sample.shape[0]
                   for each_sample in list_of_object_data]
    number_of_dims = len(list_of_object_data[0].shape)
    paddings = [
        [[0, each_pad_length]] + [[0, 0]] * (number_of_dims - 1)
        for each_pad_length in pad_lengths]
    padded = [
        np.pad(each_sample, each_padding, mode="constant",
               constant_values=pad_value)
        for each_sample, each_padding in zip(list_of_object_data, paddings)]
    return np.stack(padded, 0)


def multiclass_non_max_suppression_np(
        object_boxes: np.ndarray, object_scores: np.ndarray,
        object_classes: np.ndarray, num_classes: int,
        score_thresh: float, iou_threshold: float,
        instance_ids: Optional[np.ndarray] = None,
        max_output_size=100000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform non max suppression of batch of objects

    Parameters
    ----------
    object_boxes
        object boxes with shape [bs, num_objects, 4] and coordinates in format
        [ymin, xmin, ymax, xmax]
    object_classes
        zero-based object classes with shape [bs, num_objects]
    object_scores
        object scores with shape [bs, num_objects]
    num_classes
        number of classes
    score_thresh
        score threshold for objects
    iou_threshold
        iou threshold
    instance_ids
        instance ids of objects with shape [bs, num_objects]
    max_output_size
        max number of bounding boxes after nms

    Returns
    -------
    object_boxes
        object boxes after nms with shape [bs, num_objects, 4] and coordinates
        in format [ymin, xmin, ymax, xmax]
    object_scores
        object scores after nms with shape [bs, num_objects]
    object_classes
        zero-based object classes after nms with shape [bs, num_objects]
    instance_ids
        instance ids of objects after nms with shape [bs, num_objects]
    """
    # pylint: disable=too-many-arguments
    # all the arguments are needed to perform filter operation
    if (iou_threshold >= 1.0 and
            not isinstance(score_thresh, (list, tuple)) and
            score_thresh <= 0.0):
        return object_boxes, object_scores, object_classes, instance_ids

    detected_boxlist = np_box_list.BoxList(object_boxes)
    num_objects = object_boxes.shape[0]
    scores = np.zeros((num_objects, num_classes + 1))
    scores[np.arange(num_objects), object_classes] = object_scores
    detected_boxlist.add_field('scores', scores)
    detected_boxlist.add_field('classes', object_classes)

    if instance_ids is not None:
        detected_boxlist.add_field('ids', instance_ids)

    if isinstance(score_thresh, (list, tuple)):
        score_thresh = type(score_thresh)([0] + list(score_thresh))
    detected_boxlist_nms = multi_class_non_max_suppression_with_extra_fields(
        detected_boxlist, score_thresh, iou_threshold,
        max_output_size=max_output_size)

    object_scores = detected_boxlist_nms.get_field('scores')
    object_classes = detected_boxlist_nms.get_field('classes').astype(np.int32)
    object_boxes = detected_boxlist_nms.get()
    if detected_boxlist_nms.has_field('ids'):
        instance_ids = detected_boxlist_nms.get_field('ids').astype(np.int32)
    else:
        instance_ids = np.ones_like(object_classes) * (-1)
    return object_boxes, object_scores, object_classes, instance_ids


@tf_varscopes_utils.with_name_scope("batch_multiclass_nms")
def batch_multiclass_non_max_suppression(
        object_boxes, object_scores,
        object_classes, num_classes,
        iou_threshold,
        score_threshold=0.0,
        max_size_per_class: Union[tf.Tensor, int] = 100,
        max_total_size: Union[tf.Tensor, int] = 200,
        parallel_iterations=16,
        is_classwise=False,
        **additional_fields
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
    """
    Perform non max suppression of batch of objects

    Parameters
    ----------
    object_boxes
        object boxes with shape [bs, num_objects, 4] and coordinates in format
        [ymin, xmin, ymax, xmax]
    object_scores
        object scores with shape [bs, num_objects]
    object_classes
        zero-based object classes with shape [bs, num_objects]
    iou_threshold
        iou threshold to suppress boxes; if is not a scalar, then it should
        have size of num_classes and will be selected for each class
    score_threshold
        score threshold for nms; if is not a scalar, then it should
        have size of num_classes and will be selected for each class
    num_classes
        number of classes
    max_size_per_class
        max number of objects per class after nms
    max_total_size
        max number of objects after nms
    parallel_iterations
        number of parallel iterations to use for map_fn
    is_classwise
        if the provided data is already in classwise format, e.g. boxes
        habe shape [bs, num_objects, num_classes, 4]
    additional_fields
        additional fields to select from for nms with each value of shape
        [bs, num_objects]

    Returns
    -------
    object_boxes
        object boxes after nms with shape [bs, num_objects, 4] and coordinates
        in format [ymin, xmin, ymax, xmax]
    object_scores
        object scores after nms with shape [bs, num_objects]
    object_classes
        zero-based object classes after nms with shape [bs, num_objects]
    num_objects
        number of objects after nms
    additional_fields
        additional fields after nms with shape
    """
    # pylint: disable=too-many-arguments
    # all the arguments are needed to perform filter operation
    # pylint: disable=too-many-locals
    # cannot reduce number of locals without more code complexity
    object_boxes = tf.cast(object_boxes, tf.float32)
    object_scores = tf.cast(object_scores, tf.float32)
    if not is_classwise:
        (object_boxes_classwise, object_scores_classwise,
         additional_fields_classwise
         ) = _reshape_scores_and_boxes_classwise(object_boxes, object_scores,
                                                 object_classes,
                                                 **additional_fields,
                                                 num_classes=num_classes)
    else:
        (object_boxes_classwise, object_scores_classwise,
         additional_fields_classwise) = (object_boxes, object_scores,
                                         additional_fields)

    current_number_of_boxes = tf.shape(object_boxes)[1]
    max_total_size = tf.minimum(max_total_size,
                                current_number_of_boxes)

    additional_fields_keys = list(additional_fields)

    def _singe_sample_multiclass_nms(inputs):
        boxes_classwise = inputs[0]
        scores_classwise = inputs[1]
        additional_fields_classwise_sample = dict(
            zip(additional_fields_keys, inputs[2:]))
        (boxes_nms, scores_nms, classes_nms, num_objects_nms,
         additional_fields_nms_) = _multiclass_non_max_suppression(
             boxes_classwise, scores_classwise,
             iou_threshold, num_classes,
             score_threshold=score_threshold,
             max_total_size=max_total_size,
             max_size_per_class=max_size_per_class,
             **additional_fields_classwise_sample)
        additional_fields_nms_values_ = [
            additional_fields_nms_[each_key]
            for each_key in additional_fields_keys]
        return (boxes_nms, scores_nms, classes_nms, num_objects_nms,
                *additional_fields_nms_values_)

    additional_fields_values_classwise = [
        additional_fields_classwise[each_key]
        for each_key in additional_fields_keys]
    additional_fields_dtypes = [
        each_value.dtype for each_value in additional_fields_values_classwise]

    inputs = ([object_boxes_classwise, object_scores_classwise]
              + additional_fields_values_classwise)
    dtypes = tuple([tf.float32, tf.float32, tf.int32, tf.int32]
                   + additional_fields_dtypes)
    (object_boxes_nms, object_scores_nms, object_classes_nms, num_objects_nms,
     *additional_values_nms
     ) = tf.map_fn(_singe_sample_multiclass_nms, inputs,
                   dtype=dtypes,
                   parallel_iterations=parallel_iterations
                   )
    additional_fields_nms = dict(zip(additional_fields_keys,
                                     additional_values_nms))

    return (object_boxes_nms, object_scores_nms,
            object_classes_nms, num_objects_nms,
            additional_fields_nms)


def get_true_image_shapes(images: tf.Tensor) -> tf.Tensor:
    """
    Get the true image shapes in format which is needed by object_detection API
    from images. The same image shape will be replicated over batch size

    Parameters
    ----------
    images
        images

    Returns
    -------
    true_image_shapes
        tensor with shape [bs, 3] with image shape replicated in format
        height, width, number_of_channels
    """
    image_shape = tf.shape(images)
    batch_size = tf.shape(images)[0]
    true_image_shapes = tf.tile(tf.expand_dims(image_shape[1:], 0),
                                [batch_size, 1])
    return true_image_shapes


def crop_keypoints_to_boxes(keypoints: tf.Tensor, boxes: tf.Tensor
                            ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Crops keypoints to boxes, e.g. if the keypoint is outside of the box, it
    will be removed, e.g. zeroed

    Both keypoints and boxes must be in the same coordinate frame, e.g. either
    both normalized or both absolute

    Parameters
    ----------
    keypoints
        keypoints with shape [batch, num_objects, num_keypoints, 2] and
        coordinates in format [y, x]
    boxes
        boxes with shape [batch, num_objects, 4] and coordinates in format
        [ymin, xmin, ymax, xmax]

    Returns
    -------
    filtered_keypoints
        keypoints inside of boxes with shape
        [batch, num_objects, num_keypoints, 2] and coordinates in format
        [y, x]
    valid_keypoints_mask
        tensor of shape [batch, num_objects, num_keypoints] with 1 if the
        keypoint is valid and 0 otherwise
    """
    boxes_yx_min = tf.broadcast_to(boxes[..., tf.newaxis, :2],
                                   tf.shape(keypoints))
    boxes_yx_max = tf.broadcast_to(boxes[..., tf.newaxis, 2:],
                                   tf.shape(keypoints))
    keypoints_mask = tf.reduce_all(tf.logical_and(
        tf.greater_equal(keypoints, boxes_yx_min),
        tf.less_equal(keypoints, boxes_yx_max),
    ), -1, keepdims=True)
    keypoints_in_boxes = tf.where(
        tf.broadcast_to(keypoints_mask, tf.shape(keypoints)),
        keypoints,
        tf.zeros_like(keypoints))
    return keypoints_in_boxes, tf.squeeze(keypoints_mask)


def encode_keypoints_to_boxes(keypoints: tf.Tensor, boxes: tf.Tensor,
                              ) -> tf.Tensor:
    """
    Encode keypoint coordinates to box frame coordinates

    Parameters
    ----------
    keypoints
        keypoints with shape [batch, num_objects, num_keypoints, 2] and
        coordinates in format [y, x]
    boxes
        boxes with shape [batch, num_objects, 4] and coordinates in format
        [ymin, xmin, ymax, xmax]

    Returns
    -------
    keypoints_encoded
        encoded keypoints in relative to bounding boxes coordinates
    """
    boxes_yx_min = tf.broadcast_to(boxes[..., tf.newaxis, :2],
                                   tf.shape(keypoints))
    boxes_yx_max = tf.broadcast_to(boxes[..., tf.newaxis, 2:],
                                   tf.shape(keypoints))
    keypoints_encoded = tf.where(
        boxes_yx_max > boxes_yx_min,
        (keypoints - boxes_yx_min) / (boxes_yx_max - boxes_yx_min),
        tf.zeros_like(keypoints))
    keypoints_encoded = tf.where(
        tf.logical_and(tf.greater_equal(keypoints_encoded, 0),
                       tf.less_equal(keypoints_encoded, 1)),
        keypoints_encoded,
        tf.zeros_like(keypoints_encoded))
    return keypoints_encoded


def decode_keypoints_from_boxes(keypoints: tf.Tensor, boxes: tf.Tensor,
                                ) -> tf.Tensor:
    """
    Decode relative to box keypoint coordinates to absolute ones

    Parameters
    ----------
    keypoints
        keypoints with shape [batch, num_objects, num_keypoints, 2] and
        coordinates in format [y, x] relative to boxes
    boxes
        boxes with shape [batch, num_objects, 4] and coordinates in format
        [ymin, xmin, ymax, xmax]

    Returns
    -------
    keypoints_decoded
        decoded keypoints in relative to image coordinates
    """
    boxes_yx_min = boxes[..., tf.newaxis, :2]
    boxes_yx_max = boxes[..., tf.newaxis, 2:]
    boxes_height_width = boxes_yx_max - boxes_yx_min

    keypoints_decoded = keypoints * boxes_height_width + boxes_yx_min
    return keypoints_decoded


def create_keypoints_heatmaps(keypoints, heatmaps_image_shape,
                              keypoints_masks: Optional[tf.Tensor] = None,
                              keypoints_kernel_size: int = 3):
    """
    Create heatmaps images for keypoints. Will apply gaussian kernel and rescale
    to have max value = 1

    Parameters
    ----------
    keypoints
        encoded keypoints with shape [None, 2] and normalized coordinates
        in range [0, 1]
    heatmaps_image_shape
        output shape of heatmaps, [width, height]
    keypoints_masks
        keypoints masks with shape [None] with 1 indicating that keypoint
        is active and 0 - is not active
    keypoints_kernel_size
        kernel size to apply on keypoints for gaussian smoothing

    Returns
    -------
    keypoints_heatmaps
        keypoints heatmaps
    """
    (keypoint_centers_expand, keypoint_kernel_coordinates
     ) = _get_keypoint_coordinates(keypoints, heatmaps_image_shape,
                                   keypoints_kernel_size)
    gaussian_values_unnormalized = _gaussian_unnormalized(
        keypoint_kernel_coordinates,
        keypoint_centers_expand,
        keypoints_kernel_size // 2 or 1)
    gaussian_values_flat = tf.reshape(gaussian_values_unnormalized, [-1])

    batch_dim_range = tf.range(
        tf.shape(keypoints, out_type=tf.int64)[0], dtype=tf.int64)
    batch_dim_to_add = broadcast_with_expand_to(
        batch_dim_range, keypoint_kernel_coordinates[..., 0])[..., tf.newaxis]

    sparse_indices = tf.concat(
        [tf.reshape(batch_dim_to_add, [-1, 1]),
         tf.cast(tf.reshape(keypoint_kernel_coordinates, [-1, 2]), tf.int64)],
        -1)
    heatmaps_full_shape = tf.concat(
        [[tf.shape(keypoints)[0]],
         heatmaps_image_shape], 0)
    heatmaps_sparse = tf.SparseTensor(
        indices=sparse_indices,
        values=gaussian_values_flat,
        dense_shape=tf.cast(heatmaps_full_shape, tf.int64))
    heatmaps_sparse = tf.sparse_reorder(heatmaps_sparse)
    heatmaps = tf.sparse_tensor_to_dense(
        heatmaps_sparse, validate_indices=False)
    if keypoints_masks is not None:
        keypoints_masks_expanded = broadcast_with_expand_to(
            keypoints_masks, heatmaps)
        keypoints_masks_expanded = tf.cast(keypoints_masks_expanded,
                                           heatmaps.dtype)
        heatmaps = heatmaps * keypoints_masks_expanded
    return heatmaps


def extract_keypoints_from_heatmaps(keypoints_heatmaps,
                                    smooth_size: Union[tf.Tensor, int] = 3,
                                    normalize_smoothing_kernel: bool = False,
                                    ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Extract keypoints from its heatmaps

    First will apply gaussian filter on the keypoints_heatmaps and then select
    the max value and max indices from it.

    Parameters
    ----------
    keypoints_heatmaps
        heytmaps with keypoints with shape [batch, height, width, num_keypoints]
        and tfloat32 dtype
    smooth_size
        kernel size to apply for gaussian filter
    normalize_smoothing_kernel
        if the smoothing kernel should be normalized, e.g. sum of the kernel = 1

    Returns
    -------
    keypoints
        normalized to heatmaps dimensions coordinates of keypoints;
        [batch, num_keypoints, 2] with format [y, x], tf.float32
    keypoints_scores
        scores for keypoints, which are taken on heatmaps after gaussian filter
        application; so the values may be not normalized;
        shape [batch, num_keypoints], tf.float32
    """

    def _smooth_heatmaps():
        gaussian_kernel = _create_gaussian_kernel(
            tf.maximum(smooth_size // 2, 1))
        gaussian_kernel = tf.tile(gaussian_kernel[..., tf.newaxis, tf.newaxis],
                                  [1, 1, keypoints_heatmaps.shape[-1], 1])
        if normalize_smoothing_kernel:
            gaussian_kernel = gaussian_kernel / tf.reduce_sum(gaussian_kernel)
        heatmaps_avg = tf.nn.depthwise_conv2d(
            keypoints_heatmaps,
            gaussian_kernel,
            [1] * 4, padding="SAME")
        return heatmaps_avg

    heatmaps_avg = tf.cond(tf.greater(smooth_size, 1),
                           _smooth_heatmaps,
                           lambda: keypoints_heatmaps)

    heatmaps_avg_flat = tf.reshape(
        heatmaps_avg,
        tf.concat([tf.shape(heatmaps_avg)[:1], [-1],
                   tf.shape(heatmaps_avg)[-1:]], 0))
    keypoints_scores, indices_flat = tf.nn.top_k(
        tf.transpose(heatmaps_avg_flat, [0, 2, 1]), 1)
    keypoints_scores = tf.squeeze(keypoints_scores, -1)
    indices_flat = tf.squeeze(indices_flat, -1)
    heatmaps_y_shape = tf.shape(heatmaps_avg)[2]
    indices_x = tf.floor_div(indices_flat, heatmaps_y_shape)
    indices_y = tf.mod(indices_flat, heatmaps_y_shape)
    keypoints_indices = tf.stack([indices_x, indices_y], -1)
    image_size_divider = (tf.cast(tf.shape(heatmaps_avg)[1:-1], tf.float32
                                  )[tf.newaxis, tf.newaxis]
                          - 1)
    keypoints = (tf.cast(keypoints_indices, tf.float32)
                 / image_size_divider)
    return keypoints, keypoints_scores


def decode_instance_masks_to_image(object_instance_masks: tf.Tensor,
                                   object_boxes: tf.Tensor,
                                   image_sizes: tf.Tensor,
                                   ) -> tf.Tensor:
    """
    Decode image masks from relative to boxes to image frame

    Parameters
    ----------
    object_instance_masks
        object instance masks, [bs, num_objects, mask_height, mask_width]
    object_boxes
        object boxes relative to image, [bs, num_objects, 4]
    image_sizes
        image sizes [bs, 2], where each item is [image_height, image_width]

    Returns
    -------
    object_instance_masks_on_image
        instance masks on the image;
        [bs, num_objects, image_height, image_width]; tf.float32
    """

    def _reframe_masks_to_image(args):
        single_masks, single_boxes, image_shape = args
        single_masks_reframed = (
            od_utils_ops.reframe_box_masks_to_image_masks(
                single_masks, single_boxes,
                image_shape[0], image_shape[1]))
        return single_masks_reframed

    object_instance_masks_on_image = tf.map_fn(
        _reframe_masks_to_image,
        elems=[object_instance_masks, object_boxes, image_sizes],
        dtype=tf.float32,
    )
    object_instance_masks_on_image = tf.where(
        tf.is_nan(object_instance_masks_on_image),
        tf.zeros_like(object_instance_masks_on_image),
        object_instance_masks_on_image,
    )
    return object_instance_masks_on_image


_TENSOR_OR_ARRAY = Union[tf.Tensor, np.array]  # pylint: disable=invalid-name


def mask_inputs_to_classes(
        object_classes: _TENSOR_OR_ARRAY,
        other_inputs: Dict[str, Union[_TENSOR_OR_ARRAY, dict]],
        classes_mask: _TENSOR_OR_ARRAY,
        mask_single_input_to_classes_fn: Callable[
            [_TENSOR_OR_ARRAY, _TENSOR_OR_ARRAY], _TENSOR_OR_ARRAY]
) -> Dict[str, Union[_TENSOR_OR_ARRAY, dict]]:
    """
    Mask inputs according to classes. This method is agnostic to tensorflow /
    numpy engines, but the injected methods create_mask_fn and
    mask_single_input_to_classes_fn are not, so must be used with respect
    to the engine.

    Parameters
    ----------
    object_classes
        object classes with shape [..., num_objects], where ... is the first
        dimensions and can be omitted
    other_inputs
        other inputs to mask with class mask
    classes_mask
        class mask whith shape [first_dims] and bool dtype
    mask_single_input_to_classes_fn
        method to select the input according to class_id mask; should take
        input and classes_mask as input and output the masked array

    Returns
    -------
    result
        result with masked inputs including also the num_objects key
        representing number of objects after masking; object_classes is also
        inside of this result
    """
    dynamic_inputs_flat = nest_utils.flatten_nested_struct(other_inputs)
    all_inputs_flat = {
        ObjectDataFields.object_classes: object_classes,
    }
    all_inputs_flat.update(dynamic_inputs_flat)
    result_flat = {
        k: mask_single_input_to_classes_fn(v, classes_mask)
        for k, v in all_inputs_flat.items()
    }
    result = nest_utils.unflatten_dict_to_nested(result_flat)
    return result


def create_classes_mask(object_classes: tf.Tensor, class_ids_to_mask: List[int],
                        num_objects: Optional[tf.Tensor] = None,
                        ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Create classes_mask and calculate new number of objects

    Parameters
    ----------
    object_classes
        object classes to create mask over;
        [..., num_objects], where ... is the first dimensions and can be omitted
    class_ids_to_mask
        class ids to mask; all other class ids will be masked out
    num_objects
        number of objects; needed when masking 0 class

    Returns
    -------
    classes_mask
        boolean mask with same shape as object_classes
    num_objects
        number of objects after mask application; has same dimension as first
        dimension from object_classes except object dimension; tf.int32

    Raises
    ------
    ValueError
        if 0 is in class_ids_to_mask and num_objects was not provided
    """
    if 0 in class_ids_to_mask and num_objects is None:
        msg = "Provide num_object_detections to select 0 class"
        raise ValueError(msg)

    classes_masks = []

    for each_class_to_select in class_ids_to_mask:
        class_mask = tf.equal(object_classes, each_class_to_select)
        classes_masks.append(class_mask)
    classes_masks = tf.stack(classes_masks, -1)
    classes_mask = tf.reduce_sum(tf.cast(classes_masks, tf.uint8), -1)
    classes_mask = tf.greater(classes_mask, 0)
    if num_objects is not None:
        classes_mask_from_num_objects = tf.sequence_mask(
            num_objects, tf.shape(object_classes)[-1])
        classes_mask = tf.logical_and(classes_mask,
                                      classes_mask_from_num_objects)
    num_objects = tf.reduce_sum(tf.cast(classes_mask, tf.int32), -1)
    return classes_mask, num_objects


def filter_objects_over_mask(mask_detections,
                             object_boxes,
                             *additional_args_inputs,
                             **additional_inputs):
    """
    Filter the masked objects out and rearrange the objects in case the middle
    objects are filtered out

    Parameters
    ----------
    mask_detections
        batch of bool masks over objects with shape
        [batch_size, max_num_objects]
    object_boxes
        object boxes in format [ymin, xmin, ymax, xmax]
    additional_args_inputs
        additional arg inputs, which will be also filtered out and rearranged;
        all should have shape [batch_size, max_num_objects, ...]
    additional_inputs
        additional kwargs inputs, which will be also filtered out and
        rearranged;
        all values should have shape [batch_size, max_num_objects, ...]
    Returns
    -------
    object_boxes_filtered
        filtered object boxes
    num_objects_filtered
        filtered number of objects
    *additional_args_inputs_filtered
        filtered additional args inputs
    additional_inputs_filtered
        filtered additional kwargs inputs
    """
    filtered_indices = _get_indices_from_mask(mask_detections)
    object_boxes_filtered = tf.batch_gather(object_boxes, filtered_indices)
    num_objects_filtered = tf.reduce_sum(tf.cast(mask_detections, tf.int32), -1)
    additional_inputs_filtered = {
        each_key: tf.batch_gather(each_item, filtered_indices)
        for each_key, each_item in additional_inputs.items()
    }
    additional_args_inputs_filtered = [
        tf.batch_gather(each_item, filtered_indices)
        for each_item in additional_args_inputs
    ]

    (object_boxes_filtered, num_objects_filtered,
     *additional_args_inputs_filtered, additional_inputs_filtered
     ) = zero_out_invalid_pad_objects(object_boxes_filtered,
                                      *additional_args_inputs_filtered,
                                      **additional_inputs_filtered,
                                      num_objects=num_objects_filtered)
    return (object_boxes_filtered, num_objects_filtered,
            *additional_args_inputs_filtered,
            additional_inputs_filtered)


def offset_object_classes(object_classes: tf.Tensor, *,
                          object_boxes: Optional[tf.Tensor] = None,
                          num_objects: Optional[tf.Tensor] = None
                          ) -> tf.Tensor:
    """
    Offset the classes on valid objects. Valid objects are specified either by
    num_objects or by object boxes; if object_boxes were provided, then only
    the classes on the valid boxes will be offset

    Parameters
    ----------
    object_classes
        batch object classes to offset
    object_boxes
        object boxes
    num_objects
        number of objects

    Returns
    -------
    object_classes_offset
        original classes +1 on valid objects
    """
    assert object_boxes is not None or num_objects is not None, (
        "one of num_objects or object_boxes must be provided!"
    )
    if num_objects is None:
        _, valid_boxes_mask = _get_num_objects_from_boxes(object_boxes)
    else:
        max_num_objects = tf.shape(object_classes)[-1]
        valid_boxes_mask = get_objects_mask_from_num_objects(
            max_num_objects, num_objects)
    object_classes_offset = object_classes + tf.cast(
        valid_boxes_mask, object_classes.dtype)
    return object_classes_offset


def zero_out_invalid_pad_objects(
        object_boxes: tf.Tensor,
        *additional_args_inputs,
        num_objects: Optional[tf.Tensor] = None,
        **additional_inputs):
    """
    Filter the masked objects out and rearrange the objects in case the middle
    objects are filtered out

    Parameters
    ----------
    object_boxes
        object boxes in format [ymin, xmin, ymax, xmax]
    num_objects
        number of objects; if not provided, it will be inferred from the valid
        object boxes
    additional_args_inputs
        additional arg inputs, which will be also filtered out and rearranged;
        all should have shape [batch_size, max_num_objects, ...]
    additional_inputs
        additional kwargs inputs, which will be also filtered out and
        rearranged;
        all values should have shape [batch_size, max_num_objects, ...]
    Returns
    -------
    object_boxes_filtered
        filtered object boxes
    num_objects_filtered
        filtered number of objects
    *additional_args_inputs_filtered
        filtered additional args inputs
    additional_inputs_filtered
        filtered additional kwargs inputs
    """
    max_num_detections = tf.shape(object_boxes)[-2]
    if num_objects is None:
        num_objects, _ = _get_num_objects_from_boxes(object_boxes)
    detections_mask = get_objects_mask_from_num_objects(
        max_num_detections, num_objects)
    detections_mask_boxes = broadcast_with_expand_to(
        detections_mask, object_boxes)

    object_boxes = tf.where(
        detections_mask_boxes, object_boxes,
        tf.zeros_like(object_boxes))
    additional_inputs_processed = {
        each_key: tf.where(
            broadcast_with_expand_to(detections_mask, each_item),
            each_item,
            tf.zeros_like(each_item))
        for each_key, each_item in additional_inputs.items()
    }
    additional_args_processed = [
        tf.where(
            broadcast_with_expand_to(detections_mask, each_item),
            each_item,
            tf.zeros_like(each_item))
        for each_item in additional_args_inputs
    ]
    return (object_boxes, num_objects,
            *additional_args_processed, additional_inputs_processed)


def get_objects_mask_from_num_objects(max_num_objects: tf.Tensor,
                                      num_objects: tf.Tensor) -> tf.Tensor:
    """
    Create object mask from number of objects for each sample in batch

    Parameters
    ----------
    max_num_objects
        max number of objects in the batch
    num_objects
        number of the objects for each sample with size [batch]

    Returns
    -------
    objects_mask
        objects mask with True for active object and False for a padding
    """
    num_object_detections = tf.cast(num_objects, tf.int32)
    detections_mask = tf.sequence_mask(
        num_object_detections, max_num_objects)
    return detections_mask


def get_valid_objects_boxes_mask(object_boxes: tf.Tensor) -> tf.Tensor:
    """
    Get mask for valid object boxes, e.g. True if width and height are greater
    than 0

    Parameters
    ----------
    object_boxes
        object boxes in format [ymin, xmin, ymax, xmax]

    Returns
    -------
    valid_boxes_mask
        mask with valid objects
    """
    ymin, xmin, ymax, xmax = tf.split(object_boxes, 4, -1)
    widths = xmax - xmin
    heights = ymax - ymin
    valid_boxes_mask = tf.logical_and(tf.greater(widths, 0),
                                      tf.greater(heights, 0))[..., 0]
    return valid_boxes_mask


def _get_num_objects_from_boxes(object_boxes: tf.Tensor
                                ) -> Tuple[tf.Tensor, tf.Tensor]:
    valid_objects_mask = get_valid_objects_boxes_mask(object_boxes)
    valid_objects_mask_filled = tf.greater(
        tf.cumsum(tf.cast(valid_objects_mask, tf.int32), axis=-1, reverse=True),
        0)
    num_objects = tf.reduce_sum(tf.cast(valid_objects_mask_filled, tf.int32),
                                -1)
    return num_objects, valid_objects_mask


def _create_gaussian_kernel(sigma):
    kernel_range = tf.range(-sigma + 1, sigma)
    mesh_grid_kernel = tf.stack(tf.meshgrid(kernel_range,
                                            kernel_range), -1)
    gaussian_kernel = _gaussian_unnormalized(
        tf.cast(mesh_grid_kernel, tf.float32),
        [0., 0.], sigma=tf.cast(sigma, tf.float32))
    return gaussian_kernel


def _get_keypoint_coordinates(keypoints, heatmaps_image_shape,
                              keypoints_kernel_size):
    image_size_multiplier = tf.cast(
        heatmaps_image_shape[tf.newaxis, :], tf.float32)
    keypoint_centers = tf.ceil(
        (keypoints * image_size_multiplier)) - 1
    keypoint_centers = tf.clip_by_value(
        keypoint_centers, 0.0,
        tf.cast(heatmaps_image_shape, tf.float32) - 1)
    keypoint_centers_expand = keypoint_centers[..., tf.newaxis, tf.newaxis, :]
    kernel_range = tf.range(-keypoints_kernel_size + 1, keypoints_kernel_size)
    mesh_grid_kernel = tf.stack(tf.meshgrid(kernel_range,
                                            kernel_range), -1)
    mesh_grid_kernel = mesh_grid_kernel[tf.newaxis]
    keypoint_kernel_coordinates = (keypoint_centers_expand
                                   + tf.cast(mesh_grid_kernel, tf.float32))
    keypoint_kernel_coordinates = tf.clip_by_value(
        keypoint_kernel_coordinates, 0.0,
        tf.cast(heatmaps_image_shape, tf.float32) - 1)
    return keypoint_centers_expand, keypoint_kernel_coordinates


def _gaussian_unnormalized(coord, centers, sigma):
    centers = tf.cast(centers, tf.float32)
    exp = tf.exp(-tf.reduce_sum((coord - centers) ** 2, -1) / (2 * sigma ** 2))
    return exp


@tf_varscopes_utils.with_name_scope("multiclass_nms")
def _multiclass_non_max_suppression(
        boxes_classwise, scores_classwise,
        iou_threshold, num_classes, score_threshold=0.0,
        max_size_per_class=100,
        max_total_size=200,
        **additional_fields_classwise
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
    # pylint: disable=too-many-locals
    # cannot reduce number of locals without more code complexity
    def _select_threshold_for_class(thresholds, class_index):
        if isinstance(thresholds, tf.Tensor):
            if len(thresholds.shape) >= 1:
                return thresholds[class_index]
            return thresholds
        if len(np.shape(thresholds)) >= 1:
            return thresholds[class_index]
        return thresholds

    nms_box_lists = []
    for each_class_index in range(num_classes):
        iou_threshold_for_class = _select_threshold_for_class(
            iou_threshold, each_class_index)
        score_threshold_for_class = _select_threshold_for_class(
            score_threshold, each_class_index)
        nms_box_list_class = _non_max_suppression_single_class(
            boxes_classwise, scores_classwise, additional_fields_classwise,
            each_class_index, iou_threshold_for_class,
            score_threshold_for_class,
            max_size_per_class)
        nms_box_lists.append(nms_box_list_class)

    nms_boxes_all_classes = box_list_ops.concatenate(nms_box_lists)
    nms_boxes_all_classes = box_list_ops.sort_by_field(
        nms_boxes_all_classes, box_standard_fields.BoxListFields.scores)
    num_objects_nms = nms_boxes_all_classes.num_boxes()
    nms_boxes_all_classes = box_list_ops.pad_or_clip_box_list(
        nms_boxes_all_classes, max_total_size)

    boxes_nms = nms_boxes_all_classes.get()
    classes_nms = nms_boxes_all_classes.get_field(
        box_standard_fields.BoxListFields.classes)
    scores_nms = nms_boxes_all_classes.get_field(
        box_standard_fields.BoxListFields.scores)
    additional_fields_nms = {
        each_field: nms_boxes_all_classes.get_field(each_field)
        for each_field in additional_fields_classwise}
    return (boxes_nms, scores_nms, classes_nms, num_objects_nms,
            additional_fields_nms)


def _non_max_suppression_single_class(boxes_classwise, scores_classwise,
                                      additional_fields_classwise,
                                      class_index, iou_threshold,
                                      score_threshold,
                                      max_size_per_class):
    boxes_for_class = boxes_classwise[..., class_index, :]
    scores_for_class = scores_classwise[..., class_index]
    classes = tf.ones_like(scores_for_class, tf.int32) * class_index
    box_list_class = box_list.BoxList(boxes_for_class)
    box_list_class.add_field(
        box_standard_fields.BoxListFields.classes, classes)
    box_list_class.add_field(
        box_standard_fields.BoxListFields.scores, scores_for_class)
    for each_additional_key, each_additional_field in (
            additional_fields_classwise.items()):
        box_list_class.add_field(
            each_additional_key,
            each_additional_field[:, class_index, ...])
    selected_indices_class = tf.image.non_max_suppression(
        boxes_for_class, scores_for_class, max_size_per_class,
        iou_threshold=iou_threshold, score_threshold=score_threshold)
    nms_box_list_class = box_list_ops.gather(box_list_class,
                                             selected_indices_class)

    return nms_box_list_class


def _reshape_scores_and_boxes_classwise(
        object_boxes,
        object_scores,
        object_classes,
        num_classes=1,
        **additional_fields
) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
    if num_classes == 1:
        object_boxes = tf.expand_dims(object_boxes, -2)
        object_scores = tf.expand_dims(object_scores, -1)
        additional_fields_classwise = {
            each_field: tf.expand_dims(each_value, 2)
            for each_field, each_value in additional_fields.items()
        }
        return (object_boxes, object_scores,
                additional_fields_classwise)

    classes_mask = tf.one_hot(object_classes, num_classes)
    classes_mask_for_boxes = _tile_class_mask_to_tensor(
        classes_mask, object_boxes)
    object_boxes_classwise = _gather_by_mask(
        classes_mask_for_boxes, object_boxes)
    object_scores_classwise = _gather_by_mask(
        classes_mask, object_scores)
    additional_fields_classwise = {
        each_field: _gather_by_mask(
            _tile_class_mask_to_tensor(classes_mask, each_value),
            each_value)
        for each_field, each_value in additional_fields.items()}

    return (object_boxes_classwise, object_scores_classwise,
            additional_fields_classwise)


def _tile_class_mask_to_tensor(classes_mask, tensor):
    mask_ndims = len(classes_mask.get_shape().as_list())
    if mask_ndims == len(tensor.shape) + 1:
        return classes_mask
    tensor_last_dims = tf.shape(tensor)[2:]
    new_dims_len = len(tensor.shape.as_list()) - 2
    mask_slices = ([slice(None)] * mask_ndims
                   + [tf.newaxis] * new_dims_len)
    tiles = tf.concat([[1] * mask_ndims, tensor_last_dims], 0)
    classes_mask_tiled = tf.tile(classes_mask[mask_slices], tiles)
    return classes_mask_tiled


def _gather_by_mask(mask, values):
    values_gathered_sparse = tf.SparseTensor(
        tf.where(mask),
        tf.reshape(values, [-1]),
        dense_shape=tf.shape(mask, out_type=tf.int64))
    values_gathered_dense = tf.sparse_tensor_to_dense(values_gathered_sparse)
    return values_gathered_dense


def _get_interval_mask(value, min_value, max_value):
    greater_equal_mask = tf.greater_equal(value, min_value)
    less_equal_mask = tf.cond(tf.greater(max_value, 0),
                              lambda: tf.less_equal(value, max_value),
                              lambda: tf.ones_like(value, tf.bool))
    mask = tf.logical_and(greater_equal_mask, less_equal_mask)
    return mask


def _get_shape_multiplier(image_size, true_image_shapes, bboxes):
    if true_image_shapes is not None:
        if len(bboxes.shape) != 3:
            raise ValueError("bboxes must be a batch if you want to use"
                             "true_image_shapes!")
        multiplier = tf.tile(
            tf.cast(true_image_shapes[:, :2], tf.float32)[:, tf.newaxis, :],
            [1, 1, 2])

    else:
        height, width = image_size[0], image_size[1]
        multiplier = tf.cast(tf.stack([height, width, height, width]),
                             tf.float32)
        multiplier = tf.reshape(multiplier, [1, 4])
        if len(bboxes.shape) == 3:
            multiplier = multiplier[tf.newaxis, ...]
    return multiplier


def _get_indices_from_mask(mask: tf.Tensor) -> tf.Tensor:
    indices = tf.contrib.framework.argsort(
        tf.cast(tf.logical_not(mask), tf.float32), -1)
    return indices
