# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Postprocessors for object detection
"""

from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import nucleus7 as nc7
import tensorflow as tf

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.data_fields.object_detection import DetectionDataFields
from ncgenes7.data_fields.object_detection import ObjectDataFields
from ncgenes7.utils import object_detection_utils as od_utils
from ncgenes7.utils.general_utils import broadcast_with_expand_to
from ncgenes7.utils.general_utils import check_if_dynamic_keys_flat


class DetectionsPostprocessor(nc7.model.ModelPostProcessor):
    """
    Simply forwards inputs detections to outputs and mask with zeros all
    detections outside of num_object_detections and add 1 to classes
    if needed

    It also has a dynamic keys option, so all the additional keys that will
    be provided to its postprocessor must have first dimension after batch
    to be also object dimension, e.g. `[bs, max_num_detections, *]`.
    These additional keys are not allowed to be nested, so only 1 level dicts
    are allowed.

    Parameters
    ----------
    offset_detection_classes
        if the classes of detection should be increased by 1, e.g. should be
        set to True for second stage faster rcnn, as detection_classes there
        are 0-based

    Attributes
    ----------
    incoming_keys
        * detection_object_boxes : detection boxes in
          format [ymin, xmin, ymax, xmax], [bs, max_num_detections, 4], float32
        * detection_object_scores : detection scores, [bs, max_num_detections],
          float32
        * detection_object_classes : (optional) detection classes;
          if not defined assumed to be 1 on all active detections,
          [bs, max_num_detections], int32
        * num_object_detections : number of detections, [bs], int32
        * detection_object_instance_ids : (optional) detection instance ids;
          if not defined assumed to be 0 on all active detections,
          [bs, max_num_detections], int32
    generated_keys
        * detection_object_boxes : detection boxes in
          format [ymin, xmin, ymax, xmax], [bs, max_num_detections, 4], float32
        * detection_object_scores : detection scores ,[bs, max_num_detections],
          float32,
        * detection_object_classes : detection classes,
          [bs, max_num_detections], int32
        * num_object_detections : number of detections, [bs], int32
        * detection_object_instance_ids : (optional) detection instance ids;
          if not defined assumed to be 0 on all active detections,
          [bs, max_num_detections], int32
    """
    dynamic_incoming_keys = True
    dynamic_generated_keys = True

    incoming_keys = [
        DetectionDataFields.detection_object_boxes,
        DetectionDataFields.detection_object_scores,
        DetectionDataFields.num_object_detections,
        "_" + DetectionDataFields.detection_object_classes,
        "_" + DetectionDataFields.detection_object_instance_ids,
    ]
    generated_keys = [
        DetectionDataFields.detection_object_boxes,
        DetectionDataFields.detection_object_scores,
        DetectionDataFields.num_object_detections,
        DetectionDataFields.detection_object_classes,
        DetectionDataFields.detection_object_instance_ids,
    ]

    def __init__(self, *,
                 offset_detection_classes=False,
                 **postprocessor_kwargs):
        super().__init__(**postprocessor_kwargs)
        self.offset_detection_classes = offset_detection_classes

    @check_if_dynamic_keys_flat
    def process(self, *,
                detection_object_boxes,
                detection_object_scores,
                num_object_detections,
                detection_object_classes=None,
                detection_object_instance_ids=None,
                **dynamic_inputs):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        offset_classes = self.offset_detection_classes
        if detection_object_classes is None:
            offset_classes = True
        detection_object_classes = _maybe_init_classes(
            detection_object_classes, detection_object_scores)
        detection_object_instance_ids = _maybe_init_instance_ids(
            detection_object_classes, detection_object_instance_ids)
        (detection_object_boxes, num_object_detections,
         detection_object_scores, detection_object_classes,
         detection_object_instance_ids, dynamic_outputs
         ) = od_utils.zero_out_invalid_pad_objects(
             detection_object_boxes,
             detection_object_scores,
             detection_object_classes,
             detection_object_instance_ids,
             **dynamic_inputs,
             num_objects=num_object_detections)
        if offset_classes:
            detection_object_classes = od_utils.offset_object_classes(
                detection_object_classes, num_objects=num_object_detections
            )
        result = {
            DetectionDataFields.detection_object_boxes: detection_object_boxes,
            DetectionDataFields.detection_object_scores:
                detection_object_scores,
            DetectionDataFields.num_object_detections: num_object_detections,
            DetectionDataFields.detection_object_classes:
                detection_object_classes,
            DetectionDataFields.detection_object_instance_ids:
                detection_object_instance_ids,
        }
        result.update(dynamic_outputs)
        return result


# pylint: disable=too-many-instance-attributes
# attributes cannot be combined or extracted further
class DetectionsFilterPostprocessor(nc7.model.ModelPostProcessor):
    """
    Postprocessor which can filter detections according to scores, min and
    max width

    All of the parameters can be controlled during inference too

    It also has a dynamic keys option, so all the additional keys that will
    be provided to its postprocessor must have first dimension after batch
    to be also object dimension, e.g. `[bs, max_num_detections, *]`.
    These additional keys are not allowed to be nested, so only 1 level dicts
    are allowed.

    Parameters
    ----------
    num_classes
        number of classes; if not provided, assumes that only 1 class
    min_object_width
        objects with width less than this value will be removed;
        can be classwise e.g. list of values; also during inference, it is
        possible to provide either 1 value for all classes or list of values
        for each class
    max_object_width
        objects with width greater than this value will be removed;
        can be classwise e.g. list of values; also during inference, it is
        possible to provide either 1 value for all classes or list of values
        for each class
    min_object_height
        objects with height less than this value will be removed;
        can be classwise e.g. list of values; also during inference, it is
        possible to provide either 1 value for all classes or list of values
        for each class
    max_object_height
        objects with height greater than this value will be removed;
        can be classwise e.g. list of values; also during inference, it is
        possible to provide either 1 value for all classes or list of values
        for each class
    score_threshold
        objects with score less than this value will be removed;
        can be classwise e.g. list of values; also during inference, it is
        possible to provide either 1 value for all classes or list of values
        for each class
    reorder_filtered
        if the objects should be reordered after filtering out, e.g. 0 objects
        will be treated as paddings and set to the end of object lists and also
        the number of objects will be recalculated. If not specified, then just
        all the inputs which filtered out are zeroed.
    classes_to_select
        which classes to select; if specified, then the classes only with
        this ids will be selected

    Attributes
    ----------
    incoming_keys
        * detection_object_boxes : detection boxes in
          format [ymin, xmin, ymax, xmax], [bs, max_num_detections, 4], float32
        * detection_object_scores : detection scores, [bs, max_num_detections],
          float32
        * detection_object_classes : (optional) detection classes;
          if not defined assumed to be 1 on all active detections,
          [bs, max_num_detections], int32
        * num_object_detections : number of detections, [bs], int32
        * detection_object_instance_ids : (optional) detection instance ids;
          if not defined assumed to be 0 on all active detections,
          [bs, max_num_detections], int32
    generated_keys
        * detection_object_boxes : detection boxes in
          format [ymin, xmin, ymax, xmax], [bs, max_num_detections, 4], float32
        * detection_object_scores : detection scores ,[bs, max_num_detections],
          float32,
        * detection_object_classes : detection classes,
          [bs, max_num_detections], int32
        * num_object_detections : number of detections, [bs], int32
        * detection_object_instance_ids : (optional) detection instance ids;
          if not defined assumed to be 0 on all active detections,
          [bs, max_num_detections], int32
    """
    dynamic_incoming_keys = True
    dynamic_generated_keys = True

    incoming_keys = [
        DetectionDataFields.detection_object_boxes,
        DetectionDataFields.detection_object_scores,
        DetectionDataFields.num_object_detections,
        "_" + DetectionDataFields.detection_object_classes,
        "_" + DetectionDataFields.detection_object_instance_ids,
    ]
    generated_keys = [
        DetectionDataFields.detection_object_boxes,
        DetectionDataFields.detection_object_scores,
        DetectionDataFields.num_object_detections,
        DetectionDataFields.detection_object_classes,
        DetectionDataFields.detection_object_instance_ids,
    ]

    def __init__(self, *,
                 num_classes: int = 1,
                 min_object_width: Union[float, List[float]] = 0,
                 max_object_width: Union[float, List[float]] = 0,
                 min_object_height: Union[float, List[float]] = 0,
                 max_object_height: Union[float, List[float]] = 0,
                 score_threshold: Union[float, List[float]] = 0,
                 reorder_filtered: bool = True,
                 classes_to_select: Optional[Union[int, list]] = None,
                 **postprocessor_kwargs):
        super().__init__(**postprocessor_kwargs)
        min_object_width = _validate_classwise_parameters(
            min_object_width, num_classes, "min_object_width")
        max_object_width = _validate_classwise_parameters(
            max_object_width, num_classes, "max_object_width")
        min_object_height = _validate_classwise_parameters(
            min_object_height, num_classes, "min_object_height")
        max_object_height = _validate_classwise_parameters(
            max_object_height, num_classes, "max_object_height")
        score_threshold = _validate_classwise_parameters(
            score_threshold, num_classes, "score_threshold")

        assert all(each_threshold >= 0 for each_threshold in score_threshold), (
            "object score thresholds should be >= 0")
        self.num_classes = num_classes
        self.min_object_width = min_object_width
        self.max_object_width = max_object_width
        self.min_object_height = min_object_height
        self.max_object_height = max_object_height
        self.score_threshold = score_threshold
        self.reorder_filtered = reorder_filtered
        if isinstance(classes_to_select, int):
            classes_to_select = [classes_to_select]
        if classes_to_select and (
                not all((isinstance(each_class, int) and each_class >= 0
                         for each_class in classes_to_select))):
            msg = ("{}: provided classes_to_select is invalid! "
                   "It must be either single int or a list of ints "
                   "(provided: {})").format(self.name, classes_to_select)
            raise ValueError(msg)
        self.classes_to_select = classes_to_select

    @check_if_dynamic_keys_flat
    def process(self, *,
                detection_object_boxes,
                detection_object_scores,
                num_object_detections,
                detection_object_classes=None,
                detection_object_instance_ids=None,
                **dynamic_inputs):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        # pylint: disable=too-many-locals
        # cannot reduce number of locals without more code complexity
        min_object_width = self.add_default_placeholder(
            self.min_object_width, "min_object_width", tf.float32,
            shape=None, broadcast_shape=[self.num_classes])
        max_object_width = self.add_default_placeholder(
            self.max_object_width, "max_object_width", tf.float32,
            shape=None, broadcast_shape=[self.num_classes])
        min_object_height = self.add_default_placeholder(
            self.min_object_height, "min_object_height", tf.float32,
            shape=None, broadcast_shape=[self.num_classes])
        max_object_height = self.add_default_placeholder(
            self.max_object_height, "max_object_height", tf.float32,
            shape=None, broadcast_shape=[self.num_classes])
        score_threshold = self.add_default_placeholder(
            self.score_threshold, "score_threshold", tf.float32,
            shape=None, broadcast_shape=[self.num_classes])

        detection_object_classes = _maybe_init_classes(
            detection_object_classes, detection_object_scores)
        detection_object_instance_ids = _maybe_init_instance_ids(
            detection_object_classes, detection_object_instance_ids)

        apply_classwise = _parameters_are_classwise(
            min_object_width, max_object_width, min_object_height,
            max_object_height, score_threshold)
        mask_detections = _get_filtered_detections_mask_dynamic(
            self.num_classes,
            detection_object_boxes, detection_object_scores,
            detection_object_classes,
            num_object_detections,
            min_object_width, max_object_width,
            min_object_height, max_object_height, score_threshold,
            classwise=apply_classwise
        )
        if self.classes_to_select:
            mask_classes, _ = od_utils.create_classes_mask(
                detection_object_classes, self.classes_to_select,
                num_objects=num_object_detections)
            mask_detections = tf.logical_and(mask_detections, mask_classes)
        if self.reorder_filtered:
            (detection_object_boxes_filtered, num_object_detections_filtered,
             detection_object_classes_filtered,
             detection_object_instance_ids_filtered,
             detection_object_scores_filtered,
             dynamic_inputs) = od_utils.filter_objects_over_mask(
                 mask_detections,
                 detection_object_boxes, detection_object_classes,
                 detection_object_instance_ids, detection_object_scores,
                 **dynamic_inputs)
        else:
            num_object_detections_filtered = num_object_detections
            detection_object_boxes_filtered = _zero_according_to_object_mask(
                mask_detections, detection_object_boxes)
            detection_object_scores_filtered = _zero_according_to_object_mask(
                mask_detections, detection_object_scores)
            detection_object_classes_filtered = _zero_according_to_object_mask(
                mask_detections, detection_object_classes)
            (detection_object_instance_ids_filtered
             ) = _zero_according_to_object_mask(mask_detections,
                                                detection_object_instance_ids)
            dynamic_inputs = {
                k: _zero_according_to_object_mask(mask_detections, v)
                for k, v in dynamic_inputs.items()
            }

        result = {
            DetectionDataFields.detection_object_boxes:
                detection_object_boxes_filtered,
            DetectionDataFields.detection_object_scores:
                detection_object_scores_filtered,
            DetectionDataFields.num_object_detections:
                num_object_detections_filtered,
            DetectionDataFields.detection_object_classes:
                detection_object_classes_filtered,
            DetectionDataFields.detection_object_instance_ids:
                detection_object_instance_ids_filtered,
        }
        result.update(dynamic_inputs)
        return result


# pylint: enable=too-many-instance-attributes


class NonMaxSuppressionPostprocessor(nc7.model.ModelPostProcessor):
    """
    Perform Non Maximum Suppression on detections.

    Will add all parameters like IOU threshold to default placeholders, so
    is possible to modify them during inference.

    All of the parameters can be controlled during inference too

    It also has a dynamic keys option, so all the additional keys that will
    be provided to its postprocessor must have first dimension after batch
    to be also object dimension, e.g. `[bs, max_num_detections, *]`.
    These additional keys are not allowed to be nested, so only 1 level dicts
    are allowed.

    Parameters
    ----------
    num_classes
        number of classes; if not provided, assumes that only 1 class
    iou_threshold
        intersection over union threshold for suppression;
        can be classwise e.g. list of values; also during inference, it is
        possible to provide either 1 value for all classes or list of values
        for each class
    score_threshold
        objects score threshold for suppression;
        can be classwise e.g. list of values; also during inference, it is
        possible to provide either 1 value for all classes or list of values
        for each class
    max_size_per_class
        max number of boxes after nms per class
    max_total_size
        max number of boxes in total; if not specified, will be used the
    parallel_iterations
        number of parallel iterations to use to map the nms method over the
        batch

    Attributes
    ----------
    incoming_keys
        * detection_object_boxes : detection boxes in
          format [ymin, xmin, ymax, xmax], [bs, max_num_detections, 4], float32
        * detection_object_scores : detection scores, [bs, max_num_detections],
          float32
        * detection_object_classes : (optional) detection classes;
          if not defined assumed to be 1 on all active detections,
          [bs, max_num_detections], int32
        * num_object_detections : number of detections, [bs], int32
        * detection_object_instance_ids : (optional) detection instance ids;
          if not defined assumed to be 0 on all active detections,
          [bs, max_num_detections], int32
    generated_keys
        * detection_object_boxes : detection boxes in
          format [ymin, xmin, ymax, xmax], [bs, max_num_detections, 4], float32
        * detection_object_scores : detection scores ,[bs, max_num_detections],
          float32,
        * detection_object_classes : detection classes, zero-based,
          [bs, max_num_detections], int32
        * num_object_detections : number of detections, [bs], int32
        * detection_object_instance_ids : (optional) detection instance ids;
          if not defined assumed to be 0 on all active detections,
          [bs, max_num_detections], int32
    """
    dynamic_incoming_keys = True
    dynamic_generated_keys = True

    incoming_keys = [
        DetectionDataFields.detection_object_boxes,
        DetectionDataFields.detection_object_scores,
        DetectionDataFields.num_object_detections,
        "_" + DetectionDataFields.detection_object_classes,
        "_" + DetectionDataFields.detection_object_instance_ids,
    ]
    generated_keys = [
        DetectionDataFields.detection_object_boxes,
        DetectionDataFields.detection_object_scores,
        DetectionDataFields.num_object_detections,
        DetectionDataFields.detection_object_classes,
        DetectionDataFields.detection_object_instance_ids,
    ]

    def __init__(self, *,
                 num_classes=1,
                 iou_threshold: Union[float, List[float]] = 0.6,
                 score_threshold: Union[float, List[float]] = 0.0,
                 max_size_per_class: int = 100,
                 max_total_size: int = 200,
                 parallel_iterations: int = 16,
                 **postprocessor_kwargs):
        super().__init__(**postprocessor_kwargs)
        iou_threshold = _validate_classwise_parameters(
            iou_threshold, num_classes, "iou_threshold")
        score_threshold = _validate_classwise_parameters(
            score_threshold, num_classes, "score_threshold")
        assert all(each_threshold >= 0 for each_threshold in score_threshold), (
            "object score thresholds should be >= 0")
        assert all(each_threshold >= 0 for each_threshold in iou_threshold), (
            "iou thresholds should be >= 0")
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.parallel_iterations = parallel_iterations
        self.max_size_per_class = max_size_per_class
        self.max_total_size = max_total_size

    @check_if_dynamic_keys_flat
    def process(self, *,
                detection_object_boxes,
                detection_object_scores,
                num_object_detections,
                detection_object_classes=None,
                detection_object_instance_ids=None,
                **dynamic_inputs):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        # pylint: disable=too-many-locals
        # cannot reduce number of locals without more code complexity
        iou_threshold = self.add_default_placeholder(
            self.iou_threshold, "iou_threshold", tf.float32,
            shape=None, broadcast_shape=[self.num_classes])
        score_threshold = self.add_default_placeholder(
            self.score_threshold, "score_threshold", tf.float32,
            shape=None, broadcast_shape=[self.num_classes])
        max_size_per_class = self.add_default_placeholder(
            self.max_size_per_class, "max_size_per_class", tf.int32)
        max_total_size = self.add_default_placeholder(
            self.max_total_size, "max_total_size", tf.int32)

        detection_object_classes = _maybe_init_classes(
            detection_object_classes, detection_object_scores)
        detection_object_instance_ids = _maybe_init_instance_ids(
            detection_object_classes, detection_object_instance_ids)

        # is needed to be sure that detections are zeroed
        # according to num_object_detections
        (detection_object_boxes, num_object_detections,
         detection_object_scores, detection_object_classes,
         detection_object_instance_ids, dynamic_inputs
         ) = od_utils.zero_out_invalid_pad_objects(
             detection_object_boxes,
             detection_object_scores,
             detection_object_classes,
             detection_object_instance_ids,
             **dynamic_inputs,
             num_objects=num_object_detections)

        (detection_object_boxes_nms, detection_object_scores_nms,
         detection_object_classes_nms, num_object_detections_nms,
         additional_fields_nms
         ) = od_utils.batch_multiclass_non_max_suppression(
             object_boxes=detection_object_boxes,
             object_scores=detection_object_scores,
             object_classes=detection_object_classes,
             iou_threshold=iou_threshold,
             score_threshold=score_threshold,
             num_classes=self.num_classes,
             max_size_per_class=max_size_per_class,
             max_total_size=max_total_size,
             detection_object_instance_ids=detection_object_instance_ids,
             **dynamic_inputs)
        result = {
            DetectionDataFields.detection_object_boxes:
                detection_object_boxes_nms,
            DetectionDataFields.detection_object_scores:
                detection_object_scores_nms,
            DetectionDataFields.num_object_detections:
                num_object_detections_nms,
            DetectionDataFields.detection_object_classes:
                detection_object_classes_nms,
        }
        result.update(additional_fields_nms)
        return result


class ConverterToImageFramePostprocessor(nc7.model.ModelPostProcessor):
    """
    Convert normalized coordinates in range of [0, 1] to image coordinates with
    x in [0, width] and y in [0, height].

    Attributes
    ----------
    incoming_keys
        * images : images with shape [bs, height, width, num_channels]
        * detection_object_boxes : detection boxes in
          format [ymin, xmin, ymax, xmax], [bs, max_num_detections, 4], float32

    generated_keys
        * detection_object_boxes : detection boxes in
          format [ymin, xmin, ymax, xmax] with coordinates in image space,
          [bs, max_num_detections, 4], float32
    """

    incoming_keys = [
        ImageDataFields.images,
        DetectionDataFields.detection_object_boxes,
    ]
    generated_keys = [
        DetectionDataFields.detection_object_boxes,
    ]

    def process(self, *,
                images,
                detection_object_boxes):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        image_size = [tf.shape(images)[1], tf.shape(images)[2]]
        boxes_image_frame = od_utils.local_to_image_coordinates(
            detection_object_boxes, image_size=image_size)
        result = {
            DetectionDataFields.detection_object_boxes: boxes_image_frame}
        return result


class ExtractKeypointsFromHeatmaps(nc7.model.ModelPostProcessor):
    """
    Extract keypoints from their heatmaps. For it first apply gaussian smoothing
    and then select argmax of it

    Parameters
    ----------
    smoothing_kernel_size
        size of the gaussian smoothing kernel; can be changed during inference
    normalize_smoothing_kernel
        if the smoothing kernel should be normalized, e.g. kernel sum =1
    score_conversion_fn
        which function to use to convert the input heatmaps to its scores;
        if not specified, linear conversion is used; must be inside of
        'tf.nn' namespace

    Attributes
    ----------
    incoming_keys
        * detection_object_boxes : detection boxes in normalized coordinates,
          tf.float32, [bs, num_detections, 4]
        * detection_object_keypoints_heatmaps : predicted heatmaps (or logits)
          for keypoints; tf.float32,
          [bs, num_detections, map_width, map_height, num_keypoints]
    generated_keys
        * detection_object_keypoints : detection keypoints normalized
          to image coordinates in format [y, x]; tf.float32,
          [bs, num_detections, num_keypoints, 2]
        * detection_object_keypoints_scores :
          scores for keypoints, which are taken on heatmaps after gaussian
          filter application; so the values may be not normalized;
          shape [bs, num_detections, , num_keypoints], tf.float32
    """
    incoming_keys = [
        DetectionDataFields.detection_object_keypoints_heatmaps,
        DetectionDataFields.detection_object_boxes,
    ]
    generated_keys = [
        DetectionDataFields.detection_object_keypoints,
        DetectionDataFields.detection_object_keypoints_scores,
    ]

    def __init__(self, *,
                 smoothing_kernel_size: int = 3,
                 normalize_smoothing_kernel: bool = False,
                 score_conversion_fn: Optional[str] = None,
                 **postprocessor_kwargs):
        super().__init__(**postprocessor_kwargs)
        self.smoothing_kernel_size = smoothing_kernel_size
        self.normalize_smoothing_kernel = normalize_smoothing_kernel
        self.score_conversion_fn = score_conversion_fn

    def process(self, detection_object_keypoints_heatmaps,
                detection_object_boxes) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        smoothing_kernel_size = self.add_default_placeholder(
            self.smoothing_kernel_size, "smoothing_kernel_size", tf.int32)

        if self.score_conversion_fn:
            score_conversion_fn = getattr(tf.nn, self.score_conversion_fn)
            detection_object_keypoints_heatmaps = score_conversion_fn(
                detection_object_keypoints_heatmaps)

        num_keypoints = detection_object_keypoints_heatmaps.shape.as_list()[-1]
        heatmaps_sq = tf.reshape(
            detection_object_keypoints_heatmaps,
            tf.concat([[-1],
                       tf.shape(detection_object_keypoints_heatmaps)[2:]], 0))
        keypoints_relative_sq, keypoints_scores_sq = (
            od_utils.extract_keypoints_from_heatmaps(
                heatmaps_sq, smoothing_kernel_size,
                self.normalize_smoothing_kernel))
        first_dims = tf.shape(detection_object_keypoints_heatmaps)[:2]
        keypoints_relative = tf.reshape(
            keypoints_relative_sq,
            tf.concat([first_dims, [num_keypoints, 2]], 0))
        keypoints_scores = tf.reshape(
            keypoints_scores_sq, tf.concat([first_dims, [num_keypoints]], 0))
        keypoints = od_utils.decode_keypoints_from_boxes(
            keypoints_relative, detection_object_boxes)
        result = {
            DetectionDataFields.detection_object_keypoints: keypoints,
            DetectionDataFields.detection_object_keypoints_scores:
                keypoints_scores
        }
        return result


class KeypointsFilterPostprocessor(nc7.model.ModelPostProcessor):
    """
    Filter keypoints by their score

    Parameters
    ----------
    score_threshold
        score threshold for keypoints - all keypoints with score less it will
        be set to 0; can be changed during inference

    Attributes
    ----------
    incoming_keys
        * detection_object_keypoints : detection keypoints normalized
          to image coordinates in format [y, x]; tf.float32,
          [bs, num_detections, num_keypoints, 2]
        * detection_object_keypoints_scores :
          scores for keypoints, which are taken on heatmaps after gaussian
          filter application; so the values may be not normalized;
          shape [bs, num_detections, , num_keypoints], tf.float32
    generated_keys
        * detection_object_keypoints : detection keypoints normalized
          to image coordinates in format [y, x]; tf.float32,
          [bs, num_detections, num_keypoints, 2]
        * detection_object_keypoints_scores :
          scores for keypoints, which are taken on heatmaps after gaussian
          filter application; so the values may be not normalized;
          shape [bs, num_detections, , num_keypoints], tf.float32
    """
    incoming_keys = [
        DetectionDataFields.detection_object_keypoints,
        DetectionDataFields.detection_object_keypoints_scores,
    ]
    generated_keys = [
        DetectionDataFields.detection_object_keypoints,
        DetectionDataFields.detection_object_keypoints_scores,
    ]

    def __init__(self, score_threshold: float = 0.1,
                 **postprocessor_kwargs):
        super().__init__(**postprocessor_kwargs)
        self.score_threshold = score_threshold

    def process(self, detection_object_keypoints,
                detection_object_keypoints_scores) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        score_threshold = self.add_default_placeholder(
            self.score_threshold, "score_threshold", tf.float32)
        scores_mask = tf.greater_equal(
            detection_object_keypoints_scores,
            score_threshold
        )
        scores_filtered = tf.where(
            scores_mask,
            detection_object_keypoints_scores,
            tf.zeros_like(detection_object_keypoints_scores))
        score_mask_keypoints = tf.tile(scores_mask[..., tf.newaxis],
                                       [1, 1, 1, 2])
        keypoints_filtered = tf.where(score_mask_keypoints,
                                      detection_object_keypoints,
                                      tf.zeros_like(detection_object_keypoints))
        return {
            DetectionDataFields.detection_object_keypoints: keypoints_filtered,
            DetectionDataFields.detection_object_keypoints_scores:
                scores_filtered,
        }


class InstanceMasksToImageFrame(nc7.model.ModelPostProcessor):
    """
    Postprocessor to reframe the instance masks from boxes frames to image frame

    Parameters
    ----------
    binary_threshold
        threshold to convert values to uint8; resulted values will be 0 or 1;
        can be changed during inference

    Attributes
    ----------
    incoming_keys
        * images : images; [bs, image_height, image_width, num_channels],
          tf.float32
        * object_boxes : boxes in normalized coordinates,
          tf.float32, [bs, num_objects, 4]
        * object_instance_masks : instance masks to reframe;
          [bs, num_objects, mask_width, mask_height, {num_channels}];
          last dimension is optional; if it exists, then first the sum over
          all the channels will be taken and then this single channeled mask
          will be reframed
    generated_keys
        * object_instance_masks_on_image : instance masks reframed to image;
          [bs, num_objects, image_height, image_width]; tf.uint8
    """
    incoming_keys = [
        ImageDataFields.images,
        ObjectDataFields.object_boxes,
        ObjectDataFields.object_instance_masks,
    ]
    generated_keys = [
        ObjectDataFields.object_instance_masks_on_image,
    ]

    def __init__(self, *,
                 binary_threshold=0.5,
                 **postprocessor_kwargs):
        super().__init__(**postprocessor_kwargs)
        self.binary_threshold = binary_threshold

    def process(self, images: tf.Tensor,
                object_boxes: tf.Tensor,
                object_instance_masks: tf.Tensor
                ) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        binary_threshold = self.add_default_placeholder(
            self.binary_threshold, "binary_threshold", tf.float32)

        if len(object_instance_masks.shape) == 5:
            if object_instance_masks.shape.as_list()[-1] > 1:
                object_instance_masks = tf.reduce_sum(object_instance_masks, -1)
            else:
                object_instance_masks = tf.squeeze(object_instance_masks, -1)

        image_sizes = od_utils.get_true_image_shapes(
            images)[:, :-1]
        masks_on_image = od_utils.decode_instance_masks_to_image(
            object_instance_masks, object_boxes, image_sizes)
        masks_on_image_binary = tf.cast(tf.greater(
            masks_on_image, binary_threshold), tf.uint8)
        batch, height, width = images.shape[:3]
        masks_on_image_binary.set_shape(
            [batch, object_boxes.shape[1], height, width])
        return {
            ObjectDataFields.object_instance_masks_on_image:
                masks_on_image_binary
        }


class KeypointsConverterToImageFramePostprocessor(nc7.model.ModelPostProcessor):
    """
    Convert normalized keypoints in range of [0, 1] to image coordinates with
    x in [0, width] and y in [0, height].

    Attributes
    ----------
    incoming_keys
        * images : images with shape [bs, height, width, num_channels]
        * detection_object_keypoints : detection keypoints in
          format [y, x], [bs, max_num_detections, num_keypoints, 4], float32

    generated_keys
        * detection_object_keypoints : detection keypoints with coordinates
          in image frame in  format [y, x],
          [bs, max_num_detections, num_keypoints, 4], float32
    """

    incoming_keys = [
        ImageDataFields.images,
        DetectionDataFields.detection_object_keypoints,
    ]
    generated_keys = [
        DetectionDataFields.detection_object_keypoints,
    ]

    def process(self, *,
                images,
                detection_object_keypoints):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        image_size = [tf.shape(images)[1], tf.shape(images)[2]]
        keypoints_image_frame = (
            od_utils.keypoints_local_to_image_coordinates(
                detection_object_keypoints, image_size=image_size))
        result = {
            DetectionDataFields.detection_object_keypoints:
                keypoints_image_frame
        }
        return result


class ObjectClassesCombiner(nc7.model.ModelPostProcessor):
    """
    Postprocessor that allows to accumulate different class ids

    new class id is calculated as a flatten index of the multi dimensional array
    of shape [..., num_classes3, num_classes2, num_classes1], e.g. works similar
    to `numpy.ravel_multi_index` but on the transposed shape and reversed index.
    This is done due to the logic to have the class components in the
    priority order, e.g. first num_classes indices must be for class component
    1 and 0 for all other components.


    Parameters
    ----------
    num_classes_to_combine
        list of dicts with mapping from input key to number of classes for that
        key, e.g. [{"key1": 10}, {"key2": 2}, {"key3": 2}]; number of classes
        must include also a background, e.g. 0 class;
    mask_according_to_first_key
        if the first key should serve also as a mask, e.g. if it is 0, then
        result will be also 0 disregard of other classes
    """
    generated_keys = [
        ObjectDataFields.object_classes,
    ]
    dynamic_incoming_keys = True

    def __init__(self, *,
                 num_classes_to_combine: List[dict],
                 mask_according_to_first_key: bool = True,
                 **postprocessor_kwargs):
        super().__init__(**postprocessor_kwargs)
        self.num_classes_to_combine = num_classes_to_combine
        self.mask_according_to_first_key = mask_according_to_first_key

    def process(self, **inputs) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature

        class_ids_in_order = [
            inputs[list(each_item.keys())[0]]
            for each_item in self.num_classes_to_combine]
        num_classes = [list(each_item.values())[0]
                       for each_item in self.num_classes_to_combine]

        index_multiplier = tf.cast(
            tf.cumprod([1] + num_classes[:-1]), tf.int32)
        new_object_classes = tf.reduce_sum(
            tf.stack(class_ids_in_order, 0)
            * index_multiplier[:, tf.newaxis, tf.newaxis],
            0)
        if self.mask_according_to_first_key:
            new_object_classes = tf.where(tf.greater(class_ids_in_order[0], 0),
                                          new_object_classes,
                                          tf.zeros_like(new_object_classes))
        return {
            ObjectDataFields.object_classes: new_object_classes
        }


def _get_filtered_detections_mask_dynamic(num_classes,
                                          detection_object_boxes,
                                          detection_object_scores,
                                          detection_object_classes,
                                          num_object_detections,
                                          min_object_width,
                                          max_object_width, min_object_height,
                                          max_object_height, score_threshold,
                                          classwise: tf.Tensor):
    # pylint: disable=too-many-arguments
    # all the arguments are needed to perform filter operation
    mask_detections = tf.cond(
        classwise,
        lambda: _get_filtered_detections_mask(
            num_classes,
            detection_object_boxes, detection_object_scores,
            detection_object_classes,
            num_object_detections,
            min_object_width, max_object_width,
            min_object_height, max_object_height, score_threshold,
            classwise=True),
        lambda: _get_filtered_detections_mask(
            num_classes,
            detection_object_boxes, detection_object_scores,
            detection_object_classes,
            num_object_detections,
            min_object_width, max_object_width,
            min_object_height, max_object_height, score_threshold,
            classwise=False)
    )
    return mask_detections


def _get_filtered_detections_mask(num_classes,
                                  detection_object_boxes,
                                  detection_object_scores,
                                  detection_object_classes,
                                  num_object_detections,
                                  min_object_width,
                                  max_object_width, min_object_height,
                                  max_object_height, score_threshold,
                                  classwise: bool = False):
    # pylint: disable=too-many-arguments,too-many-locals
    # all the arguments are needed to perform filter operation
    if not classwise:
        num_classes = 1

    masks_detections_per_classes = [
        _get_filtered_detections_mask_single_class(
            class_index, detection_object_boxes, detection_object_scores,
            min_object_width, max_object_width, min_object_height,
            max_object_height, score_threshold
        )
        for class_index in range(num_classes)
    ]
    if classwise:
        mask_classes = [tf.equal(detection_object_classes, class_index + 1)
                        for class_index in range(num_classes)]
        masks_detections_per_classes = [
            tf.logical_and(each_classwise_filtered, each_class_mask)
            for each_classwise_filtered, each_class_mask in zip(
                masks_detections_per_classes, mask_classes)
        ]
    max_num_detections = tf.shape(detection_object_boxes)[-2]
    mask_num_detections = od_utils.get_objects_mask_from_num_objects(
        max_num_detections, num_object_detections)
    masks_detections = tf.reduce_any(
        tf.stack(masks_detections_per_classes, -1), -1)
    mask_detections = tf.logical_and(
        masks_detections, mask_num_detections)
    return mask_detections


def _get_filtered_detections_mask_single_class(
        class_index,
        detection_object_boxes,
        detection_object_scores,
        min_object_width,
        max_object_width, min_object_height,
        max_object_height, score_threshold):
    # pylint: disable=too-many-arguments
    # all the arguments are needed to perform filter operation
    mask_scores = tf.greater_equal(
        detection_object_scores, score_threshold[class_index])
    mask_width = od_utils.get_filter_mask_by_width(
        detection_object_boxes,
        min_object_width[class_index], max_object_width[class_index])
    mask_height = od_utils.get_filter_mask_by_height(
        detection_object_boxes,
        min_object_height[class_index], max_object_height[class_index])

    mask_detections = tf.reduce_all(
        tf.stack([mask_scores, mask_width, mask_height], -1), -1)
    return mask_detections


def _maybe_init_classes(detection_object_classes, detection_object_scores):
    if detection_object_classes is None:
        detection_object_classes = tf.cast(
            tf.zeros_like(detection_object_scores), tf.int32)
    return detection_object_classes


def _maybe_init_instance_ids(detection_object_classes,
                             detection_object_instance_ids):
    if detection_object_instance_ids is None:
        detection_object_instance_ids = tf.zeros_like(detection_object_classes)
    return detection_object_instance_ids


def _parameters_are_classwise(*parameters):
    parameters_are_classwise = tf.logical_not(tf.reduce_all(
        tf.stack([
            tf.reduce_all(tf.equal(each_item, each_item[0]))
            for each_item in parameters
        ])))
    return parameters_are_classwise


def _validate_classwise_parameters(parameter, num_classes, parameter_name):
    if not isinstance(parameter, (list, tuple)):
        parameter = [parameter]
    else:
        assert len(parameter) == num_classes, (
            "Parameter {} should be of length num_classes, {}, to use "
            "classwise! To use it class agnostic, provide only a scalar"
        ).format(parameter_name, num_classes)
    return parameter


def _zero_according_to_object_mask(mask: tf.Tensor, item: tf.Tensor
                                   ) -> tf.Tensor:
    mask = tf.cast(broadcast_with_expand_to(mask, item), item.dtype)
    item_zeroed = mask * item
    return item_zeroed
