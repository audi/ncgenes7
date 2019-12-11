# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Object detection callbacks
"""
from typing import Optional

import nucleus7 as nc7
from nucleus7.utils import deprecated
from nucleus7.utils import io_utils
import numpy as np
import object_detection.utils.visualization_utils as od_vis_utils

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.data_fields.object_detection import ObjectDataFields
from ncgenes7.utils import object_detection_utils
from ncgenes7.utils import vis_utils
from ncgenes7.utils.object_detection_io_utils import maybe_create_category_index
from ncgenes7.utils.object_detection_io_utils import save_objects_to_file


class ObjectDrawerCallback(nc7.coordinator.CoordinatorCallback):
    """
    Draw objects on the images

    Parameters
    ----------
    num_classes
        number of classes
    class_names_to_labels_mapping
        either str with file name to mapping or dict with mapping itself;
        mapping should be in format {"class name": {"class_id": 1}, },
        where class_id is an unique integer
    use_normalized_coordinates
        if the boxes and optional keypoints are in normalized coordinates
    score_threshold
        all detections which score is less than this threshold will be ignored
    max_boxes_to_draw
        max number of drawn boxes (objects)
    line_thickness
        line thickness of drawn bounding boxes

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [bs, h, w, num_channels], np.float
        * objects_boxes : bounding boxes in format [ymin, xmin, ymax, xmax] with
          normalized to image shape coordinates, [bs, num_objects, 4],
          np.float32
        * objects_classes : class ids, [bs, num_objects], np.int
        * objects_scores : (optional) confidence scores for each object,
          [bs, num_objects], np.float
        * object_instance_ids : (optional) instance ids of objects;
          if provided, colors of the drawn boxes will be calculated according
          to ids, not classes; [bs, num_objects], np.int
        * object_keypoints : detection keypoints normalized
          to image coordinates in format [y, x]; np.float32,
          [bs, num_detections, num_keypoints, 2]
        * object_instance_masks_on_image : instance masks reframed to image;
          [bs, num_objects image_height, image_width]; tf.uint8
    generated_keys
        * images_with_objects : images overlaid with bounding boxes,
          [bs, h, w, num_channels], np.float
    """
    incoming_keys = [
        ImageDataFields.images,
        ObjectDataFields.object_boxes,
        ObjectDataFields.object_classes,
        "_" + ObjectDataFields.object_scores,
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
                 class_names_to_labels_mapping=None,
                 use_normalized_coordinates=True,
                 score_threshold=0.0,
                 max_boxes_to_draw=None,
                 line_thickness=4,
                 **callback_kwargs):
        super().__init__(**callback_kwargs)
        self.class_names_to_labels_mapping = class_names_to_labels_mapping
        self.num_classes = num_classes
        self.category_index = None
        self.score_threshold = score_threshold
        self.max_boxes_to_draw = max_boxes_to_draw
        self.line_thickness = line_thickness
        self.use_normalized_coordinates = use_normalized_coordinates

    def build(self):
        self.num_classes, self.category_index = maybe_create_category_index(
            self.num_classes, self.class_names_to_labels_mapping)
        return super().build()

    def on_iteration_end(
            self, *,
            images: np.ndarray,
            object_boxes: np.ndarray,
            object_classes: np.ndarray,
            object_scores: Optional[np.ndarray] = None,
            object_instance_ids: Optional[np.ndarray] = None,
            object_keypoints: Optional[np.ndarray] = None,
            object_instance_masks_on_image: Optional[np.ndarray] = None
    ) -> dict:
        # pylint: disable=arguments-differ,too-many-locals
        # parent method has more generic signature
        object_classes = object_classes.astype(np.int64)
        object_scores = _maybe_initialize_object_scores(
            object_classes, object_scores)

        images_with_objects = []
        for sample_index, image in enumerate(images):
            if image.ndim == 3 and image.shape[-1] == 1:
                image = np.tile(image, [1, 1, 3])
            image = (image * 255.0).astype(np.uint)
            track_ids = (object_instance_ids[sample_index]
                         if object_instance_ids is not None else None)
            sample_object_keypoints = (
                object_keypoints[sample_index]
                if object_keypoints is not None else None)
            sample_instance_masks = (
                object_instance_masks_on_image[sample_index].astype(np.uint8)
                if object_instance_masks_on_image is not None else None)

            image_with_objects = (
                od_vis_utils.visualize_boxes_and_labels_on_image_array(
                    image=image,
                    boxes=object_boxes[sample_index],
                    classes=object_classes[sample_index],
                    scores=object_scores[sample_index],
                    keypoints=sample_object_keypoints,
                    instance_masks=sample_instance_masks,
                    category_index=self.category_index,
                    use_normalized_coordinates=
                    self.use_normalized_coordinates,
                    track_ids=track_ids,
                    max_boxes_to_draw=self.max_boxes_to_draw,
                    min_score_thresh=self.score_threshold,
                    line_thickness=self.line_thickness))
            image_with_objects_float = (image_with_objects.astype(np.float32)
                                        / 255.0)
            images_with_objects.append(image_with_objects_float)
        images_with_objects_batch = np.stack(images_with_objects, 0)
        result = {
            ObjectDataFields.images_with_objects: images_with_objects_batch,
        }
        return result


class KeypointsConnectionsDrawerCallback(nc7.coordinator.CoordinatorCallback):
    """
    Draw keypoints connections on image

    Can add a legend on the right side of the image with mapping of the
    connection color to connection name. So the image width will be increased

    Parameters
    ----------
    connection_map
        dict representing the mapping of connection name to indices of the
        connections and the color of connection in form
        {"top-shoulder_left": {"color": [170, 0, 255], "connection": [0, 6]};
        or the file name for json file with this mapping;
        if not all colors are specified, then random colors will be used.
    thickness
        thickness of connections in pixels
    add_legend
        if the legend should be added to right side of image

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [bs, h, w, num_channels], np.float
        * object_keypoints : detection keypoints in absolute coordinates
          in format [y, x]; np.float32, [bs, num_detections, num_keypoints, 2]
    generated_keys
        * images_with_objects : images overlaid with keypoints connections
          and optional legend on the right side, [bs, h, w+, num_channels],
          np.float
    """
    incoming_keys = [
        ImageDataFields.images,
        ObjectDataFields.object_keypoints,
    ]
    generated_keys = [
        ObjectDataFields.images_with_objects,
    ]

    def __init__(self, *,
                 connection_map: dict,
                 thickness: int = 2,
                 add_legend: bool = True,
                 **callback_kwargs):
        super().__init__(**callback_kwargs)
        self.connection_map = connection_map
        self.thickness = thickness
        self.add_legend = add_legend
        self._connection_map_only = None
        self._colors = None

    def build(self):
        super().build()
        connection_map = io_utils.maybe_load_json(self.connection_map)
        self._connection_map_only = {
            each_name: each_item["connection"]
            for each_name, each_item in connection_map.items()}
        try:
            self._colors = {
                each_name: each_item["color"]
                for each_name, each_item in connection_map.items()}
        except KeyError:
            self._colors = None
        return self

    def on_iteration_end(
            self, *,
            images: np.ndarray,
            object_keypoints: np.ndarray
    ) -> dict:
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        images_with_objects = []
        for sample_index, image in enumerate(images):
            if image.ndim == 3 and image.shape[-1] == 1:
                image = np.tile(image, [1, 1, 3])
            image = (image * 255.0).astype(np.uint8)
            image_with_objects = vis_utils.draw_keypoints_connections(
                image, object_keypoints[sample_index],
                connection_map=self._connection_map_only,
                colors=self._colors,
                thickness=self.thickness,
                add_legend=self.add_legend,
            )
            image_with_objects_float = (image_with_objects.astype(np.float32)
                                        / 255.0)
            images_with_objects.append(image_with_objects_float)
        images_with_objects_batch = np.stack(images_with_objects, 0)
        result = {
            ObjectDataFields.images_with_objects: images_with_objects_batch,
        }
        return result


class AdditionalAttributesDrawerCallback(nc7.coordinator.CoordinatorCallback):
    """
    Can draw the attributes of objects as a rectangles marks on the right or
    left side of the bounding box and attributes represented as colors.

    Can add a legend on the right side of the image with mapping of the
    connection color to connection name. So the image width will be increased

    Attributes can be provided as arbitrary incoming keys, but their names must
    be inside of attributes_class_names_to_labels_mappings.

    Parameters
    ----------
    attributes_class_names_to_labels_mappings
        mapping from attribute name the mapping of to class id and its color as
        {"attribute1": {"attribute1_class_1":
        {"class_id": 1, "color": [50, 255, 0]}}};
        all classes for each attribute must have mapping to class_id;
        or the file name for json file with this mapping;
        if not all colors are specified, then random colors will be used.
    thickness
        thickness of attribute rectangle mark in pixels
    add_legend
        if the legend should be added to right side of image

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [bs, h, w, num_channels], np.float
        * object_boxes : object boxes in absolute coordinates in format
          [ymin, xmin, ymax, xmax], [bs, max_num_detections, 4], float32
    generated_keys
        * images_with_objects : images overlaid with attributes
          and optional legend on the right side, [bs, h, w+, num_channels],
          np.float

    """
    dynamic_incoming_keys = True
    incoming_keys = [
        ImageDataFields.images,
        ObjectDataFields.object_boxes,
    ]
    generated_keys = [
        ObjectDataFields.images_with_objects,
    ]

    def __init__(self,
                 attributes_class_names_to_labels_mappings: dict,
                 thickness: int = 5,
                 add_legend: bool = True,
                 **callback_kwargs):
        super().__init__(**callback_kwargs)
        self.attributes_class_names_to_labels_mappings = (
            attributes_class_names_to_labels_mappings)
        self.thickness = thickness
        self.add_legend = add_legend
        self._class_ids_to_names_mapping = None
        self._colors = None

    def build(self):
        super().build()
        self._class_ids_to_names_mapping = {}
        self._colors = {}
        for each_attribute_name, each_mapping in (
                self.attributes_class_names_to_labels_mappings.items()):
            each_mapping = io_utils.maybe_load_json(each_mapping)
            class_id_to_name_mapping = {
                each_item["class_id"]: each_class_name
                for each_class_name, each_item in each_mapping.items()}
            self._class_ids_to_names_mapping[each_attribute_name] = (
                class_id_to_name_mapping)
            if self._colors is None:
                continue

            try:
                colors = {
                    each_item["class_id"]: each_item["color"]
                    for each_class_name, each_item in each_mapping.items()
                }
                self._colors[each_attribute_name] = colors
            except KeyError:
                self._colors = None
        return self

    def on_iteration_end(self, images, object_boxes, **attributes) -> dict:
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        images_with_objects = []
        attributes = {k: v for k, v in attributes.items()
                      if k in self._class_ids_to_names_mapping}
        for sample_index, image in enumerate(images):
            if image.ndim == 3 and image.shape[-1] == 1:
                image = np.tile(image, [1, 1, 3])
            image = (image * 255.0).astype(np.uint8)
            attributes_sample = {k: v[sample_index]
                                 for k, v in attributes.items()}
            image_with_objects = vis_utils.draw_attributes_as_marks(
                image, object_boxes[sample_index], attributes_sample,
                class_ids_to_names_mapping=self._class_ids_to_names_mapping,
                colors=self._colors,
                thickness=self.thickness,
                add_legend=self.add_legend,
            )
            image_with_objects_float = (image_with_objects.astype(np.float32)
                                        / 255.0)
            images_with_objects.append(image_with_objects_float)
        images_with_objects_batch = np.stack(images_with_objects, 0)
        result = {
            ObjectDataFields.images_with_objects: images_with_objects_batch,
        }
        return result


class NonMaxSuppressionCallback(nc7.coordinator.CoordinatorCallback):
    """
    Perform non-max suppression on object bounding boxes

    Parameters
    ----------
    num_classes
        number of classes
    score_threshold
        all detections which score is less than this threshold will be ignored
    nms_iou_threshold
        if detection has iou with other detections more than this threshold,
        that detection will be ignored (non max suppression)

    Attributes
    ----------
    incoming_keys
        * objects_boxes : bounding boxes in format [ymin, xmin, ymax, xmax] with
          normalized to image shape coordinates, [bs, num_detections, 4],
          np.float32
        * objects_classes : class ids, [bs, num_detections], np.int
        * objects_scores : (optional) confidence scores for each object,
          [bs, num_detections], np.float
        * objects_instance_ids : (optional) instance ids,
          [bs, num_detections], np.int
    generated_keys
        * objects_boxes : bounding boxes in format [ymin, xmin, ymax, xmax] with
          normalized to image shape coordinates, [bs, num_detections, 4],
          np.float32
        * objects_classes : class ids, [bs, num_detections], np.int
        * objects_scores : (optional) confidence scores for each object,
          [bs, num_detections], np.float
        * objects_instance_ids : (optional) instance ids,
          [bs, num_detections], np.int
        * num_objects
          number of objects for each sample, [bs], np.int
    """
    incoming_keys = [
        ObjectDataFields.object_boxes,
        ObjectDataFields.object_classes,
        "_" + ObjectDataFields.object_instance_ids,
        "_" + ObjectDataFields.object_scores,
    ]
    generated_keys = [
        ObjectDataFields.object_boxes,
        ObjectDataFields.object_classes,
        ObjectDataFields.object_instance_ids,
        ObjectDataFields.object_scores,
        ObjectDataFields.num_objects,
    ]

    def __init__(self, *,
                 num_classes: int,
                 score_threshold: float = 0.0,
                 nms_iou_threshold: float = 1.0,
                 **callback_kwargs):
        super(NonMaxSuppressionCallback, self).__init__(**callback_kwargs)
        assert 0.0 <= score_threshold <= 1.0, (
            "score_threshold must be in [0.0, 1.0] (provided: {})".format(
                score_threshold))
        assert 0.0 <= nms_iou_threshold <= 1.0, (
            "nms_iou_threshold must be in [0.0, 1.0] (provided: {})".format(
                score_threshold))
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def on_iteration_end(self, *, object_boxes, object_classes,
                         object_scores=None, object_instance_ids=None):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        object_classes = object_classes.astype(np.int64)
        object_instance_ids = _maybe_initialize_instance_ids(
            object_classes, object_instance_ids)
        object_scores = _maybe_initialize_object_scores(
            object_classes, object_scores)

        batch_size = len(object_boxes)
        result = {ObjectDataFields.object_boxes: [],
                  ObjectDataFields.object_classes: [],
                  ObjectDataFields.object_scores: [],
                  ObjectDataFields.object_instance_ids: []}
        num_objects = []
        for sample_index in range(batch_size):
            boxes_nms, scores_nms, classes_nms, instance_ids_nms = (
                object_detection_utils.multiclass_non_max_suppression_np(
                    object_boxes=object_boxes[sample_index],
                    object_scores=object_scores[sample_index],
                    object_classes=object_classes[sample_index],
                    num_classes=self.num_classes,
                    score_thresh=self.score_threshold,
                    iou_threshold=self.nms_iou_threshold,
                    instance_ids=object_instance_ids[sample_index]))
            scores_mask = scores_nms >= self.score_threshold
            result[ObjectDataFields.object_boxes].append(
                boxes_nms[scores_mask])
            result[ObjectDataFields.object_classes].append(
                classes_nms[scores_mask])
            result[ObjectDataFields.object_scores].append(
                scores_nms[scores_mask])
            result[ObjectDataFields.object_instance_ids].append(
                instance_ids_nms[scores_mask])
            num_objects.append(scores_mask.sum())

        num_objects = np.stack(num_objects, 0)
        result = {
            each_key: object_detection_utils.stack_and_pad_object_data(
                each_object_data,
                (-1 if each_key == ObjectDataFields.object_instance_ids else 0))
            for each_key, each_object_data in result.items()
        }
        result[ObjectDataFields.num_objects] = num_objects
        return result


class ObjectsSaver(nc7.coordinator.SaverCallback):
    """
    Save objects in json format:

    [{'bbox': {'xmin': , 'ymin': 'xmax': , 'ymax': , 'w': , 'h': },
      'class_label': ,
      'id': , 'score': ,
      'keypoints': [[y, x], [y, x], ...],
      'keypoints_scores': [...],
      'keypoints_visibilities': [...]}, ...]

    Attributes
    ----------
    incoming_keys
        * objects_boxes : bounding boxes in format [ymin, xmin, ymax, xmax] with
          normalized or absolute to image shape coordinates,
          [bs, num_objects, 4], np.float32
        * objects_classes : class ids, [bs, num_objects], np.int
        * objects_scores : (optional) confidence scores for each object,
          [bs, num_detections], np.float
        * objects_instance_ids : (optional) instance ids,
          [bs, num_objects], np.int
        * save_names : (optional) save names to use;
          see `nc7.coordinator.SaverCallback` for more info
    """
    dynamic_incoming_keys = True
    incoming_keys = [
        ObjectDataFields.object_boxes,
        ObjectDataFields.object_classes,
        "_" + ObjectDataFields.object_instance_ids,
        "_" + ObjectDataFields.object_scores,
    ]

    def save_sample(self, *, object_boxes, object_classes,
                    object_instance_ids=None, object_scores=None,
                    **dynamic_object_attributes):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        save_fname = ".".join([self.save_name, "json"])
        save_objects_to_file(
            fname=save_fname,
            object_boxes=object_boxes,
            object_classes=object_classes,
            object_scores=object_scores,
            instance_ids=object_instance_ids,
            **dynamic_object_attributes
        )

    def on_iteration_end(self, *, object_boxes, object_classes,
                         object_instance_ids=None, object_scores=None,
                         save_names=None,
                         **dynamic_object_attributes):
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        object_classes = object_classes.astype(np.int64)
        object_instance_ids = _maybe_initialize_instance_ids(
            object_classes, object_instance_ids)
        object_scores = _maybe_initialize_object_scores(
            object_classes, object_scores)
        super(ObjectsSaver, self).on_iteration_end(
            object_boxes=object_boxes,
            object_classes=object_classes,
            object_instance_ids=object_instance_ids,
            object_scores=object_scores,
            save_names=save_names,
            **dynamic_object_attributes
        )


class ConverterToImageFrameCallback(nc7.coordinator.CoordinatorCallback):
    """
    Convert normalized coordinates in range of [0, 1] to image coordinates with
    x in [0, width] and y in [0, height].

    Attributes
    ----------
    incoming_keys
        * images : images with shape [bs, height, width, num_channels]
        * object_boxes : object boxes in
          format [ymin, xmin, ymax, xmax], [bs, max_num_detections, 4], float32

    generated_keys
        * object_boxes : object boxes in
          format [ymin, xmin, ymax, xmax] with coordinates in image space,
          [bs, max_num_detections, 4], float32
    """

    incoming_keys = [
        ImageDataFields.images,
        ObjectDataFields.object_boxes,
    ]
    generated_keys = [
        ObjectDataFields.object_boxes,
    ]

    def on_iteration_end(self, *,
                         images,
                         object_boxes):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        image_size = images.shape[1:3]
        boxes_image_frame = (
            object_detection_utils.local_to_image_coordinates_np(
                object_boxes, image_size=image_size))
        result = {ObjectDataFields.object_boxes: boxes_image_frame}
        return result


class NormalizeCoordinatesCallback(nc7.coordinator.CoordinatorCallback):
    """
    Convert image coordinates with x in range of [0, width] and y in range of
    [0, height] to normalized coordinates in range [0, 1]

    Attributes
    ----------
    incoming_keys
        * images : images with shape [bs, height, width, num_channels]
        * object_boxes : object boxes in
          format [ymin, xmin, ymax, xmax], [bs, max_num_detections, 4], float32

    generated_keys
        * object_boxes : object boxes in
          format [ymin, xmin, ymax, xmax] with normalized coordinates,
          [bs, max_num_detections, 4], float32
    """

    incoming_keys = [
        ImageDataFields.images,
        ObjectDataFields.object_boxes,
    ]
    generated_keys = [
        ObjectDataFields.object_boxes,
    ]

    def on_iteration_end(self, *,
                         images,
                         object_boxes):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        image_size = images.shape[1:3]
        boxes_image_frame = object_detection_utils.normalize_bbox_np(
            object_boxes, image_size=image_size)
        result = {ObjectDataFields.object_boxes: boxes_image_frame}
        return result


def _maybe_initialize_object_scores(object_classes, object_scores):
    # pylint: disable=no-member
    # some bug on some environments
    if object_scores is None:
        object_scores = np.ones_like(object_classes).astype(np.float32)
    return object_scores


def _maybe_initialize_instance_ids(object_classes,
                                   object_instance_ids):
    if object_instance_ids is None:
        object_instance_ids = -1 * np.ones_like(object_classes)
    return object_instance_ids
