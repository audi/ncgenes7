# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Data readers for COCO dataset
"""

import os
from typing import Optional

import nucleus7 as nc7

from ncgenes7.data_fields.object_detection import ObjectDataFields
from ncgenes7.data_fields.semantic_segmentation import SegmentationDataFields
from ncgenes7.utils import coco_utils


class CocoObjectsReader(nc7.data.DataReader):
    """
    Object reader for COCO dataset

    Parameters
    ----------
    file_name_instance_annotations
        path to instance annotation file like instances_val2017.json
    remove_unused_classes
        if the unused classes must be removed and so at the end there
        are 80 object classes

    Attributes
    ----------
    generated_keys
        * object_boxes : detection boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [bs, num_detections, 4]
          and with values in [0, 1]; np.float32
        * object_classes : classes for detections, 1-based,
          [bs, num_detections]; np.int64
        * object_instance_ids : instance ids for detections,
          [bs, num_detections]; == 0 if no id was found; np.int64
    """
    file_list_keys = [
        "images",
    ]
    generated_keys = [
        ObjectDataFields.object_boxes,
        ObjectDataFields.object_classes,
        ObjectDataFields.object_instance_ids,
    ]

    def __init__(self, *,
                 file_name_instance_annotations: str,
                 remove_unused_classes: bool = False,
                 **reader_kwargs):
        super().__init__(**reader_kwargs)
        self.file_name_instance_annotations = file_name_instance_annotations
        self.remove_unused_classes = remove_unused_classes
        self._image_id_to_objects = None

    def build(self):
        super().build()
        self._build_labels_annotations()
        return self

    def read(self, images):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        image_fname = os.path.splitext(images)[0] + ".jpg"
        result = self._read_objects_from_fname(image_fname)
        return result

    def _read_objects_from_fname(self, images_fnames: str) -> dict:
        image_id = os.path.basename(images_fnames)
        (object_boxes, object_classes, object_instance_ids
         ) = coco_utils.read_objects_from_fname(image_id,
                                                self._image_id_to_objects)

        result = {ObjectDataFields.object_classes: object_classes,
                  ObjectDataFields.object_instance_ids: object_instance_ids,
                  ObjectDataFields.object_boxes: object_boxes}
        return result

    def _build_labels_annotations(self):
        self._image_id_to_objects = (
            coco_utils.get_class_descriptions_mapping(
                self.file_name_instance_annotations,
                remove_unused_classes=self.remove_unused_classes))


class CocoSemanticSegmentationReader(nc7.data.DataReader):
    """
    panoptic labels reader for COCO dataset which represents panoptic labels
    as semantic segmentation

    Parameters
    ----------
    image_size
        image size in format [height, width]; if provided, segmentation images
        will be resized to this size
    file_name_panoptic_annotations
        path to instance annotation file like instances_val2017.json
    segmentation_dtype
        dtype of result segmentation image
    remove_unused_classes
        if the unused classes must be removed and so at the end there
        are 133 segmentation classes

    Attributes
    ----------
    generated_keys
        * segmentation_classes : semantic segmentation classes,
          [bs, height, width, 1], segmentation_dtype
    """
    file_list_keys = [
        "panoptic",
    ]
    generated_keys = [
        SegmentationDataFields.segmentation_classes,
    ]

    def __init__(self, *,
                 image_size: Optional[list] = None,
                 file_name_panoptic_annotations: str = None,
                 segmentation_dtype: str = "uint8",
                 remove_unused_classes: bool = False,
                 **reader_kwargs):
        super().__init__(**reader_kwargs)
        self.image_size = image_size
        self.file_name_panoptic_annotations = file_name_panoptic_annotations
        self.segmentation_dtype = segmentation_dtype
        self.remove_unused_classes = remove_unused_classes
        self._image_fname_to_class_id_hash_fn = None

    def build(self):
        super().build()
        self._build_labels_annotations()
        return self

    def read(self, *, panoptic):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        result = self._read_coco_panoptic_from_fname(panoptic)
        return result

    def _read_coco_panoptic_from_fname(self, panoptic_fnames: str) -> dict:
        segmentation_classes = coco_utils.read_segmentation_from_fname(
            panoptic_fnames, self._image_fname_to_class_id_hash_fn,
            self.image_size, dtype=self.segmentation_dtype)
        result = {SegmentationDataFields.segmentation_classes:
                      segmentation_classes}
        return result

    def _build_labels_annotations(self):
        annotations = self.file_name_panoptic_annotations
        (self._image_fname_to_class_id_hash_fn
         ) = coco_utils.panoptic_categories_to_rgb_hash_fn(
             annotations, remove_unused_classes=self.remove_unused_classes)


class CocoPersonKeypointsReader(nc7.data.DataReader):
    """
    Keypoints reader for COCO dataset

    Parameters
    ----------
    file_name_person_keypoints_annotations
        path to person keypoints annotation file like
        person_keypoints_val2017.json

    Attributes
    ----------
    generated_keys
        * object_keypoints : object keypoints coordinates normalized to image
          coordinates and with shape [num_objects, 17, 2] with coordinates in
          format [y, x]; np.float32
        * object_keypoints_visibilities : visibility flag from coco dataset,
          where 0 - not visible keypoint, 1 - not visible but annotated and
          2 - visible and annotated; shape [num_objects, 17], np.int32
        * object_boxes : detection boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [bs, num_detections, 4]
          and with values in [0, 1]; np.float32
        * object_classes : classes for detections, 1-based,
          [bs, num_detections]; np.int64
        * object_instance_ids : instance ids for detections,
          [bs, num_detections]; == 0 if no id was found; np.int64
    """
    file_list_keys = [
        "images",
    ]
    generated_keys = [
        ObjectDataFields.object_keypoints,
        ObjectDataFields.object_keypoints_visibilities,
        ObjectDataFields.object_boxes,
        ObjectDataFields.object_classes,
        ObjectDataFields.object_instance_ids,
    ]

    def __init__(self, *,
                 file_name_person_keypoints_annotations,
                 **reader_kwargs):
        super().__init__(**reader_kwargs)
        self.file_name_person_keypoints_annotations = (
            file_name_person_keypoints_annotations)
        self._image_id_to_objects = None

    def build(self):
        super().build()
        self._build_labels_annotations()
        return self

    def read(self, images):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        image_fname = os.path.splitext(images)[0] + ".jpg"
        result = self._read_objects_from_fname(image_fname)
        return result

    def _read_objects_from_fname(self, images_fnames: str) -> dict:
        image_id = os.path.basename(images_fnames)
        (object_boxes, object_classes, object_instance_ids
         ) = coco_utils.read_objects_from_fname(image_id,
                                                self._image_id_to_objects)
        (keypoints, keypoints_vis
         ) = coco_utils.read_keypoints_from_fname(image_id,
                                                  self._image_id_to_objects)

        result = {
            ObjectDataFields.object_keypoints: keypoints,
            ObjectDataFields.object_keypoints_visibilities: keypoints_vis,
            ObjectDataFields.object_classes: object_classes,
            ObjectDataFields.object_instance_ids: object_instance_ids,
            ObjectDataFields.object_boxes: object_boxes}
        return result

    def _build_labels_annotations(self):
        self._image_id_to_objects = (
            coco_utils.get_class_descriptions_mapping(
                self.file_name_person_keypoints_annotations,
                remove_unused_classes=False, with_keypoints=True))
