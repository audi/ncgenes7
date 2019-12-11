# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Data readers for open images dataset
"""

import os
from typing import Optional

import nucleus7 as nc7
import numpy as np

from ncgenes7.data_fields.object_detection import ObjectDataFields
from ncgenes7.utils.open_images_utils import get_annotation_mapping
from ncgenes7.utils.open_images_utils import get_class_descriptions_mapping
from ncgenes7.utils.open_images_utils import query_labels_by_image_id


class OpenImagesObjectsReader(nc7.data.DataReader):
    """
    Object reader for OpenImages dataset

    Parameters
    ----------
    file_name_annotations
        path to annotations_bbox.csv
    file_name_class_descriptions
        path to class_descriptions.csv

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
                 file_name_annotations: str,
                 file_name_class_descriptions: str,
                 **reader_kwargs):
        super().__init__(**reader_kwargs)
        self.file_name_annotations = file_name_annotations
        self.file_name_class_descriptions = file_name_class_descriptions
        self._class_descriptions = None  # type: Optional[dict]
        self._annotations = None

    def build(self):
        super().build()
        self._build_labels_annotations()
        return self

    def read(self, images):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        result = self._read_objects_from_fname(images)
        return result

    def _read_objects_from_fname(self, images_fnames: str) -> dict:
        image_id = os.path.splitext(os.path.split(images_fnames)[-1])[0]
        object_classes, object_boxes = query_labels_by_image_id(
            image_id, self._annotations, self._class_descriptions)
        object_instance_ids = np.zeros_like(object_classes, np.int64) - 1

        result = {ObjectDataFields.object_classes: object_classes,
                  ObjectDataFields.object_instance_ids: object_instance_ids,
                  ObjectDataFields.object_boxes: object_boxes}
        return result

    def _build_labels_annotations(self):
        self._class_descriptions = get_class_descriptions_mapping(
            self.file_name_class_descriptions)
        self._annotations = get_annotation_mapping(self.file_name_annotations)
