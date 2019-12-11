# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Data readers for semantic segmentation
"""

from typing import Optional
from typing import Union

import nucleus7 as nc7
from nucleus7.utils import io_utils
import numpy as np

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.data_fields.semantic_segmentation import SegmentationDataFields
from ncgenes7.data_readers.image import ImageDataReaderTfRecords
from ncgenes7.utils import image_io_utils
from ncgenes7.utils import image_utils


# pylint: disable=too-many-instance-attributes
# attributes cannot be combined or extracted further
class SemanticSegmentationReader(nc7.data.DataReader):
    """
    Extract class labels from png encoded label images and optionally edges
    between specific classes with specified edge thickness.

    Parameters
    ----------
    rgb_class_mapping
        dict which maps class ids to its rbg values; one class can be mapped
        by multiple rbg values, but not vice versa;
        format should be following:
        `{"class_ids": [1, 2, 2], "rgb": [[0, 0, 0], [0, 0, 10], [0, 10, 0]]}`.
        In that case, rbg values [0, 0, 0] correspond to class id 1, whereas
        [0, 0, 10], [0, 10, 0] both correspond to class 2
    extract_class_ids
        if the images with class indices should be generated;
        size will be [height, width]
    extract_edges
        if the images with edges should be generated;
        size will be [height, width]
    class_ids_for_edge_extraction
        semantic classes to use for edge extraction; if not specified, edges
        will be extracted for all classes
    edge_thickness
        thickness of generated edges
    image_number_of_channels
        number of channels in image; if is not equal to original image, then
        it will be modified, e.g. gray -> rgb or rgb -> gray
    image_size
        resize image to this size before saving; if not specified, images
        will be encoded in original size
    segmentation_dtype
        dtype of result segmentation image

    Attributes
    ----------
    generated_keys
        * segmentation_classes : semantic segmentation classes,
          [bs, height, width, 1], segmentation_dtype
        * segmentation_edges : optional pixel wise edges,
          [bs, height, width, 1], np.int64

    """
    file_list_keys = [
        "labels",
    ]
    generated_keys = [
        "_" + SegmentationDataFields.segmentation_classes,
        "_" + SegmentationDataFields.segmentation_edges,
    ]

    def __init__(self, *,
                 rgb_class_mapping: Union[dict, str],
                 image_size: Optional[list] = None,
                 extract_class_ids: bool = True,
                 extract_edges: bool = True,
                 class_ids_for_edge_extraction: Optional[list] = None,
                 edge_thickness: int = 3,
                 segmentation_dtype: str = "uint8",
                 **reader_kwargs):
        if not extract_edges and not extract_class_ids:
            raise ValueError("One of extract_edges or extract_class_ids must "
                             "be set!")
        super().__init__(**reader_kwargs)
        self.extract_class_ids = extract_class_ids
        self.extract_edges = extract_edges
        self.edge_classes = class_ids_for_edge_extraction
        self.edge_thickness = edge_thickness
        self.rgb_class_mapping = rgb_class_mapping
        self.image_size = image_size
        self.segmentation_dtype = segmentation_dtype
        self._rgb_class_ids_mapping_hashed = None  # type: dict

    def build(self):
        super().build()
        self.rgb_class_mapping = io_utils.maybe_load_json(
            self.rgb_class_mapping)
        self._rgb_class_ids_mapping_hashed = dict(
            zip(map(image_utils.rgb_hash_fun, self.rgb_class_mapping['rgb']),
                self.rgb_class_mapping['class_ids']))
        return self

    def read(self, labels):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        labels_rgb = image_io_utils.read_image_with_number_of_channels(
            labels, 3, image_size=self.image_size, interpolation_order=0)
        class_ids = image_utils.decode_class_ids_from_rgb(
            labels_rgb, self._rgb_class_ids_mapping_hashed)
        result = {}
        if self.extract_edges:
            edges = image_utils.extract_edges_from_classes(
                labels_rgb, class_ids, self.edge_classes,
                self.edge_thickness)
            edges = np.expand_dims(edges.astype(np.uint8), -1)
            result[SegmentationDataFields.segmentation_edges] = edges
        if self.extract_class_ids:
            class_ids = np.expand_dims(
                class_ids.astype(self.segmentation_dtype), -1)
            result[SegmentationDataFields.segmentation_classes] = class_ids

        return result


# pylint: enable=too-many-instance-attributes

class SemanticSegmentationReaderTfRecords(ImageDataReaderTfRecords):
    """
    Read semantic segmentation classes from tfrecords files.

    Parameters
    ----------
    encoding
        encoding to use like "PNG", "JPEG" etc.; if not provided, assumes that
        images are stored as a raw bytes array of uint8 and are under
        `segmentation_classes`
        key inside of tfrecords file; otherwise it looks for features with
        key "segmentation_classes_{ENCODING}" inside of tfrecords files and
        decodes it
        correspondingly; in case of encoding, `image_sizes` key from the
        tfrecords file will be not used
    image_size
        size of image as  [height, width]; if specified, loaded images will be
        resized to it; if not specified, then images will have the
        original size, which will be inferred from `image_sizes` field of
        tfrecords file; if none of the are specified, runtime error of
        tensorflow will be raised.
    align_corners
        see `tf.image.resize_images`
    preserve_aspect_ratio
        see `tf.image.resize_images`

    Attributes
    ----------
    generated_keys
        * segmentation_classes : images, [width, height, num_channels], tf.uint8
        * segmentation_classes_fnames : file names of images, tf.string
    """
    generated_keys = [
        SegmentationDataFields.segmentation_classes,
        SegmentationDataFields.segmentation_classes_fnames,
    ]

    def __init__(self, **reader_kwargs):
        super().__init__(
            image_number_of_channels=1,
            interpolation_order=0,
            tfrecords_image_key=SegmentationDataFields.segmentation_classes,
            tfrecords_fname_key=
            SegmentationDataFields.segmentation_classes_fnames,
            result_dtype="uint8",
            **reader_kwargs)

    def postprocess_tfrecords(self, **image_data) -> dict:
        images_result = super().postprocess_tfrecords(**image_data)
        result = {
            SegmentationDataFields.segmentation_classes:
                images_result[ImageDataFields.images],
            SegmentationDataFields.segmentation_classes_fnames:
                images_result[ImageDataFields.images_fnames],
        }
        return result


class EdgesReaderTfRecords(ImageDataReaderTfRecords):
    """
    Read binary edges from tfrecords files.

    Parameters
    ----------
    encoding
        encoding to use like "PNG", "JPEG" etc.; if not provided, assumes that
        images are stored as a raw bytes array of uint8 and are under
        `segmentation_edges`
        key inside of tfrecords file; otherwise it looks for features with
        key "segmentation_edges_{ENCODING}" inside of tfrecords files and
        decodes it
        correspondingly; in case of encoding, `image_sizes` key from the
        tfrecords file will be not used
    image_size
        size of image as  [height, width]; if specified, loaded images will be
        resized to it; if not specified, then images will have the
        original size, which will be inferred from `image_sizes` field of
        tfrecords file; if none of the are specified, runtime error of
        tensorflow will be raised.
    result_dtype
        result dtype
    align_corners
        see `tf.image.resize_images`
    preserve_aspect_ratio
        see `tf.image.resize_images`

    Attributes
    ----------
    generated_keys
        * segmentation_edges : images, [width, height, num_channels], tf.uint8
        * segmentation_edges_fnames : file names of images, tf.string
    """
    generated_keys = [
        SegmentationDataFields.segmentation_edges,
        SegmentationDataFields.segmentation_edges_fnames,
    ]

    def __init__(self, **reader_kwargs):
        super().__init__(
            image_number_of_channels=1,
            interpolation_order=0,
            tfrecords_image_key=SegmentationDataFields.segmentation_edges,
            tfrecords_fname_key=
            SegmentationDataFields.segmentation_edges_fnames,
            result_dtype="uint8",
            **reader_kwargs)

    def postprocess_tfrecords(self, **image_data) -> dict:
        images_result = super().postprocess_tfrecords(**image_data)
        result = {
            SegmentationDataFields.segmentation_edges:
                images_result[ImageDataFields.images],
            SegmentationDataFields.segmentation_edges_fnames:
                images_result[ImageDataFields.images_fnames],
        }
        return result
