# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Data readers for object detection
"""

import os
from typing import Dict
from typing import Optional
from typing import Union

import nucleus7 as nc7
import numpy as np
import tensorflow as tf

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.data_fields.object_detection import ObjectDataFields
from ncgenes7.utils import object_detection_io_utils as od_io_utils
from ncgenes7.utils import object_detection_utils


class ObjectDetectionReader(nc7.data.DataReader):
    """
    Reader to read the objects from json files in following format:

    * [{'bbox': {'xmin': , 'ymin': , 'w': , 'h': }, 'class_label': ,
      'id': , 'score': }, ...]

    * or bbox can be a list in format
      [ymin, xmin, ymax, xmax] or in format
      {'xmin': , 'ymin': , 'xmax': , 'ymax': }

    Boxes can be both - normalized and not. But you can normalize them by
    setting normalize_boxes = True if you want to normalize image coordinates.
    Also you need to provide the image size for it.

    Parameters
    ----------
    image_size
        image size as [height, width] in pixels; only needed if you want to
        normalize boxes
    normalize_boxes
        if the object boxes must be normalized using image_size

    Attributes
    ----------
    generated_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [bs, num_objects, 4]
          and with values in [0, 1]; np.float32
        * object_classes : classes for objects, 1-based,
          [bs, num_objects]; np.int64
        * object_instance_ids : instance ids for objects,
          [bs, num_objects]; == 0 if no id was found; np.int64
        * object_scores : object scores if they were found in the json file and
          np.ones of object_classes size; np.float32
        * object_fnames : file names, e.g. some identifier that was stored
          inside of tfrecords; [bs], str
        * num_objects : number of objects, which is inferred from the shape of
          object_boxes; [bs], np.int64
    """
    file_list_keys = [
        "labels",
    ]
    generated_keys = [
        ObjectDataFields.object_boxes,
        ObjectDataFields.object_classes,
        ObjectDataFields.object_instance_ids,
        ObjectDataFields.object_scores,
        ObjectDataFields.object_fnames,
        ObjectDataFields.num_objects,
    ]

    def __init__(self, *,
                 image_size: Optional[list] = None,
                 normalize_boxes: Optional[bool] = False,
                 **reader_kwargs):
        if normalize_boxes and not image_size:
            raise ValueError("Provide image_size to normalize boxes!")
        super().__init__(**reader_kwargs)
        self.image_size = image_size
        self.normalize_boxes = normalize_boxes

    def read(self, *, labels):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        result = self._read_objects_from_fname(labels)
        return result

    def _read_objects_from_fname(self, labels: str) -> dict:
        (object_classes, object_instance_ids, object_boxes, object_scores
         ) = od_io_utils.read_objects(labels,
                                      image_size=self.image_size,
                                      normalize_bbox=self.normalize_boxes,
                                      with_scores=True)
        num_objects = object_boxes.shape[0]
        labels_basename = os.path.basename(labels)
        result = {
            ObjectDataFields.object_classes: object_classes,
            ObjectDataFields.object_instance_ids: object_instance_ids,
            ObjectDataFields.object_boxes: object_boxes,
            ObjectDataFields.object_scores: object_scores,
            ObjectDataFields.num_objects: num_objects,
            ObjectDataFields.object_fnames: labels_basename,
        }
        return result


class ObjectDetectionReaderTF(ObjectDetectionReader):
    """
    Read objects from json files to tensorflow in following format:

    * [{'bbox': {'xmin': , 'ymin': , 'w': , 'h': }, 'class_label': ,
      'id': , 'score': }, ...]

    * or bbox can be a list in format
      [ymin, xmin, ymax, xmax] or in format
      {'xmin': , 'ymin': , 'xmax': , 'ymax': }

    Boxes can be both - normalized and not. But you can normalize them by
    setting normalize_boxes = True if you want to normalize image coordinates.
    Also you need to provide the image size for it.

    Parameters
    ----------
    image_size
        image size as [height, width] in pixels; only needed if you want to
        normalize boxes
    normalize_boxes
        if the object boxes must be normalized using image_size

    Attributes
    ----------
    generated_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [bs, num_objects, 4]
          and with values in [0, 1]; tf.float32
        * object_classes : classes for objects, 1-based,
          [bs, num_objects]; tf.int64
        * object_instance_ids : instance ids for objects,
          [bs, num_objects]; == 0 if no id was found; tf.int64
        * object_scores : object scores from tfrecords;
          == -1 if no score was found;
          [bs, num_objects]; tf.float32
        * object_fnames : file names, e.g. some identifier that was stored
          inside of tfrecords; [bs], tf.string
        * num_objects : number of objects, which is inferred from the shape of
          object_boxes; [bs], tf.int64
    """
    is_tensorflow = True

    def read(self, *, labels):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        result = self._read_objects_tf(labels)
        return result

    def _read_objects_tf(self, labels):
        result_dtypes = {
            ObjectDataFields.object_classes: tf.int64,
            ObjectDataFields.object_instance_ids: tf.int64,
            ObjectDataFields.object_boxes: tf.float32,
            ObjectDataFields.object_scores: tf.float32,
            ObjectDataFields.num_objects: tf.int64,
            ObjectDataFields.object_fnames: tf.string,
        }
        result_shapes = {
            ObjectDataFields.object_classes: [None],
            ObjectDataFields.object_instance_ids: [None],
            ObjectDataFields.object_boxes: [None, 4],
            ObjectDataFields.object_scores: [None],
            ObjectDataFields.num_objects: [],
            ObjectDataFields.object_fnames: [],
        }
        result_keys_sorted = sorted(self.generated_keys_all)

        def _read_np(labels_):
            labels_ = labels_.decode()
            result_as_dict_ = super(ObjectDetectionReaderTF, self).read(
                labels=labels_)
            result_as_list_ = [
                result_as_dict_[each_key] for each_key in result_keys_sorted]
            return result_as_list_

        result_dtypes_as_list = [
            result_dtypes[each_key] for each_key in result_keys_sorted]

        result_as_list = tf.py_func(
            _read_np, [labels], result_dtypes_as_list)
        result = dict(zip(result_keys_sorted, result_as_list))
        for each_key, each_tensor in result.items():
            each_tensor.set_shape(result_shapes[each_key])
        return result


class ObjectDetectionReaderTfRecords(nc7.data.TfRecordsDataReader):
    """
    Reader to read the objects from tfrecords files

    tfrecords files should have following content:

    - object_boxes - bounding boxes in format [ymin, xmin, ymax, xmax], and
      shape [None, 4] of type tf.float32; can be both - normalized and not,
      and will be passes as is
    - object_classes - object classes of shape [None] of tf.int64 type
    - object_scores - scores of shape [None] of tf.float32 type
    - object_instance_ids - instance ids of shape [None] of tf.int64 type
    - object_fnames - optional file names of objects, e.g. some identifier;
      [], tf.string
    - image_sizes - if coordinate frame of boxes should be modified, e.g.
      normalized or converted to image frame; [height, width], tf.int32

    Parameters
    ----------
    normalize_boxes
        if the object boxes must be normalized using `image_sizes` key
        from tfrecord files
    convert_boxes_to_image_frame
        if the boxes should be converted to image coordinates using
        `image_sizes` key from tfrecord files; e.g. inverse
        of normalize boxes

    Attributes
    ----------
    generated_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [bs, num_objects, 4]
          and with values in [0, 1]; tf.float32
        * object_classes : classes for objects, 1-based,
          == 0 if no class was found;
          [bs, num_objects]; tf.int64
        * object_instance_ids : instance ids for objects,
          == -1 if no id was found;
          [bs, num_objects]; tf.int64
        * object_scores : object scores from tfrecords;
          == -1 if no score was found;
          [bs, num_objects]; tf.float32
        * object_fnames : file names, e.g. some identifier that was stored
          inside of tfrecords; [bs], tf.string
        * num_objects : number of objects, which is inferred from the shape of
          object_boxes; [bs], tf.int64
    """
    generated_keys = [
        ObjectDataFields.object_boxes,
        ObjectDataFields.object_classes,
        ObjectDataFields.object_instance_ids,
        ObjectDataFields.object_scores,
        ObjectDataFields.num_objects,
        ObjectDataFields.object_fnames,
    ]

    def __init__(self, *,
                 normalize_boxes: bool = False,
                 convert_boxes_to_image_frame: bool = False,
                 **reader_kwargs):
        super().__init__(**reader_kwargs)
        if normalize_boxes and convert_boxes_to_image_frame:
            raise ValueError(
                "{}: Either normalize or convert to image frame or none of "
                "them!".format(self.name))
        self.normalize_boxes = normalize_boxes
        self.convert_boxes_to_image_frame = convert_boxes_to_image_frame

    def get_tfrecords_features(self) -> dict:
        object_classes_default = np.zeros([0], np.int64).tostring()
        object_instance_ids_default = np.zeros([0], np.int64).tostring()
        object_boxes_default = np.zeros([0, 4], np.float32).tostring()
        object_fnames_default = "no_file_name"
        object_scores_default = np.zeros([0], np.float32).tostring()
        image_sizes_default = np.zeros([2], np.int32).tostring()
        features = {
            ObjectDataFields.object_classes:
                tf.FixedLenFeature((), tf.string, object_classes_default),
            ObjectDataFields.object_instance_ids:
                tf.FixedLenFeature((), tf.string, object_instance_ids_default),
            ObjectDataFields.object_boxes:
                tf.FixedLenFeature((), tf.string, object_boxes_default),
            ObjectDataFields.object_scores:
                tf.FixedLenFeature((), tf.string, object_scores_default),
            ObjectDataFields.object_fnames:
                tf.FixedLenFeature((), tf.string, object_fnames_default),
            ImageDataFields.image_sizes:
                tf.FixedLenFeature((), tf.string, image_sizes_default),
        }
        return features

    def postprocess_tfrecords(self, *,
                              object_classes: tf.Tensor,
                              object_instance_ids: tf.Tensor,
                              object_boxes: tf.Tensor,
                              object_scores: tf.Tensor,
                              object_fnames: tf.Tensor,
                              image_sizes: tf.Tensor) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base method has more generic signature
        object_boxes = tf.reshape(object_boxes, [-1, 4])
        num_objects = tf.cast(tf.shape(object_boxes)[0], tf.int64)
        object_classes = _pad_attributes_to_num_objects(
            tf.reshape(object_classes, [-1]), num_objects, 0)
        object_instance_ids = _pad_attributes_to_num_objects(
            tf.reshape(object_instance_ids, [-1]), num_objects, -1)
        object_scores = _pad_attributes_to_num_objects(
            tf.reshape(object_scores, [-1]), num_objects, -1)

        if self.normalize_boxes:
            object_boxes = object_detection_utils.normalize_bbox(
                object_boxes, tf.cast(image_sizes, tf.float32))
        if self.convert_boxes_to_image_frame:
            object_boxes = object_detection_utils.local_to_image_coordinates(
                object_boxes, tf.cast(image_sizes, tf.float32))

        result = {
            ObjectDataFields.object_boxes: object_boxes,
            ObjectDataFields.object_classes: object_classes,
            ObjectDataFields.object_instance_ids: object_instance_ids,
            ObjectDataFields.object_scores: object_scores,
            ObjectDataFields.num_objects: num_objects,
            ObjectDataFields.object_fnames: object_fnames,
        }
        return result

    def get_tfrecords_output_types(self) -> dict:
        output_types = {ObjectDataFields.object_classes: tf.int64,
                        ObjectDataFields.object_instance_ids: tf.int64,
                        ObjectDataFields.object_boxes: tf.float32,
                        ObjectDataFields.object_scores: tf.float32,
                        ImageDataFields.image_sizes: tf.int32}
        return output_types


class KeypointsReaderTfRecords(nc7.data.TfRecordsDataReader):
    """
    Reader to read the keypoints from tfrecords files

    Parameters
    ----------
    num_keypoints
        number of keypoints

    tfrecords files should have following content:
    - object_keypoints - object keypoints coordinates
      shape [num_objects, num_keypoints, 2] of type tf.float32;
      can be both - normalized and not, and will be passed as is
    - object_keypoints_visibilities - visibilities of keypoints with shape
      [num_objects, num_keypoints],
      where 0 - not visible keypoint, 1 - not visible but annotated and
      2 - visible and annotated; tf.int32

    Attributes
    ----------
    generated_keys
        * object_keypoints : object keypoints coordinates
          shape [num_objects, num_keypoints, 2]; tf.float32
        * object_keypoints_visibilities : visibilities of keypoints with shape
          [num_objects, num_keypoints]; tf.int32
    """
    generated_keys = [
        ObjectDataFields.object_keypoints,
        ObjectDataFields.object_keypoints_visibilities,
    ]

    def __init__(self, *,
                 num_keypoints: int,
                 **reader_kwargs):
        super().__init__(**reader_kwargs)
        self.num_keypoints = num_keypoints

    def get_tfrecords_features(self) -> dict:
        object_keypoints_default = np.zeros(
            [0, self.num_keypoints, 2], np.float32).tostring()
        object_visibilities_default = np.zeros(
            [0, self.num_keypoints], np.int32).tostring()

        features = {
            ObjectDataFields.object_keypoints:
                tf.FixedLenFeature((), tf.string, object_keypoints_default),
            ObjectDataFields.object_keypoints_visibilities:
                tf.FixedLenFeature((), tf.string, object_visibilities_default),
        }
        return features

    def postprocess_tfrecords(self, *,
                              object_keypoints: tf.Tensor,
                              object_keypoints_visibilities: tf.Tensor
                              ) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base method has more generic signature
        object_keypoints = tf.reshape(
            object_keypoints, [-1, self.num_keypoints, 2])
        num_objects = tf.cast(tf.shape(object_keypoints)[0], tf.int64)
        object_keypoints_visibilities = _pad_attributes_to_num_objects(
            tf.reshape(object_keypoints_visibilities, [-1, self.num_keypoints]),
            num_objects=num_objects, pad_value=2)

        result = {
            ObjectDataFields.object_keypoints: object_keypoints,
            ObjectDataFields.object_keypoints_visibilities:
                object_keypoints_visibilities,
        }
        return result

    def get_tfrecords_output_types(self) -> dict:
        output_types = {
            ObjectDataFields.object_keypoints: tf.float32,
            ObjectDataFields.object_keypoints_visibilities: tf.int32,
        }
        return output_types


class ObjectClassSelectorTf(nc7.data.DataProcessor):
    """
    Processor that selects only particular classes and masks all other inputs
    accordingly

    Parameters
    ----------
    classes_to_select
        single class or a list of classes to select

    Attributes
    ----------
    incoming_keys
        * object_classes : classes for objects, 1-based,
          == 0 if no class was found;
          [num_objects]; tf.int64
    generated_keys
        * object_classes : classes for objects, 1-based,
          == 0 if no class was found;
          [num_objects]; tf.int same as incoming
        * num_objects : number of objects, which is inferred from the shape of
          object_boxes; [], tf.int32
    """
    is_tensorflow = True
    incoming_keys = [
        ObjectDataFields.object_classes,
    ]
    generated_keys = [
        ObjectDataFields.object_classes,
        ObjectDataFields.num_objects,
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

    def process(self, object_classes: tf.Tensor,
                **dynamic_inputs) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base method has more generic signature
        _ = dynamic_inputs.pop(ObjectDataFields.num_objects, None)
        classes_mask, num_objects = self._create_classes_mask(object_classes)
        result = object_detection_utils.mask_inputs_to_classes(
            object_classes, dynamic_inputs,
            classes_mask,
            tf.boolean_mask,
        )
        result[ObjectDataFields.num_objects] = num_objects
        return result

    def _create_classes_mask(self, object_classes):
        classes_mask, num_objects = object_detection_utils.create_classes_mask(
            object_classes, self.classes_to_select)
        return classes_mask, num_objects


class BoxAdjusterByKeypoints(nc7.data.DataProcessor):
    """
    Resize boxes so that all keypoints lie inside of new bounding box

    If no boxes provided, will create them from keypoints. Otherwise will
    adjust the boxes to include all keypoints

    Attributes
    ----------
    incoming_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [bs, num_objects, 4]
          and with values in [0, 1]; tf.float32
        * object_keypoints : object keypoints coordinates with normalized to
          image coordinates in format [y, x],
          shape [num_objects, num_keypoints, 2]; tf.float32
    generated_keys
        * object_boxes : object boxes with normalized to image coordinates
          in format [ymin, xmin, ymax, xmax], shape [bs, num_objects, 4]
          and with values in [0, 1]; tf.float32
    """
    is_tensorflow = True
    incoming_keys = [
        ObjectDataFields.object_keypoints,
        "_" + ObjectDataFields.object_boxes,
    ]
    generated_keys = [
        ObjectDataFields.object_boxes,
    ]

    def process(self, object_keypoints: tf.Tensor,
                object_boxes: Optional[tf.Tensor]) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base method has more generic signature
        keypoints_mask = tf.greater(object_keypoints, 0)
        objects_with_keypoints_mask = tf.greater(
            tf.reduce_sum(tf.cast(keypoints_mask, tf.int32), [-1, -2]),
            0)
        boxes_from_keypoints = tf.concat(
            [tf.reduce_min(tf.where(keypoints_mask,
                                    object_keypoints,
                                    tf.fill(tf.shape(object_keypoints),
                                            tf.float32.max)),
                           -2),
             tf.reduce_max(tf.where(keypoints_mask,
                                    object_keypoints,
                                    tf.fill(tf.shape(object_keypoints),
                                            tf.float32.min)),
                           -2)], -1)
        boxes_from_keypoints = tf.where(objects_with_keypoints_mask,
                                        boxes_from_keypoints,
                                        tf.zeros_like(boxes_from_keypoints))
        if object_boxes is None:
            return {
                ObjectDataFields.object_boxes: boxes_from_keypoints
            }

        boxes_from_keypoints_xy_min = tf.where(
            objects_with_keypoints_mask,
            boxes_from_keypoints[..., :2],
            tf.fill(tf.shape(boxes_from_keypoints[..., :2]), tf.float32.max)
        )
        boxes_from_keypoints_xy_max = tf.where(
            objects_with_keypoints_mask,
            boxes_from_keypoints[..., 2:],
            tf.fill(tf.shape(boxes_from_keypoints[..., 2:]), tf.float32.min)
        )

        boxes_adjusted = tf.concat(
            [tf.minimum(boxes_from_keypoints_xy_min, object_boxes[..., :2]),
             tf.maximum(boxes_from_keypoints_xy_max, object_boxes[..., 2:])],
            -1)
        return {
            ObjectDataFields.object_boxes: boxes_adjusted
        }


def _pad_attributes_to_num_objects(data, num_objects, pad_value=-1):
    pad_length = num_objects - tf.shape(data, out_type=tf.int64)[0]
    data_ndim = len(data.shape)
    paddings = [[0, pad_length], *[[0, 0]] * (data_ndim - 1)]
    data_padded = tf.pad(data, paddings, constant_values=pad_value)
    return data_padded
