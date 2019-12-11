# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for coco dataset
"""
import os
from typing import Optional
from typing import Union

from nucleus7.utils import io_utils
import numpy as np

from ncgenes7.utils.image_io_utils import read_image_with_number_of_channels
from ncgenes7.utils.image_utils import decode_class_ids_from_rgb


def panoptic_categories_to_rgb_hash_fn(fname_panoptic_annotations: str,
                                       remove_unused_classes: bool = False):
    """
    Read panoptic annotation file and create mapping in the form:
        {file_name: {id: class label}}

    Inside of panoptic_annotations['segments_info'] all of segments have its
    category (coco class) and its id, where id = r + g*256 + b*256**2 uses
    as a hash to map rgb values on panoptic rgb images to its segment label.

    Parameters
    ----------
    fname_panoptic_annotations
        file name of panoptic annotations json file
    remove_unused_classes
        if the unused classes must be removed and so at the end there
        are 133 class ids instead of 200

    Returns
    -------
    panoptic_image_fname_to_class_id_hash_fn
        mapping of type {file_name: {id: class label}}
    """
    panoptic_coco = io_utils.load_json(fname_panoptic_annotations)
    panoptic_coco_categories = panoptic_coco['categories']
    class_ids_to_name_coco = {p['id']: p['name']
                              for p in panoptic_coco_categories}
    if remove_unused_classes:
        _, class_ids_mapping, _ = _remove_empty_classes(
            class_ids_to_name_coco)
    else:
        class_ids_mapping = dict(zip(*[class_ids_to_name_coco] * 2))

    panoptic_image_fname_to_class_id_hash_fn = {}
    for ann in panoptic_coco['annotations']:
        file_name = ann['file_name']
        hash_to_class = {a['id']: class_ids_mapping[a['category_id']]
                         for a in ann['segments_info']}
        hash_to_class[0] = 0
        panoptic_image_fname_to_class_id_hash_fn[file_name] = hash_to_class

    return panoptic_image_fname_to_class_id_hash_fn


def get_class_descriptions_mapping(fname_annotations: str,
                                   remove_unused_classes: bool = False,
                                   sort_objects_by_class_and_id: bool = True,
                                   with_keypoints: bool = False):
    """
    Create mapping of type
    {image_fname:
    [{class_label: , id: , bbox: [ymin, xmin, ymax, xmax]}, {...}]}

    Parameters
    ----------
    fname_annotations
        file name of instances.json
    remove_unused_classes
        if the unused classes must be removed and so at the end there
        are 80 class ids instead of 90
    sort_objects_by_class_and_id
        if the result objects should be sorted first by class id and then by
        instance id
    with_keypoints
        if the keypoints data should be read from annotations; must be used
        only if fname_annotations points to person keypoints annotation file

    Returns
    -------
    image_fname_to_objects
        mapping of type
         {image_fname:
         [{class_label: , id: , bbox: [ymin, xmin, ymax, xmax]}, {...}]}
         with normalized to particular image shape bounding box coordinates
    """
    instances_coco = io_utils.load_json(fname_annotations)
    class_ids_coco = {p['id']: p['name'] for p in instances_coco['categories']}
    if remove_unused_classes:
        _, class_ids_mapping, _ = _remove_empty_classes(class_ids_coco)
    else:
        class_ids_mapping = dict(zip(*[class_ids_coco] * 2))

    image_fname_to_image_size = {p['file_name']: {'width': p['width'],
                                                  'height': p['height']}
                                 for p in instances_coco['images']}
    image_id_to_fname = {p['id']: p['file_name']
                         for p in instances_coco['images']}
    image_fname_to_objects = {}
    for ann in instances_coco['annotations']:
        image_fname = image_id_to_fname[ann['image_id']]
        image_size = image_fname_to_image_size[image_fname]
        object_data = _read_object_from_annotation(
            ann, image_size, class_ids_mapping,
            with_keypoints=with_keypoints)
        image_fname_to_objects.setdefault(image_fname, [])
        image_fname_to_objects[image_fname].append(object_data)
    if sort_objects_by_class_and_id:
        image_fname_to_objects = _sort_instances_by_attributes(
            image_fname_to_objects)
    return image_fname_to_objects


def read_objects_from_fname(image_id: str,
                            image_fname_to_objects: dict) -> tuple:
    """
    Read objects from image_fname_to_objects mapping under corresponding
    image file name

    Parameters
    ----------
    image_id
        image id to read the objects for
    image_fname_to_objects
        mapping of type
         {image_fname:
         [{class_label: , id: , bbox: [ymin, xmin, ymax, xmax]}, {...}]}
         with normalized to particular image shape bounding box coordinates

    Returns
    -------
    bboxes
        bounding boxes with shape [num_objects, 4] and normalized coordinates in
        format [ymin, xmin, ymax, xmax]
    class_labels
        1-based classes for objects with shape [num_objects]
    instnace_ids
        instance ids with shape [num_objects]
    """
    if (image_id not in image_fname_to_objects or
            not image_fname_to_objects[image_id]):
        instance_ids = np.zeros([0], np.int64)
        class_labels = np.zeros([0], np.int64)
        bboxes = np.zeros([0, 4], np.float32)
    else:
        labels = image_fname_to_objects[image_id]
        class_labels = np.stack([l['class_label']
                                 for l in labels], 0).astype(np.int64)
        bboxes = np.stack([l['bbox'] for l in labels], 0).astype(np.float32)
        instance_ids = np.stack(
            [l['id'] for l in labels], 0).astype(np.int64)

    return bboxes, class_labels, instance_ids


def read_segmentation_from_fname(panoptic_fname: str,
                                 panoptic_image_fname_to_class_id_hash_fn,
                                 image_size: Optional[list] = None,
                                 dtype="uint8",
                                 ) -> np.ndarray:
    """
    Read segmentation labels from file name and panoptic coco labels

    Parameters
    ----------
    panoptic_fname
        file name with panoptic image
    panoptic_image_fname_to_class_id_hash_fn
        mapping of type {file_name: {id: class label}}
    image_size
        image size if labels should be resized to; if not specified, labels will
        be used as is
    dtype
        dtype for the result

    Returns
    -------
    segmentation_classes
        segmentation classes
    """

    def _hash_fun_coco(rgb, nd=True):
        # pylint: disable=unused-argument,invalid-name
        # this signature is needed by the caller
        # pylint: disable=unbalanced-tuple-unpacking
        r, g, b = np.split(rgb, 3, axis=-1)
        return r + g * 256 + b * (256 ** 2)

    basename = os.path.basename(panoptic_fname)
    rgb_class_ids_mapping_hashed = (
        panoptic_image_fname_to_class_id_hash_fn[basename])
    segmentation_img_rgb = read_image_with_number_of_channels(
        panoptic_fname, 3, image_size=image_size, interpolation_order=0)
    segmentation_classes = decode_class_ids_from_rgb(
        segmentation_img_rgb, rgb_class_ids_mapping_hashed,
        hash_fun=_hash_fun_coco)
    segmentation_classes = segmentation_classes.astype(dtype)
    return segmentation_classes


def read_keypoints_from_fname(image_id: str,
                              image_fname_to_objects: dict) -> tuple:
    """
    Read keypoints from image_fname_to_objects mapping under corresponding
    image file name

    Parameters
    ----------
    image_id
        image id to read the objects for
    image_fname_to_objects
        mapping of type
         {image_fname:
         [{keypoints: [17, 2], keypoints_visibilities: ]}
         with normalized to particular image shape bounding box coordinates

    Returns
    -------
    keypoints
        keypoints with shape [num_objects, 17, 24] and normalized coordinates in
        format [y, x]
    keypoints_visibilities
        visibilities of keypoints as [num_objects, 17]
    """
    if (image_id not in image_fname_to_objects or
            not image_fname_to_objects[image_id]):
        keypoints = np.zeros([0], np.float32)
        keypoints_visibilities = np.zeros([0], np.int32)
        return keypoints, keypoints_visibilities

    labels = image_fname_to_objects[image_id]
    keypoints = np.stack([each_label["keypoints"] for each_label in labels], 0
                         ).astype(np.float32)
    keypoints_visibilities = np.stack(
        [each_label["keypoints_visibilities"] for each_label in labels], 0
    ).astype(np.int32)
    return keypoints, keypoints_visibilities


def _xywh_to_yxyx(bbox, image_width, image_height):
    """
    coco uses other [xmin, ymin, width, height] to store bounding box
    coordinates. Because of convension used inside of ncgenes7, this
    coordinates should be in format [ymin, xmin, ymax, xmax] and be
    normalized to image size

    Parameters
    ----------
    bbox
        absolute coordinates in format [xmin, ymin, width, height]
    image_width
        image width
    image_height
        image height

    Returns
    -------
    box
        normalized coordinates in format [ymin, xmin, ymax, xmax]
    """
    xmin, ymin, width, height = bbox
    xmin = xmin / image_width
    ymin = ymin / image_height
    width = width / image_width
    height = height / image_height
    xmax = xmin + width
    ymax = ymin + height
    bbox = [ymin, xmin, ymax, xmax]
    return bbox


def _remove_empty_classes(class_ids_with_empty: dict):
    """
    In coco dataset there are empty classes, e.g. 90 class definitions for
    object detection and only 80 of them are inside (same for panoptic classes -
    from 200 only 133 are usable)

    Parameters
    ----------
    class_ids_with_empty
        keys are coco class ids

    Returns
    -------
    class_ids_without_empty
        dict mapping new ranged class ids to same values from
        class_ids_with_empty
    class_ids_to_non_empty
        mapping from original classes to new ranged
    class_ids_non_empty_to_coco
        mapping from new classes to original coco classes
    """
    class_ids = sorted(class_ids_with_empty)
    class_ids_without_empty = list(range(1, len(class_ids) + 1))
    class_ids_to_non_empty = dict(zip(class_ids, class_ids_without_empty))
    class_ids_non_empty_to_coco = dict(zip(class_ids_without_empty, class_ids))
    class_ids_without_empty = {class_ids_to_non_empty[cl]: v
                               for cl, v in class_ids_with_empty.items()}
    return (class_ids_without_empty,
            class_ids_to_non_empty,
            class_ids_non_empty_to_coco)


def _read_object_from_annotation(object_annotation, image_size,
                                 class_ids_mapping,
                                 with_keypoints=False):
    image_height = image_size['height']
    image_width = image_size['width']
    bbox = object_annotation['bbox']
    bbox = _xywh_to_yxyx(bbox, image_width, image_height)
    class_label = class_ids_mapping[object_annotation['category_id']]
    object_data = {'class_label': class_label,
                   'id': object_annotation['id'],
                   'bbox': bbox}
    if with_keypoints:
        keypoints_raw = object_annotation["keypoints"]
        keypoints_reshaped = np.reshape(keypoints_raw, [17, 3])
        keypoints_visibilities = keypoints_reshaped[:, -1].astype(np.int32)
        keypoints_coord = _keypoints_xy_to_yx_norm(keypoints_reshaped[:, :2],
                                                   image_width, image_height)
        object_data["keypoints"] = keypoints_coord
        object_data["keypoints_visibilities"] = keypoints_visibilities
    return object_data


def _keypoints_xy_to_yx_norm(keypoints, image_width, image_height):
    point_x, point_y = keypoints.transpose()
    point_x = point_x / image_width
    point_y = point_y / image_height
    return np.stack([point_y, point_x], -1)


def _sort_instances_by_attributes(
        image_ids_to_data,
        attribute_names: Union[list, tuple] = ("class_label", "id")) -> dict:
    image_ids_to_data_sorted = {}
    for each_key, each_data in image_ids_to_data.items():
        image_ids_to_data_sorted[each_key] = sorted(
            each_data, key=lambda x: [x[each_attr_name]
                                      for each_attr_name in attribute_names])
    return image_ids_to_data_sorted
