# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for IO bounding box operations
"""

from collections import OrderedDict
import json
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from nucleus7.utils import io_utils
import numpy as np

from ncgenes7.utils.object_detection_utils import get_category_index
from ncgenes7.utils.object_detection_utils import local_to_image_coordinates_np
from ncgenes7.utils.object_detection_utils import normalize_bbox_np


def maybe_create_category_index(
        num_classes: int,
        class_names_to_labels_mapping: Union[str, dict],
        class_offset: int = 1) -> Tuple[int, dict]:
    """
    Create category index from num_classes or from class_names to labels
    mapping. This is the same category index as is used in object_detection API

    Parameters
    ----------
    num_classes
        number of classes
    class_names_to_labels_mapping
        either str with file name to mapping or dict with mapping itself;
        mapping should be in format {"class name": {"class_id": 1}, },
        where class_id is an unique integer
    class_offset
        class id to start it from 0 (for classification) or 1 (for object
        detection)

    Returns
    -------
    num_classes
        number of classes
    category_index
        category index in format {"class name": {"id": , {"name": ,}}, ...}
    """
    if class_names_to_labels_mapping is not None:
        class_names_to_labels_mapping = io_utils.maybe_load_json(
            class_names_to_labels_mapping, as_ordereddict=True)
        if isinstance(class_names_to_labels_mapping, OrderedDict):
            categories = {v['class_id']: k for k, v in
                          reversed(class_names_to_labels_mapping.items())}
        else:
            categories = {v['class_id']: k for k, v in
                          class_names_to_labels_mapping.items()}
        category_index = {k: {'id': k, 'name': v}
                          for k, v in sorted(categories.items())}
    else:
        assert num_classes is not None, (
            "Either num_classes or class_names_to_labels_mapping "
            "should be provided!!!")
        category_index = get_category_index(num_classes, class_offset)
    num_classes = len(category_index)
    return num_classes, category_index


def read_objects(fname: str, image_size: Optional[list] = None,
                 normalize_bbox: bool = True,
                 with_scores=False) -> Tuple[np.ndarray, ...]:
    """
    Read labels like bounding boxes, instance_id and class_label
    from .json file.

    Objects inside of file should be formatted in following way:

    [{'bbox': {'xmin': , 'ymin': , 'w': , 'h': }, 'class_label': ,
      'id': , 'score': }, ...] or bbox can be a list in format
      [ymin, xmin, ymax, xmax]

    Parameters
    ----------
    fname
        file name with labels inside
    image_size
        image size in format height / width
    normalize_bbox
        if bounding box should be normalized with respect to image_size
    with_scores
        if detection scores should be returned

    Returns
    -------
    class_labels
        class ids with shape [num_objects]
    instance_ids
        instance ids with shape [num_objects]
    bboxes
        bounding boxes with shape [num_objects, 4] with [ymin, xmin, ymax, xmax]
        format
    scores
        only if return_scores == True; if no scores inside of labels was found,
        then it will be returned as 1
    """
    instance_ids, class_labels, bboxes, scores = _read_objects_json(
        fname, with_scores=with_scores)
    bboxes = bboxes.astype(np.float32)
    if normalize_bbox:
        bboxes = normalize_bbox_np(bboxes, image_size)
    instance_ids = instance_ids.astype(np.int64)
    class_labels = class_labels.astype(np.int64)
    bboxes = bboxes.astype(np.float32)
    if with_scores:
        scores = scores.astype(np.float32)
        return class_labels, instance_ids, bboxes, scores
    return class_labels, instance_ids, bboxes


def save_objects_to_file(
        fname: str, object_boxes: np.ndarray,
        object_classes: np.ndarray,
        object_scores: np.ndarray,
        instance_ids: Optional[np.ndarray] = None,
        image_shape: Optional[List[int]] = None,
        **additional_attributes: Optional[Dict[str, np.ndarray]]):
    """
    Save objects to json file according in following format:
    `[{'bbox': {'xmin': , 'ymin': 'xmax': , 'ymax': , 'w': , 'h': },
    'class_label': , 'id': , 'score': , **additional_attributes]`

    In case of csv files, objects will be saved in one line with
    each detection in format [instance_id, class, score, x1, y1, w, h] + [...].
    File will have one line header.

    Parameters
    ----------
    fname
        file name to save
    object_boxes
        bounding boxes coordinates in format [y1, x1, ymax, xmax]
    object_classes
        object classes
    object_scores
        object scores
    instance_ids
        object instance ids
    image_shape
        if not provided, objects will be saved in local coordinates
    additional_attributes
        additional object attributes to save in json file, which must be dict
        with mapping of str keys to numpy arrays of shape [num_objects, ...]
    """
    # pylint: disable=too-many-arguments,too-many-locals
    # TODO(oleksandr.vorobiov@audi.de): refactor
    if image_shape is not None:
        image_size = image_shape[:2]
        object_boxes = local_to_image_coordinates_np(object_boxes, image_size)
    # pylint: disable=unbalanced-tuple-unpacking
    ymin, xmin, ymax, xmax = np.split(object_boxes, 4, axis=-1)
    # pylint: enable=unbalanced-tuple-unpacking
    mask = np.squeeze(np.logical_and((ymax - ymin > 0), (xmax - xmin > 0)), -1)
    object_boxes = object_boxes[mask]
    object_classes = object_classes[mask]
    object_scores = object_scores[mask]
    if instance_ids is not None:
        instance_ids = instance_ids[mask]
    if additional_attributes:
        for key in additional_attributes:
            additional_attributes[key] = additional_attributes[key][mask]

    _save_objects_to_json(
        fname=fname,
        object_boxes=object_boxes,
        object_classes=object_classes,
        object_scores=object_scores,
        instance_ids=instance_ids,
        **additional_attributes
    )


def _read_objects_json(fname, with_scores=False):
    """
    Read objects from json file

    Objects inside of file should be formatted in following way:

    `[{'bbox': {'xmin': , 'ymin': , 'w': , 'h': }, 'class_label': ,
    'id': , 'score': }, ...]` or bbox can be a list in format
    [ymin, xmin, ymax, xmax]

    Parameters
    ----------
    fname
        file name
    with_scores
        if scores should be read from file

    Returns
    -------
    class_labels
        class ids with shape [num_objects]
    instance_ids
        instance ids with shape [num_objects]
    bboxes
        bounding boxes with shape [num_objects, 4] and format
        [ymin, xmin, ymax, xmax]
    scores
        only if return_scores == True; if no scores inside of labels was found,
        then it will be returned as 1

    """
    try:
        data = io_utils.load_json(fname)
    except:  # pylint: disable=bare-except
        data = []
    instance_ids, class_labels, bboxes, scores = (
        _combine_object_labels(data, with_scores=with_scores))
    return instance_ids, class_labels, bboxes, scores


def _save_objects_to_json(
        fname: str,
        object_boxes: np.ndarray,
        object_classes: np.ndarray,
        object_scores: np.ndarray,
        instance_ids: Optional[np.ndarray] = None,
        **additional_attributes: Optional[Dict[str, np.ndarray]]):
    """
    Save objects to json file as list of objects like
    [{'bbox': {'xmin': , 'ymin': 'xmax': , 'ymax': , 'w': , 'h': },
      'class_label': ,
      'id': , 'score': ,  ...]

    Parameters
    ----------
    fname
        file name to save
    object_boxes
        coordinates in format [y1, x1, ymax, xmax]
    object_classes
        object classes
    object_scores
        object scores
    instance_ids
        object instance ids
    additional_attributes
        additional object attributes to save in json file, which must be dict
        with mapping of str keys to numpy arrays of shape [num_objects, ...]
    """
    # pylint: disable=too-many-arguments,too-many-locals
    # TODO(oleksandr.vorobiov@audi.de): refactor
    if instance_ids is None:
        instance_ids = np.arange(object_classes.shape[0]) + 1
    data_json = []
    num_detections = instance_ids.shape[0]
    for i in range(num_detections):
        class_label = int(object_classes[i])
        if class_label == 0:
            continue
        bbox_dict = _get_bbox_dict(object_boxes[i])
        detection_data = {'bbox': bbox_dict,
                          'class_label': class_label,
                          'id': int(instance_ids[i]),
                          'score': float(object_scores[i])}
        if additional_attributes:
            for key in additional_attributes:
                detection_data[key] = additional_attributes[key][i].tolist()
        data_json.append(detection_data)
    with open(fname, 'w', encoding='utf-8') as file:
        data = json.dumps(data_json, ensure_ascii=False,
                          indent=4, sort_keys=True)
        file.write(data)


def _get_bbox_dict(bbox):
    ymin, xmin, ymax, xmax = bbox.tolist()
    width = xmax - xmin
    height = ymax - ymin
    box_dict = {'xmin': float(xmin),
                'ymin': float(ymin),
                'xmax': float(xmax),
                'ymax': float(ymax),
                'w': float(width),
                'h': float(height)}
    return box_dict


def _combine_object_labels(data: List[dict], with_scores=False):
    if not data:
        instance_ids = np.zeros([0])
        class_labels = np.zeros([0])
        bboxes = np.zeros([0, 4])
        if with_scores:
            scores = np.ones([0])
        else:
            scores = None
        return instance_ids, class_labels, bboxes, scores
    class_labels = np.stack([d['class_label'] for d in data], -1)
    instance_ids = np.stack([d['id'] for d in data], -1)
    if with_scores:
        scores = np.stack([d.get('score', d.get('object_score', 1.0))
                           for d in data], -1)
    else:
        scores = None

    def _get_coord_from_box(box, coord_name):
        # pylint: disable=too-many-return-statements
        if isinstance(box, dict):
            try:
                return box[coord_name]
            except KeyError:
                if coord_name == "w":
                    return box["xmax"] - box["xmin"]
                return box["ymax"] - box["ymin"]
        if coord_name == 'ymin':
            return box[0]
        if coord_name == 'xmin':
            return box[1]
        if coord_name == 'h':
            return box[2] - box[0]
        if coord_name == 'w':
            return box[3] - box[1]
        raise ValueError(
            "coordinate name {} not understood!".format(coord_name))

    xmin = [_get_coord_from_box(d['bbox'], 'xmin') for d in data]
    ymin = [_get_coord_from_box(d['bbox'], 'ymin') for d in data]
    xmax = [_get_coord_from_box(d['bbox'], 'xmin') +
            _get_coord_from_box(d['bbox'], 'w')
            for d in data]
    ymax = [_get_coord_from_box(d['bbox'], 'ymin') +
            _get_coord_from_box(d['bbox'], 'h')
            for d in data]
    bboxes = np.stack([ymin, xmin, ymax, xmax], -1)
    return instance_ids, class_labels, bboxes, scores
