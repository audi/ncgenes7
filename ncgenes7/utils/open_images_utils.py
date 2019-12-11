# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for open images dataset
"""

import logging
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd


def get_class_descriptions_mapping(fname_class_descriptions):
    """
    Create mapping {class_hash}: (class_id, class_name); the class ids is a
    range for sorted class names

    Parameters
    ----------
    fname_class_descriptions
        file name with class_descriptions.csv

    Returns
    -------
    class_descriptions
        mapping of class hash string to tuple of (class id, class name)
    """
    class_descriptions = pd.read_csv(fname_class_descriptions, header=None)
    class_descriptions = {
        tag: (i + 1, each_class)
        for i, (each_class, tag) in enumerate(
            sorted(zip(class_descriptions[1], class_descriptions[0])))}
    return class_descriptions


def get_annotation_mapping(fname_annotations: str) -> pd.DataFrame:
    """
    Read annotations

    Parameters
    ----------
    fname_annotations
        file name of annotations.csv

    Returns
    -------
    annotation
        data frame of annotations with ImageID as index
    """
    logger = logging.getLogger(__name__)
    logger.info("Read annotations to memory")
    annotations = pd.read_csv(fname_annotations)
    annotations = annotations.set_index("ImageID").sort_index()
    return annotations


def query_labels_by_image_id(image_id: Union[str, bytes],
                             annotations: pd.DataFrame,
                             class_descriptions: dict,
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Query labels from annotations by image id. If image id is not inside of
    annotations data frame, it will return zero boxes and class labels,
    otherwise will return all labels forund for this image id

    Parameters
    ----------
    image_id
        image id to query the labels
    annotations
        open images annotations
    class_descriptions
        mapping of class hash to class id and class name

    Returns
    -------
    class_labels
        class labels
    bboxes
        boxes with normalized coordinates in format [ymin, xmin, ymax, xmax]
    """
    try:
        if isinstance(image_id, bytes):
            image_id = image_id.decode()
        labels = annotations.loc[[image_id]]
    except KeyError:
        class_labels = np.zeros([1], np.int64)
        bboxes = np.zeros([1, 4], np.float32)
        return class_labels, bboxes

    labels['LabelName'] = labels['LabelName'].apply(
        lambda x: class_descriptions[x][0])
    bboxes = labels[['YMin', 'XMin', 'YMax', 'XMax']].values.astype(np.float32)
    class_labels = labels["LabelName"].values.astype(np.int64)
    return class_labels, bboxes
